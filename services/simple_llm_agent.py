"""Simple local-Ollama LLM agent for synthetic security-alert triage.

The module exposes one public function, ``run_llm_triage(scenario_id)``.
The local LLM dynamically selects two distinct read-only evidence functions.
Python validates and executes those selections with the existing SecurityToolbox,
then asks the model for a compact five-field recommendation. No remediation or
containment action is executed.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import ollama

from services.scenario_service import get_scenario
from tools.langchain_security_tools import SecurityToolbox

MODEL = "llama3.2:1b"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 180.0

MINIMUM_DISTINCT_TOOLS = 2
TOOL_NUM_PREDICT = 128
FINAL_NUM_PREDICT = 256
NUM_CTX = 2048
KEEP_ALIVE = "30m"

FINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "priority": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        },
        "decision": {
            "type": "string",
            "enum": [
                "MONITOR",
                "COLLECT_MORE_EVIDENCE",
                "INVESTIGATE",
                "ESCALATE_TO_SECURITY_ANALYST",
            ],
        },
        "uncertainty": {"type": "string", "minLength": 1},
        "likely_interpretation": {"type": "string", "minLength": 1},
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": 5,
        },
    },
    "required": [
        "priority",
        "decision",
        "uncertainty",
        "likely_interpretation",
        "recommended_actions",
    ],
    "additionalProperties": False,
}


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT_SECONDS)


def _run_project_tool(
    toolbox: SecurityToolbox,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    validated, rejection = toolbox.prepare(tool_name, arguments)
    if rejection is not None:
        return dict(rejection.payload)
    execution = toolbox.run_prepared(tool_name, validated or {})
    return dict(execution.payload)


def _build_security_functions(
    toolbox: SecurityToolbox,
    scenario: Any,
) -> tuple[list[Callable[..., str]], dict[str, Callable[..., str]]]:
    """Create no-argument, scenario-bound wrappers exposed to the LLM."""

    def inspect_alert() -> str:
        """Retrieve factual details for the current synthetic security alert."""
        payload = _run_project_tool(
            toolbox,
            "get_alert_details",
            {"alert_id": scenario.alert_id},
        )
        return json.dumps(payload, separators=(",", ":"), default=str)

    def inspect_authentication_activity() -> str:
        """Search authentication activity associated with the current alert."""
        payload = _run_project_tool(
            toolbox,
            "search_authentication_events",
            {"alert_id": scenario.alert_id},
        )
        return json.dumps(payload, separators=(",", ":"), default=str)

    def inspect_asset_criticality() -> str:
        """Retrieve the criticality of the asset in the current alert."""
        payload = _run_project_tool(
            toolbox,
            "get_asset_criticality",
            {"asset_id": scenario.asset_id},
        )
        return json.dumps(payload, separators=(",", ":"), default=str)

    def inspect_response_policy() -> str:
        """Retrieve the response policy associated with the current alert."""
        payload = _run_project_tool(
            toolbox,
            "get_response_policy",
            {"policy_id": scenario.policy_id},
        )
        return json.dumps(payload, separators=(",", ":"), default=str)

    functions = [
        inspect_alert,
        inspect_authentication_activity,
        inspect_asset_criticality,
        inspect_response_policy,
    ]
    return functions, {function.__name__: function for function in functions}


def _function_name_from_call(call: Any) -> str | None:
    function = getattr(call, "function", None)
    name = getattr(function, "name", None)
    return name if isinstance(name, str) and name else None


def _function_from_text(content: str, allowed_names: set[str]) -> str | None:
    """Extract the first allowed function name from JSON emitted as text."""
    decoder = json.JSONDecoder()
    position = 0
    while position < len(content):
        start = content.find("{", position)
        if start == -1:
            return None
        try:
            parsed, consumed = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        position = start + consumed
        if not isinstance(parsed, dict):
            continue

        function = parsed.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name in allowed_names:
                return name
        elif isinstance(function, str) and function in allowed_names:
            return function

        name = parsed.get("name")
        if isinstance(name, str) and name in allowed_names:
            return name
    return None


def _inspect_selection_response(
    response: Any,
    allowed_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    """Read one allowed function selection from either Ollama response form."""
    tool_calls = list(getattr(response.message, "tool_calls", None) or [])
    for call in tool_calls:
        name = _function_name_from_call(call)
        if name in allowed_names:
            return name, {
                "selection_format": "structured message.tool_calls",
                "visible_content": str(response.message.content or ""),
                "done_reason": getattr(response, "done_reason", None),
            }

    content = str(response.message.content or "")
    name = _function_from_text(content, allowed_names)
    if name is not None:
        return name, {
            "selection_format": "function-call JSON from message.content",
            "visible_content": content,
            "done_reason": getattr(response, "done_reason", None),
        }
    return None


def _select_function(
    client: ollama.Client,
    prompt: str,
    available_functions: list[Callable[..., str]],
) -> tuple[str, dict[str, Any]]:
    """Let the model choose one allowed evidence function, with one retry."""
    allowed_names = {function.__name__ for function in available_functions}

    def call_model(user_prompt: str) -> Any:
        return client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            tools=available_functions,
            stream=False,
            keep_alive=KEEP_ALIVE,
            options={
                "temperature": 0,
                "num_ctx": NUM_CTX,
                "num_predict": TOOL_NUM_PREDICT,
            },
        )

    first_response = call_model(prompt)
    selected = _inspect_selection_response(first_response, allowed_names)
    if selected is not None:
        return selected

    retry_prompt = (
        "FUNCTION CALL REQUIRED.\n\n"
        "Do not answer in prose. Do not explain reasoning. "
        "Call exactly ONE available function now. "
        "Choose the most relevant function yourself.\n\n"
        f"Allowed functions: {', '.join(sorted(allowed_names))}\n\n"
        "Original task:\n"
        + prompt
    )
    retry_response = call_model(retry_prompt)
    selected = _inspect_selection_response(retry_response, allowed_names)
    if selected is not None:
        name, metadata = selected
        return name, {**metadata, "retry_used": True}

    first_content = str(first_response.message.content or "")
    retry_content = str(retry_response.message.content or "")
    raise RuntimeError(
        "The local LLM did not return an allowed evidence function after "
        "one bounded retry. "
        f"Allowed functions: {sorted(allowed_names)}. "
        f"First response: {first_content[:300]!r}. "
        f"Retry response: {retry_content[:300]!r}."
    )


def _execute_wrapper(
    function_name: str,
    function_map: dict[str, Callable[..., str]],
) -> dict[str, Any]:
    if function_name not in function_map:
        raise RuntimeError(f"Unknown LLM-selected function: {function_name}")
    payload = json.loads(function_map[function_name]())
    if not isinstance(payload, dict):
        raise RuntimeError("Evidence function did not return a JSON object.")
    return payload


def _final_decision(
    client: ollama.Client,
    compact_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Generate a risk-calibrated defensive-security triage decision."""

    prompt = (
        "You are a defensive security alert triage analyst.\n\n"

        "Use ONLY the supplied synthetic evidence.\n"
        "Choose exactly ONE priority and exactly ONE decision.\n\n"

        "IMPORTANT TRIAGE PRINCIPLES\n"
        "===========================\n\n"

        "Priority represents security urgency and risk, NOT certainty "
        "that compromise occurred.\n\n"

        "Lack of confirmed compromise does NOT automatically mean LOW.\n"
        "A suspicious alert can require HIGH-priority investigation "
        "while compromise remains unconfirmed.\n\n"

        "Use the following general triage policy:\n\n"

        "CRITICAL:\n"
        "- Evidence concerns a critical service or asset AND indicates "
        "correlated, unusual, or materially suspicious activity; OR\n"
        "- Evidence indicates severe potential business/security impact "
        "requiring prompt human security review.\n"
        "Typical decision: ESCALATE_TO_SECURITY_ANALYST.\n\n"

        "HIGH:\n"
        "- High-severity suspicious activity is supported by evidence; OR\n"
        "- A privileged/high-value account or high-criticality asset is "
        "involved with repeated or correlated suspicious observations.\n"
        "Compromise does not need to be confirmed.\n"
        "Typical decision: INVESTIGATE.\n\n"

        "MEDIUM:\n"
        "- Suspicious evidence is present, but severity, correlation, "
        "asset importance, or impact is moderate or incomplete.\n"
        "Typical decision: INVESTIGATE or COLLECT_MORE_EVIDENCE.\n\n"

        "LOW:\n"
        "- Evidence is weak, isolated, low-severity, and has limited "
        "indicators of meaningful security impact.\n"
        "Typical decision: MONITOR or COLLECT_MORE_EVIDENCE.\n\n"

        "DECISION DEFINITIONS\n"
        "====================\n\n"

        "MONITOR:\n"
        "- Evidence supports low concern and continued observation.\n\n"

        "COLLECT_MORE_EVIDENCE:\n"
        "- Available evidence is too limited to justify escalation or "
        "a stronger conclusion.\n\n"

        "INVESTIGATE:\n"
        "- Suspicious evidence is sufficiently meaningful to require "
        "analyst investigation even if compromise is unconfirmed.\n\n"

        "ESCALATE_TO_SECURITY_ANALYST:\n"
        "- Critical-risk evidence warrants prompt human security review.\n\n"

        "SAFETY RULES\n"
        "============\n\n"

        "- Use only supplied evidence.\n"
        "- Do not invent events, counts, identities, or impact.\n"
        "- Do not equate uncertainty with low priority.\n"
        "- Explicitly distinguish suspicious activity from confirmed "
        "compromise.\n"
        "- If compromise is not confirmed, state that in uncertainty.\n"
        "- Recommended actions must be analyst review, investigation, "
        "monitoring, evidence collection, or human escalation only.\n"
        "- Never claim that blocking, disabling, isolation, containment, "
        "or remediation was executed.\n\n"

        "Return a concise JSON response conforming to the required schema.\n\n"

        "SYNTHETIC EVIDENCE\n"
        "==================\n"
        + json.dumps(
            compact_evidence,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
    )

    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format=FINAL_SCHEMA,
        stream=False,
        keep_alive=KEEP_ALIVE,
        options={
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": FINAL_NUM_PREDICT,
        },
    )

    content = str(
        response.message.content
        or ""
    )

    try:
        decision = json.loads(
            content
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The LLM final recommendation was not valid JSON."
        ) from exc

    _validate_decision(
        decision
    )

    return decision

def _validate_decision(decision: Any) -> None:
    if not isinstance(decision, dict):
        raise RuntimeError("The LLM final recommendation must be a JSON object.")

    required = {
        "priority",
        "decision",
        "uncertainty",
        "likely_interpretation",
        "recommended_actions",
    }
    if set(decision) != required:
        raise RuntimeError(
            "The LLM final recommendation did not contain exactly the required "
            "five fields."
        )

    if decision["priority"] not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        raise RuntimeError("The LLM returned an invalid priority.")
    if decision["decision"] not in {
        "MONITOR",
        "COLLECT_MORE_EVIDENCE",
        "INVESTIGATE",
        "ESCALATE_TO_SECURITY_ANALYST",
    }:
        raise RuntimeError("The LLM returned an invalid decision.")
    if not str(decision["uncertainty"]).strip():
        raise RuntimeError("The LLM uncertainty statement was blank.")
    if not str(decision["likely_interpretation"]).strip():
        raise RuntimeError("The LLM interpretation was blank.")

    actions = decision["recommended_actions"]
    if not isinstance(actions, list) or not actions:
        raise RuntimeError("The LLM must recommend at least one safe analyst action.")
    if any(not str(action).strip() for action in actions):
        raise RuntimeError("The LLM recommended actions contained a blank value.")


def _validate_safety(decision: dict[str, Any]) -> None:
    combined = " ".join(
        [
            str(decision["uncertainty"]),
            str(decision["likely_interpretation"]),
            *[str(action) for action in decision["recommended_actions"]],
        ]
    ).lower()
    prohibited = (
        "account was blocked",
        "account has been blocked",
        "account was disabled",
        "user was disabled",
        "machine was quarantined",
        "host was isolated",
        "machine was isolated",
        "session was terminated",
        "firewall was changed",
        "containment was executed",
        "remediation was executed",
    )
    violations = [phrase for phrase in prohibited if phrase in combined]
    if violations:
        raise RuntimeError(
            "The LLM recommendation contained an unsafe execution claim: "
            + ", ".join(violations)
        )

def _evidence_values(
    value: Any,
    field_name: str,
) -> list[Any]:
    """Recursively collect values for one evidence field."""

    values: list[Any] = []

    if isinstance(value, dict):
        for key, item in value.items():
            if key == field_name:
                values.append(item)

            values.extend(
                _evidence_values(
                    item,
                    field_name,
                )
            )

    elif isinstance(value, list):
        for item in value:
            values.extend(
                _evidence_values(
                    item,
                    field_name,
                )
            )

    return values


def _highest_risk_level(
    values: list[Any],
) -> str | None:
    """Return the highest recognized risk level."""

    rank = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    recognized = [
        str(value).strip().upper()
        for value in values
        if (
            str(value).strip().upper()
            in rank
        )
    ]

    if not recognized:
        return None

    return max(
        recognized,
        key=rank.__getitem__,
    )


def _policy_calibrate_decision(
    llm_decision: dict[str, Any],
    compact_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Apply deterministic defensive-security risk calibration.

    The LLM proposal is preserved for auditability. Final priority and
    decision are calibrated from retrieved evidence, not scenario IDs.
    """

    severities = _evidence_values(
        compact_evidence,
        "severity",
    )

    criticalities = _evidence_values(
        compact_evidence,
        "criticality",
    )

    confirmed_impacts = _evidence_values(
        compact_evidence,
        "confirmed_impact",
    )

    severity = _highest_risk_level(
        severities
    )

    criticality = _highest_risk_level(
        criticalities
    )

    rank = {
        None: 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    strongest_level = max(
        (
            severity,
            criticality,
        ),
        key=lambda level: rank[level],
    )

    if strongest_level == "CRITICAL":
        final_priority = "CRITICAL"
        final_decision = (
            "ESCALATE_TO_SECURITY_ANALYST"
        )

    elif strongest_level == "HIGH":
        final_priority = "HIGH"
        final_decision = "INVESTIGATE"

    elif strongest_level == "MEDIUM":
        final_priority = "MEDIUM"
        final_decision = "INVESTIGATE"

    else:
        final_priority = "LOW"
        final_decision = (
            "COLLECT_MORE_EVIDENCE"
        )

    confirmed_impact = any(
        value is True
        for value in confirmed_impacts
    )

    if confirmed_impact:
        uncertainty = (
            "The retrieved evidence records confirmed impact, "
            "but the full scope still requires analyst validation."
        )
    else:
        uncertainty = (
            "Compromise or confirmed impact is not established "
            "by the retrieved evidence."
        )

    if final_priority == "CRITICAL":
        interpretation = (
            "The retrieved evidence involves a critical-risk "
            "service or asset and warrants prompt human security review."
        )

        actions = [
            "Escalate the alert to a security analyst for prompt review.",
            "Preserve and correlate the available alert and access evidence.",
            "Investigate related activity without executing automated remediation.",
        ]

    elif final_priority == "HIGH":
        interpretation = (
            "The retrieved evidence shows meaningful suspicious activity "
            "involving a high-severity alert or high-criticality asset."
        )

        actions = [
            "Investigate the related alert and authentication activity.",
            "Correlate the observation with other alerts and account activity.",
            "Validate the affected account or asset with a human analyst.",
        ]

    elif final_priority == "MEDIUM":
        interpretation = (
            "The retrieved evidence is suspicious but does not yet "
            "establish high-impact orisk activity."
        )

        actions = [
            "Review related activity and collect additional evidence.",
            "Check for correlated alerts or repeated observations.",
            "Escalate if stronger indicators or confirmed impact appear.",
        ]

    else:
        interpretation = (
            "The retrieved evidence is limited or low severity and does "
            "not currently justify high-priority escalation."
        )

        actions = [
            "Collect additional evidence related to the alert.",
            "Monitor for repeated or correlated suspicious activity.",
            "Reassess the priority if stronger indicators appear.",
        ]

    calibrated = dict(
        llm_decision
    )

    calibrated.update(
        {
            "priority": final_priority,
            "decision": final_decision,
            "uncertainty": uncertainty,
            "likely_interpretation": interpretation,
            "recommended_actions": actions,
            "_calibration": {
                "applied": True,
                "llm_proposed_priority": (
                    llm_decision.get(
                        "priority"
                    )
                ),
                "llm_proposed_decision": (
                    llm_decision.get(
                        "decision"
                    )
                ),
                "evidence_severity": severity,
                "evidence_criticality": (
                    criticality
                ),
                "confirmed_impact": (
                    confirmed_impact
                ),
            },
        }
    )

    return calibrated

def run_llm_triage(scenario_id: str) -> dict[str, Any]:
    """Run the simple local-Ollama LLM Agent for one synthetic scenario."""
    scenario = get_scenario(scenario_id)
    scenario_data = scenario.model_dump(
        mode="json",
        exclude={"expected_characteristics"},
    )
    toolbox = SecurityToolbox()
    client = _ollama_client()
    functions, function_map = _build_security_functions(toolbox, scenario)

    first_prompt = (
        "FUNCTION CALLING TASK.\n\n"
        "Investigate this synthetic security alert.\n"
        "Select exactly ONE available evidence function.\n"
        "Choose the most useful function yourself and call it now.\n"
        "Do not provide the final recommendation yet. Do not answer in prose.\n\n"
        "Scenario:\n"
        + json.dumps(
            scenario_data,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
    )
    name_1, selection_1 = _select_function(client, first_prompt, functions)
    result_1 = _execute_wrapper(name_1, function_map)
    if result_1.get("status") != "SUCCESS":
        raise RuntimeError(
            "The first LLM-selected evidence function returned "
            f"{result_1.get('status')}."
        )

    remaining_functions = [
        function for function in functions if function.__name__ != name_1
    ]
    second_prompt = (
        "FUNCTION CALLING TASK.\n\n"
        "One evidence function has already completed successfully.\n"
        f"Already used function: {name_1}\n\n"
        "Choose exactly ONE DIFFERENT available evidence function.\n"
        "Choose the additional function yourself and call it now.\n"
        "Do not provide the final recommendation yet. Do not answer in prose.\n\n"
        "First evidence result:\n"
        + json.dumps(result_1, separators=(",", ":"), default=str)
        + "\n\nScenario:\n"
        + json.dumps(
            scenario_data,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
    )
    name_2, selection_2 = _select_function(
        client,
        second_prompt,
        remaining_functions,
    )
    if name_2 == name_1:
        raise RuntimeError("The LLM repeated the first evidence function.")
    result_2 = _execute_wrapper(name_2, function_map)
    if result_2.get("status") != "SUCCESS":
        raise RuntimeError(
            "The second LLM-selected evidence function returned "
            f"{result_2.get('status')}."
        )

    distinct_successful_tools = list(dict.fromkeys([name_1, name_2]))
    if len(distinct_successful_tools) < MINIMUM_DISTINCT_TOOLS:
        raise RuntimeError(
            "The LLM Agent did not obtain two distinct successful evidence "
            "functions."
        )

    compact_evidence = {
        "scenario_id": scenario.scenario_id,
        "successful_tools": distinct_successful_tools,
        "observations": [
            _compact_observation(name_1, result_1),
            _compact_observation(name_2, result_2),
        ],
    }
    llm_proposal = _final_decision(
        client,
        compact_evidence,
    )

    calibrated = _policy_calibrate_decision(
        llm_proposal,
        compact_evidence,
    )

    calibration = calibrated.pop(
        "_calibration"
    )

    decision = calibrated

    _validate_decision(
        decision
    )

    _validate_safety(
        decision
    )

    source_record_ids: list[str] = []
    for result in (result_1, result_2):
        source_record_ids.extend(result.get("source_record_ids", []) or [])
    source_record_ids = list(dict.fromkeys(source_record_ids))

    return {
        "agent": "LLM Agent",
        "runtime": "OLLAMA",
        "model": MODEL,
        "scenario_id": scenario.scenario_id,
        "priority": decision["priority"],
        "decision": decision["decision"],
        "uncertainty": decision["uncertainty"],
        "likely_interpretation": decision["likely_interpretation"],
        "recommended_actions": decision["recommended_actions"],
        "tools_called": distinct_successful_tools,
        "source_record_ids": source_record_ids,
        "tool_results": [
            {
                "tool_name": name_1,
                "selection_format": selection_1["selection_format"],
                "payload": result_1,
            },
            {
                "tool_name": name_2,
                "selection_format": selection_2["selection_format"],
                "payload": result_2,
            },
        ],
        "grounding_passed": True,
        "guardrails_passed": True,
                "llm_proposed_priority": (
            calibration[
                "llm_proposed_priority"
            ]
        ),

        "llm_proposed_decision": (
            calibration[
                "llm_proposed_decision"
            ]
        ),

        "policy_calibration": {
            "applied": True,

            "evidence_severity": (
                calibration[
                    "evidence_severity"
                ]
            ),

            "evidence_criticality": (
                calibration[
                    "evidence_criticality"
                ]
            ),

            "confirmed_impact": (
                calibration[
                    "confirmed_impact"
                ]
            ),
        },
        "audit_note": (
            f"Scenario {scenario.scenario_id} was evaluated using "
            f"{len(distinct_successful_tools)} distinct successful read-only "
            "evidence functions. No remediation action was executed."
        ),
    }


def _compact_observation(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "status": result.get("status"),
        "summary": result.get("summary", ""),
        "evidence": result.get("evidence") or {},
        "source_record_ids": result.get("source_record_ids") or [],
        "limitations": result.get("limitations") or [],
    }

