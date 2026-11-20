"""Prompt text for the LLM agent.

The prompts describe the task and the required output shape. Tool-selection
prompts never contain tool results: the model must call tools to obtain
evidence. Prompts also never contain a scenario's expected_characteristics,
because those are fixture-design notes that would reveal the intended answer.
"""

import json

from models.schemas import SecurityScenario
from tools.langchain_security_tools import TOOL_NAMES


# Fixture design notes are not part of the percept the agent should reason about.
_EXCLUDED_SCENARIO_FIELDS = {"expected_characteristics"}


SYSTEM_PROMPT = f"""You are a defensive security alert triage assistant working on SIMULATED, synthetic data.

Your job is to gather relevant evidence for a security alert and support a safe,
proportionate recommendation for a human analyst.

Rules you must follow:
1. You MUST obtain successful evidence from at least two different tools that are relevant to this scenario before a final decision can be produced. Available tools: {", ".join(TOOL_NAMES)}.
2. You decide which relevant evidence tools to call and what valid arguments to use.
3. When appropriate, request multiple distinct relevant tools in the same response.
4. Base every factual statement on successful tool evidence. Never invent IDs, counts, accounts, addresses or events.
5. A tool result with status other than SUCCESS is not positive factual evidence.
6. You only RECOMMEND. You never block, disable, quarantine, terminate, isolate, delete, contain, attack, or change firewall rules or systems, and you never claim that you did.
7. Consequential actions require human authorization.
8. Do not claim compromise or confirmed impact unless successful tool evidence confirms it.
9. Weak or incomplete evidence must not be over-escalated, and strong evidence on a critical asset must not be under-escalated.
10. Hidden reasoning or chain-of-thought must never be included in the visible response.

During evidence gathering, use tool calls rather than producing the final triage decision.
"""


def build_scenario_message(scenario: SecurityScenario) -> str:
    """Build the first-stage scenario message used for evidence gathering."""

    fields = scenario.model_dump(
        mode="json",
        exclude=_EXCLUDED_SCENARIO_FIELDS,
    )

    return (
        "/no_think\n\n"
        "Analyze this simulated defensive-security alert.\n"
        "Do not produce the final triage decision yet.\n"
        "Select the relevant evidence tools needed for this scenario.\n"
        "When appropriate, request two or more distinct relevant tools "
        "in the same response.\n"
        "At least two distinct relevant tools must complete successfully "
        "before a final decision can be produced.\n"
        "Do not infer facts that an available evidence tool can verify.\n\n"
        + json.dumps(
            fields,
            indent=2,
            sort_keys=True,
        )
    )


def build_tool_correction_message(
    minimum_distinct_tools: int,
    used: tuple[str, ...],
) -> str:
    """One bounded correction when insufficient evidence has been gathered."""

    return (
        "/no_think\n\n"
        "The final decision cannot be produced yet because insufficient "
        "successful tool evidence has been collected.\n"
        f"Required successful distinct relevant tools: "
        f"{minimum_distinct_tools}.\n"
        f"Successful relevant tools so far: "
        f"{', '.join(used) or 'none'}.\n"
        "Do not return the final decision yet.\n"
        "Request the remaining relevant evidence tools now.\n"
        "When appropriate, request multiple distinct relevant tools "
        "in the same response."
    )


def build_compact_decision_message(
    scenario: SecurityScenario,
    compact_evidence: dict[str, object],
) -> str:
    """Build the second-stage prompt after minimum evidence is satisfied."""

    return (
        "/no_think\n\n"
        "The minimum evidence requirement has already been satisfied.\n"
        "No additional evidence tools are available or needed for this stage.\n"
        "Using only the evidence below, return exactly one valid JSON object.\n"
        "Do not use markdown and do not add text outside the JSON object.\n\n"
        "Return exactly these five keys:\n"
        "{\n"
        '  "priority": "LOW|MEDIUM|HIGH|CRITICAL",\n'
        '  "decision": "MONITOR|COLLECT_MORE_EVIDENCE|INVESTIGATE|'
        'ESCALATE_TO_SECURITY_ANALYST",\n'
        '  "uncertainty": "short factual uncertainty statement",\n'
        '  "likely_interpretation": "short evidence-based interpretation",\n'
        '  "recommended_actions": ["safe investigation or review action"]\n'
        "}\n\n"
        "If compromise or impact is not confirmed, state that explicitly "
        "in uncertainty.\n"
        "Recommend only safe investigation, review, monitoring, evidence "
        "collection or human-escalation steps.\n"
        "Do not claim that remediation, containment or any consequential "
        "action was executed.\n"
        "Do not invent facts that are not present in the evidence.\n\n"
        f"Scenario ID: {scenario.scenario_id}\n"
        "Evidence:\n"
        + json.dumps(
            compact_evidence,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def build_compact_decision_repair_message(
    problems: tuple[str, ...],
    compact_evidence: dict[str, object],
) -> str:
    """One bounded repair request for invalid compact-decision JSON."""

    listed = "\n".join(
        f"- {problem}"
        for problem in problems
    )

    return (
        "/no_think\n\n"
        "The compact triage decision was rejected for these reasons:\n"
        f"{listed}\n\n"
        "Return only the corrected JSON object.\n"
        "Use exactly these five keys:\n"
        "priority, decision, uncertainty, likely_interpretation, "
        "recommended_actions.\n"
        "Use only facts present in the evidence below.\n"
        "Do not add markdown or surrounding prose.\n\n"
        "Evidence:\n"
        + json.dumps(
            compact_evidence,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def build_repair_message(
    problems: tuple[str, ...],
) -> str:
    """Legacy bounded repair used by the current orchestrator until Phase 6."""

    listed = "\n".join(
        f"- {problem}"
        for problem in problems
    )

    return (
        "Your answer was rejected for these reasons:\n"
        f"{listed}\n"
        "Reply again with ONLY the corrected JSON object. "
        "Use only tool_call_ids and source_record_ids that "
        "appeared in the tool results above. "
        "Do not call additional tools unless you must."
    )