"""DEVELOPMENT / TEST ONLY: a scripted stand-in for a tool-calling chat model.

This is NOT a language model and it is NOT part of the submission runtime. It exists so the orchestrator,
grounding and guardrails can be exercised, and the UI demonstrated, without Ollama. It validates
orchestration plumbing only, never real tool-calling ability. Its `runtime_mode` is MOCK, so its results can
never be mistaken for a live run.

Loading rules (enforced by services/llm_runtime.py, not here):
  * only when LLM_RUNTIME_MODE=MOCK and ENABLE_MOCK_LLM=true,
  * never when SUBMISSION_MODE=true, and never automatically or as a fallback for a failed LIVE run.
The unit tests import it through tests/fakes/fake_tool_calling_model.py.

Each script step is one of: an AIMessage, a callable(messages) -> AIMessage (used to build answers from
the REAL tool results already in the conversation), or an exception instance to raise.
"""

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from agents.llm_agent import LLMAgent
from config.llm_config import OllamaConfig
from models.llm_models import RuntimeMode
from models.schemas import SecurityScenario

__all__ = [
    "FAKE_MODEL_NAME", "MOCK_OUTCOMES", "ScriptedChatModel", "answer_step", "build_answer", "happy_path_model",
    "mock_llm_agent_factory", "standard_calls", "tool_calls_message", "tool_results",
]

FAKE_MODEL_NAME = "fake-tool-calling-model"

# What each scripted scenario "decides". Mock outcomes only; they say nothing about a real model.
MOCK_OUTCOMES = {
    "A": ("HIGH", "INVESTIGATE"),
    "B": ("LOW", "COLLECT_MORE_EVIDENCE"),
    "C": ("CRITICAL", "ESCALATE_TO_SECURITY_ANALYST"),
}

_ACTIONS = {
    "INVESTIGATE": [
        "Open a security investigation and review authentication events for the targeted account.",
        "Validate whether the source is expected.",
        "Review correlated alerts and assess the affected asset.",
    ],
    "COLLECT_MORE_EVIDENCE": [
        "Gather additional evidence about the alert before deciding on escalation.",
        "Review authentication events for the affected account.",
    ],
    "ESCALATE_TO_SECURITY_ANALYST": [
        "Escalate to a security analyst or the SOC for review.",
        "Review correlated alerts and authentication events together.",
        "Request authorized containment review only if evidence later justifies it.",
    ],
}


class ScriptedChatModel:
    runtime_mode = RuntimeMode.MOCK
    model_name = FAKE_MODEL_NAME

    def __init__(self, steps: Sequence[Any]) -> None:
        self._steps = list(steps)
        self.bound_tool_names: tuple[str, ...] = ()
        self.invocations: list[list[Any]] = []

    def bind_tools(self, tools: Sequence[Any]) -> "ScriptedChatModel":
        self.bound_tool_names = tuple(tool.name for tool in tools)
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.invocations.append(list(messages))
        if not self._steps:
            raise AssertionError("the scripted model was invoked more times than the script allows")
        step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step(messages) if callable(step) else step

    @property
    def call_count(self) -> int:
        return len(self.invocations)


def tool_calls_message(*calls: tuple[str, Any] | tuple[str, Any, str]) -> AIMessage:
    """An AIMessage requesting the given (name, args[, id]) calls in one turn."""
    entries = [
        {"name": call[0], "args": call[1], "id": call[2] if len(call) == 3 else f"call_{n}", "type": "tool_call"}
        for n, call in enumerate(calls, start=1)
    ]
    return AIMessage(content="", tool_calls=entries)


def standard_calls(scenario: SecurityScenario) -> list[tuple[str, dict[str, Any]]]:
    """A sensible four-tool sequence for any scenario, using its own IDs."""
    return [
        ("get_alert_details", {"alert_id": scenario.alert_id}),
        ("search_authentication_events", {"alert_id": scenario.alert_id}),
        ("get_asset_criticality", {"asset_id": scenario.asset_id}),
        ("get_response_policy", {"policy_id": scenario.policy_id}),
    ]


def tool_results(messages: Sequence[Any]) -> list[tuple[ToolMessage, dict[str, Any]]]:
    """Every ToolMessage in the conversation with its parsed JSON payload."""
    return [(m, json.loads(m.content)) for m in messages if isinstance(m, ToolMessage)]


def _facts(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    evidence = payload.get("evidence") or {}
    if tool_name == "search_authentication_events":
        return {"failed_login_count": evidence.get("failed_login_count")}
    if tool_name == "get_asset_criticality":
        return {"asset_criticality": evidence.get("criticality")}
    return {}


def build_answer(
    messages: Sequence[Any], scenario: SecurityScenario, *, priority: str | None = None,
    decision: str | None = None, uncertainty: str | None = None,
) -> dict[str, Any]:
    """A well-formed final answer built ONLY from the real tool results in the conversation."""
    default_priority, default_decision = MOCK_OUTCOMES[scenario.scenario_id]
    decision = decision or default_decision
    evidence, called, sources = [], [], []
    for message, payload in tool_results(messages):
        name = message.name or payload["tool_name"]
        called.append(name)
        if payload["status"] != "SUCCESS":
            continue
        ids = list(payload["source_record_ids"])
        sources.extend(ids)
        evidence.append({
            "claim": payload["summary"], "tool_name": name, "tool_call_id": message.tool_call_id,
            "source_record_ids": ids, "is_absence_observation": False,
            "asserted_facts": {k: v for k, v in _facts(name, payload).items() if v is not None},
        })
    return {
        "priority": priority or default_priority, "decision": decision, "evidence": evidence,
        "uncertainty": uncertainty or "Compromise is not confirmed; the evidence shows possible risk only.",
        "likely_interpretation": "Possible suspicious activity that a human analyst should review.",
        "recommended_actions": _ACTIONS.get(decision, _ACTIONS["INVESTIGATE"]),
        "audit_note": "Recommendation only; no action was executed. Human authorization is required for any response.",
        "tools_called": list(dict.fromkeys(called)), "source_record_ids": list(dict.fromkeys(sources)),
    }


def answer_step(
    scenario: SecurityScenario, *, mutate: Callable[[dict[str, Any]], None] | None = None, **overrides: Any,
) -> Callable[[Sequence[Any]], AIMessage]:
    """A script step that answers from the tool results, optionally mutated to inject a specific defect."""
    def step(messages: Sequence[Any]) -> AIMessage:
        answer = build_answer(messages, scenario, **overrides)
        if mutate is not None:
            mutate(answer)
        return AIMessage(content=json.dumps(answer))
    return step


def happy_path_model(scenario: SecurityScenario, *, parallel: bool = True) -> ScriptedChatModel:
    """Call all four tools (together or one per turn) and then answer."""
    calls = standard_calls(scenario)
    turns = [tool_calls_message(*calls)] if parallel else [tool_calls_message(call) for call in calls]
    return ScriptedChatModel([*turns, answer_step(scenario)])


def mock_llm_agent_factory(scenario: SecurityScenario) -> LLMAgent:
    """The LLM Agent wired to the scripted model, one tool call per turn so each call is visible in the trace."""
    return LLMAgent(model=happy_path_model(scenario, parallel=False), config=OllamaConfig())
