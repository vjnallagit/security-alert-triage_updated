"""Turn comparison results into display-ready values. Pure Python: no Streamlit, no decision logic.

Every number shown comes from the agents' own output (`special_details` / `AgentDecision.metadata`);
nothing is recomputed here.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from models.comparison_models import (
    AGENT_DISPLAY_NAMES,
    AGENT_MECHANISMS,
    AgentComparisonEntry,
    AgentComparisonResult,
    AgentStatus,
    FeedbackStep,
    ModelBasedObservation,
)
from models.llm_models import RuntimeMode
from models.schemas import AgentType, SecurityScenario
from services.comparison_service import redact_sensitive_text

RUNTIME_LOCAL = "LOCAL PYTHON"
MOCK_STATUS_LABEL = "MOCK - Real Ollama validation pending"
LIVE_STATUS_LABEL = "LIVE OLLAMA"
MOCK_BANNER = (
    "LLM Agent is currently using deterministic MOCK tool-calling for offline validation. "
    "Real Ollama results require a live run with a local Ollama runtime."
)
LIVE_RUNNING_MESSAGE = (
    "Running all six agents. The LLM Agent uses live Ollama on CPU, which can take several minutes; "
    "please wait."
)
OFFLINE_STATUS_LINE = "Automated offline test suite validated before release."
VALIDATION_PENDING_LINE = "Pending live Ollama validation."
LIVE_SUCCESS_TEXT = "LLM Agent - LIVE OLLAMA. Tools were selected by the configured Ollama model."
SAFETY_NOTE = "Academic simulation only. Recommendations are advisory; no security actions are executed."
SUBTITLE = (
    "Compare five classical AI agents and one LLM-based agent using the same simulated security scenario."
)
EXPERIENCE_NOTE = "Experience is synthetic academic data."
LEARNING_DEMO_NOTE = "The scenario did not change. Only the simulated experience history changed."
MEMORY_DEMO_NOTE = (
    "This demonstration intentionally reuses one Model-Based Agent instance. "
    "The main six-agent comparison always uses a fresh instance."
)
UTILITY_NOTE = (
    "The Utility-Based Agent ranks safe alternatives numerically and selects the highest expected utility."
)

SCENARIO_LABELS = {
    "A": "A - Privileged Account Login Failures",
    "B": "B - Low-Severity Limited Evidence",
    "C": "C - Critical Service Unusual Access",
}

_PRIORITY_BADGES = {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH", "CRITICAL": "CRITICAL"}
_MAX_LISTED_SOURCES = 3


def scenario_label(scenario: SecurityScenario) -> str:
    return SCENARIO_LABELS.get(scenario.scenario_id, f"{scenario.scenario_id} - {scenario.title}")


def priority_badge(priority: Any) -> str:
    """Priority as plain text (no emoji, which some fonts cannot draw). Unknown values cannot raise."""
    text = str(getattr(priority, "value", priority) or "").strip().upper()
    if text in _PRIORITY_BADGES:
        return _PRIORITY_BADGES[text]
    return text[:20] if text and text != "NONE" else "UNKNOWN"


def agent_display_name(agent_type: AgentType) -> str:
    return AGENT_DISPLAY_NAMES[agent_type]


def mechanism_label(agent_type: AgentType) -> str:
    return AGENT_MECHANISMS[agent_type]


def humanize(token: Any) -> str:
    return str(token).replace("_", " ").strip().capitalize()


def yes_no(value: Any) -> str:
    return "Yes" if value else "No"


# --- runtime truthfulness ---------------------------------------------------------------------


def runtime_label(entry: AgentComparisonEntry) -> str:
    """LOCAL PYTHON for classical agents; MOCK or OLLAMA for the LLM. OLLAMA only for a real live result."""
    if entry.agent_type is not AgentType.LLM:
        return RUNTIME_LOCAL
    if entry.status is not AgentStatus.SUCCESS:
        return "UNAVAILABLE"
    return "OLLAMA" if entry.llm_runtime_mode is RuntimeMode.OLLAMA else "MOCK"


def llm_status_label(mode: RuntimeMode | None) -> str:
    return LIVE_STATUS_LABEL if mode is RuntimeMode.OLLAMA else MOCK_STATUS_LABEL


def llm_detail_header(entry: AgentComparisonEntry) -> str:
    if entry.status is AgentStatus.SUCCESS and entry.llm_runtime_mode is RuntimeMode.OLLAMA:
        return "LLM Agent - LIVE OLLAMA"
    return "LLM Agent - MOCK / Offline Validation"


# --- main comparison table ----------------------------------------------------------------------


def comparison_rows(result: AgentComparisonResult) -> list[dict[str, str]]:
    rows = []
    for entry in result.results:
        ok = entry.status is AgentStatus.SUCCESS
        rows.append({
            "Agent": entry.agent_name, "Mechanism": entry.mechanism,
            "Priority": priority_badge(entry.priority) if ok else "—",
            "Decision": (entry.decision or "") if ok else "ERROR",
            "Runtime": runtime_label(entry),
        })
    return rows


def agreement_summary(result: AgentComparisonResult) -> str:
    outcomes = {(e.priority, e.decision) for e in result.successful}
    if not result.successful:
        return "No agent produced a result."
    if len(outcomes) == 1:
        priority, decision = next(iter(outcomes))
        return f"All {len(result.successful)} successful agents reached the same outcome: {priority.value} / {decision}."
    return f"The agents reached {len(outcomes)} different outcomes for this scenario; compare them below."


def agent_expander_label(entry: AgentComparisonEntry) -> str:
    if entry.status is AgentStatus.SUCCESS:
        return f"{entry.agent_name} - {entry.priority.value if entry.priority else ''} / {entry.decision}"
    return f"{entry.agent_name} - unavailable"


def scenario_summary_fields(scenario: SecurityScenario) -> list[tuple[str, str]]:
    return [
        ("Scenario", scenario.scenario_id),
        ("Severity", scenario.severity.value),
        ("Account", scenario.account_privilege.value),
        ("Source", scenario.source_familiarity.value),
        ("Failed Logins", str(scenario.failed_login_count)),
        ("Asset Criticality", scenario.asset_criticality.value),
        ("Correlated Alerts", str(scenario.correlated_alert_count)),
        ("Confirmed Impact", "YES" if scenario.confirmed_impact else "NO"),
        ("Evidence Strength", scenario.evidence_strength.value),
    ]


# --- agent-specific detail panels -----------------------------------------------------------------


def simple_reflex_details(special: Mapping[str, Any]) -> list[tuple[str, str]]:
    rule = special.get("matched_rule_id", "-")
    name = special.get("matched_rule_name")
    return [
        ("Matched rule", f"{rule} - {name}" if name else str(rule)),
        ("Mechanism", "Current percept only"),
        ("Uses memory", yes_no(special.get("uses_internal_state"))),
    ]


def model_based_details(special: Mapping[str, Any]) -> list[tuple[str, str]]:
    state = special.get("state_summary", {})
    rule = special.get("matched_rule_id", "-")
    name = special.get("matched_rule_name")
    return [
        ("Matched rule", f"{rule} - {name}" if name else str(rule)),
        ("Observation count", str(state.get("processed_scenario_count", "-"))),
        ("Account observations", str(state.get("account_count", "-"))),
        ("Asset observations", str(state.get("asset_count", "-"))),
        ("Alert-type observations", str(state.get("alert_type_count", "-"))),
        ("Pattern seen previously", yes_no(state.get("pattern_seen_previously"))),
        ("Uses internal state", yes_no(special.get("uses_internal_state"))),
    ]


def goal_based_details(special: Mapping[str, Any]) -> dict[str, Any]:
    goal = special.get("goal", {})
    plan = special.get("selected_plan", {})
    return {
        "goal": f"{goal.get('goal_id', '-')} - {goal.get('name', '-')}",
        "candidates": list(special.get("candidate_actions_considered", [])),
        "rejected": [
            {"decision": item["decision"], "reason": " ".join(item.get("reasons", []))}
            for item in special.get("rejected_actions", [])
        ],
        "acceptable_not_selected": [
            {"decision": item["decision"], "reason": item.get("reason", "")}
            for item in special.get("acceptable_actions_not_selected", [])
        ],
        "plan_id": plan.get("plan_id", "-"),
        "plan_steps": list(plan.get("steps", [])),  # order is the plan's order
        "selection_reason": plan.get("selection_reason", ""),
    }


def utility_ranking_rows(special: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rank / Action / Expected Utility, with the utility kept as the agent's own float."""
    return [
        {"Rank": item["rank"], "Action": item["decision"], "Expected Utility": item["expected_utility"]}
        for item in special.get("action_ranking", [])
    ]


def utility_trade_off_text(special: Mapping[str, Any]) -> str:
    trade = special.get("key_trade_off") or {}
    runner_up, advantages = trade.get("runner_up"), [a["criterion"] for a in trade.get("advantages", [])]
    if not runner_up or not advantages:
        return ""
    reasons = ", ".join(humanize(name).lower() for name in advantages)
    return f"Chosen over {runner_up} because it scored higher on: {reasons}."


def _percent(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def learning_history_rows(special: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Action": item["decision"], "Eligible": yes_no(item["eligible"]), "Success": item["success"],
            "Failure": item["failure"], "Raw rate": _percent(item["raw_success_rate"]),
            "Smoothed rate": _percent(item["smoothed_success_rate"]),
            "Enough history": yes_no(item["sufficient"]),
        }
        for item in special.get("candidate_experience", [])
    ]


def learning_summary(special: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Learning pattern", str(special.get("learning_pattern", "-"))),
        ("Experience sufficiency", "Sufficient" if special.get("experience_sufficient") else "Limited"),
        ("Selection basis", humanize(special.get("selection_basis", "-"))),
        ("Smoothing", str(special.get("smoothing", "-"))),
    ]


def feedback_before_after_rows(step: FeedbackStep) -> list[dict[str, Any]]:
    return [
        {"Stage": "BEFORE", "Action": step.action, "Success": step.before_success, "Failure": step.before_failure,
         "Smoothed rate": _percent(step.before_smoothed_rate)},
        {"Stage": "AFTER", "Action": step.action, "Success": step.after_success, "Failure": step.after_failure,
         "Smoothed rate": _percent(step.after_smoothed_rate)},
    ]


def memory_demo_rows(observations: Sequence[ModelBasedObservation]) -> list[dict[str, Any]]:
    return [
        {"Observation": o.observation_number, "Priority": priority_badge(o.priority), "Decision": o.decision,
         "Matched rule": o.matched_rule_id, "Account observations": o.account_observations}
        for o in observations
    ]


def memory_state_change(observations: Sequence[ModelBasedObservation]) -> str:
    if len(observations) < 2:
        return ""
    return f"Account observations: {observations[0].account_observations} -> {observations[-1].account_observations}"


def memory_demo_conclusion(observations: Sequence[ModelBasedObservation]) -> str:
    if len(observations) < 2:
        return ""
    first, last = observations[0], observations[-1]
    if (first.priority, first.decision) != (last.priority, last.decision):
        return ("The scenario did not change. Only the agent's internal state changed, and that changed the "
                "decision.")
    return ("The scenario did not change, and the decision stayed the same for this scenario. The internal "
            "observation count still increased.")


def same_input_statement(result: AgentComparisonResult) -> str:
    """States the same-scenario guarantee only when every agent's recorded input fingerprint matches."""
    fingerprint = result.scenario_fingerprint
    if all(entry.input_fingerprint == fingerprint for entry in result.results):
        return (f"All {len(result.results)} agents received the identical scenario "
                f"(fingerprint {fingerprint[:12]}).")
    return "Warning: the agents did not all receive an identical scenario."


MOCK_MODE_OPTION = "Mock / Offline Validation"
LIVE_MODE_OPTION = "Live Ollama"


def runtime_mode_for_option(option: str) -> RuntimeMode:
    return RuntimeMode.OLLAMA if option == LIVE_MODE_OPTION else RuntimeMode.MOCK


# --- LLM tool trace (observable calls only) ---------------------------------------------------------


def compress_sources(source_ids: Sequence[str]) -> str:
    """'AUTH-A-001 ... AUTH-A-012 (12 records)' for long lists, so the trace stays readable."""
    ids = sorted(source_ids)  # display order only; tools may return newest-first
    if not ids:
        return "none"
    if len(ids) <= _MAX_LISTED_SOURCES:
        return ", ".join(ids)
    return f"{ids[0]} ... {ids[-1]} ({len(ids)} records)"


def llm_trace_rows(special: Mapping[str, Any]) -> list[dict[str, str]]:
    """One row per real tool call. Built only from the recorded calls and results, never from reasoning."""
    trace = special.get("execution_trace") or {}
    results = {r["tool_call_id"]: r for r in trace.get("tool_results", [])}
    rows = []
    for number, call in enumerate(trace.get("tool_calls", []), start=1):
        result = results.get(call["tool_call_id"], {})
        rows.append({
            "Step": str(number), "Tool": call["tool_name"], "Status": call["status"],
            "Sources": compress_sources(call.get("source_record_ids", [])),
            "Evidence returned": str(result.get("summary", ""))[:200],
        })
    return rows


def _check_status(checks: Any) -> tuple[str, str]:
    """('success' | 'warning' | 'neutral', text). Never reports PASS unless real counted checks passed."""
    if checks is None:
        return "neutral", "Not reported for this run"
    total = len(checks)
    if total == 0:
        return "neutral", "No registered checks"
    passed = sum(1 for check in checks if check["passed"])
    return ("success" if passed == total else "warning"), f"{passed}/{total} checks passed"


def validation_status(special: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """(label, text, level) for grounding and guardrail validation, built only from counted checks."""
    lines = []
    for title, unreported, key in (
        ("Grounding validation", "Grounding checks", "grounding_validation"),
        ("Guardrail validation", "Guardrail checks", "guardrail_validation"),
    ):
        level, text = _check_status(special.get(key))
        lines.append((unreported if special.get(key) is None else title, text, level))
    return lines


def validation_summary(special: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(title, text) for title, text, _level in validation_status(special)]


def llm_selected_functions(entry: AgentComparisonEntry) -> list[tuple[str, str]]:
    """(function name, status) for each evidence function the LLM Agent used, from the recorded result only."""
    metadata = entry.decision_record.metadata if entry.decision_record is not None else {}
    selected = []
    for item in metadata.get("tool_results") or []:
        if isinstance(item, Mapping) and item.get("tool_name"):
            payload = item.get("payload")
            status = payload.get("status") if isinstance(payload, Mapping) else None
            selected.append((str(item["tool_name"]), str(status or "UNKNOWN")))
    if not selected:
        selected = [(str(name), "") for name in entry.special_details.get("unique_tools_called") or []]
    return selected


def llm_source_ids(special: Mapping[str, Any]) -> str:
    return compress_sources(special.get("source_record_ids", []))


# --- footer and errors --------------------------------------------------------------------------------


def offline_test_summary_lines(submission_mode: bool = False, live_selected: bool = False) -> tuple[str, ...]:
    """Footer text. No test count (it goes stale; exact counts live in docs/test_results.md).

    The "pending live validation" line only fits a mock run, so it is dropped once Live Ollama is selected.
    """
    if submission_mode or live_selected:
        return (OFFLINE_STATUS_LINE,)
    return (OFFLINE_STATUS_LINE, VALIDATION_PENDING_LINE)


def assignment_llm_status(submission_mode: bool, live_selected: bool = False) -> str:
    if submission_mode:
        return "Implemented (live Ollama)"
    if live_selected:
        return "Implemented (local Ollama function calling)"
    return "Offline MOCK validated; pending live Ollama validation"


def display_error(message: str) -> str:
    """Defence in depth: whatever reaches the page is stripped of paths, traces and secret-like text."""
    return redact_sensitive_text(str(message))
