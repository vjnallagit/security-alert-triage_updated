"""UI formatter tests (UI-01..UI-12) and functional tests that drive the real Streamlit app with AppTest."""

import hashlib
from typing import Any

import pytest

import ui.formatters as fmt
from agents import GoalBasedAgent, UtilityBasedAgent
from models.comparison_models import (
    AGENT_ORDER,
    AgentStatus,
    FeedbackStep,
    ModelBasedObservation,
)
from models.llm_models import LLMExecutionTrace, RuntimeMode
from models.learning_models import FeedbackOutcome
from models.schemas import AgentType, Priority
from services.comparison_service import (
    AgentComparisonService,
    LearningFeedbackDemo,
    ModelBasedMemoryDemo,
    special_details_for,
)
from services.data_loader import DATA_DIR
from services.scenario_service import get_scenario
from tests.app_helpers import (
    REQUIRED_HEADINGS,
    all_text,
    comparison_rows,
    expander,
    expander_text,
    run_all_agents,
    run_app,
    select_scenario,
)

from tests.fakes.service import mock_service

# Development/offline settings for every test here: MOCK runtime, explicitly enabled, no live integration.
pytestmark = pytest.mark.usefixtures("dev_mock_environment")

EXPECTED = {
    "A": ("HIGH", "INVESTIGATE"),
    "B": ("LOW", "COLLECT_MORE_EVIDENCE"),
    "C": ("CRITICAL", "ESCALATE_TO_SECURITY_ANALYST"),
}
AGENT_LABELS = ["Simple Reflex", "Model-Based Reflex", "Goal-Based", "Utility-Based", "Learning Agent", "LLM Agent"]


def result_for(scenario_id: str = "A"):
    return mock_service().compare(get_scenario(scenario_id))


# ------------------------------------------------------------------------------ UI helper tests


def test_ui_01_priority_formatting_supports_all_four_priorities() -> None:
    badges = {p: fmt.priority_badge(p) for p in Priority}
    assert len(set(badges.values())) == 4
    for priority, badge in badges.items():
        assert priority.value in badge  # text, not colour alone
        assert fmt.priority_badge(priority.value) == badge and fmt.priority_badge(priority.value.lower()) == badge
    assert all(badge == priority.value for priority, badge in badges.items())  # plain text, no emoji glyph


@pytest.mark.parametrize("value", [None, "", "  ", "URGENT", 7, object(), "x" * 500, ["HIGH"]])
def test_ui_02_unknown_priority_cannot_crash_formatting(value: Any) -> None:
    badge = fmt.priority_badge(value)
    assert isinstance(badge, str) and badge and len(badge) < 40


def test_ui_03_agent_display_names_are_correct() -> None:
    assert [fmt.agent_display_name(t) for t in AGENT_ORDER] == AGENT_LABELS


def test_ui_04_mechanism_labels_are_correct() -> None:
    assert [fmt.mechanism_label(t) for t in AGENT_ORDER] == [
        "Current percept + condition/action rules",
        "Current percept + internal state + rules",
        "Goal evaluation + defensive plan selection",
        "Expected utility + safe action ranking",
        "Historical outcome feedback + adaptation",
        "LangChain + dynamic evidence tool calling",
    ]


def test_ui_05_utility_ranking_formatter_preserves_numeric_values() -> None:
    scenario = get_scenario("A")
    metadata = UtilityBasedAgent().decide(scenario).metadata
    rows = fmt.utility_ranking_rows({"action_ranking": metadata["action_ranking"]})
    assert [r["Expected Utility"] for r in rows] == [i["expected_utility"] for i in metadata["action_ranking"]]
    assert all(isinstance(r["Expected Utility"], float) for r in rows)
    assert [r["Rank"] for r in rows] == [1, 2, 3, 4] and [r["Action"] for r in rows][0] == "INVESTIGATE"
    assert [r["Expected Utility"] for r in rows] == sorted((r["Expected Utility"] for r in rows), reverse=True)
    assert f"{rows[0]['Expected Utility']:.4f}" == "0.7812"


def test_ui_06_goal_steps_remain_ordered() -> None:
    metadata = GoalBasedAgent().decide(get_scenario("A")).metadata
    details = fmt.goal_based_details(metadata)
    assert details["plan_steps"] == metadata["selected_plan"]["steps"] and len(details["plan_steps"]) >= 2
    assert details["goal"].startswith("GB-G01 - Safe and proportionate")
    assert details["candidates"] == ["MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"]
    assert {r["decision"] for r in details["rejected"]} >= {"MONITOR", "COLLECT_MORE_EVIDENCE"}


def test_ui_07_learning_formatter_exposes_before_and_after_correctly() -> None:
    demo, scenario = LearningFeedbackDemo(), get_scenario("A")
    try:
        step = demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
    finally:
        demo.close()
    before, after = fmt.feedback_before_after_rows(step)
    assert (before["Stage"], after["Stage"]) == ("BEFORE", "AFTER")
    assert (before["Success"], before["Failure"]) == (step.before_success, step.before_failure)
    assert (after["Success"], after["Failure"]) == (step.after_success, step.after_failure)
    assert after["Failure"] == before["Failure"] + 1 and before["Action"] == after["Action"] == step.action
    assert before["Smoothed rate"].endswith("%") and before["Smoothed rate"] != after["Smoothed rate"]


def test_ui_07_learning_history_rows_expose_counts_and_rates() -> None:
    special = special_details_for(AgentType.LEARNING, result_for("A").entry(AgentType.LEARNING).decision_record)
    rows = fmt.learning_history_rows(special)
    assert [r["Action"] for r in rows] == ["MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"]
    investigate = next(r for r in rows if r["Action"] == "INVESTIGATE")
    assert (investigate["Success"], investigate["Failure"]) == (8, 2) and investigate["Raw rate"] == "80.0%"
    assert investigate["Smoothed rate"] == "75.0%" and investigate["Eligible"] == "Yes"
    assert ("Experience sufficiency", "Sufficient") in fmt.learning_summary(special)


def test_ui_08_llm_trace_formatter_uses_observable_calls_only() -> None:
    entry = result_for("A").entry(AgentType.LLM)
    rows = fmt.llm_trace_rows(entry.special_details)
    recorded = entry.special_details["execution_trace"]["tool_calls"]
    assert [r["Tool"] for r in rows] == [c["tool_name"] for c in recorded] == [
        "get_alert_details", "search_authentication_events", "get_asset_criticality", "get_response_policy"]
    assert [r["Status"] for r in rows] == ["SUCCESS"] * 4 and [r["Step"] for r in rows] == ["1", "2", "3", "4"]
    assert rows[0]["Sources"] == "ALERT-A-001" and rows[2]["Sources"] == "ASSET-IDP-01"
    assert rows[1]["Sources"].startswith("AUTH-A-001 ... AUTH-A-012") and "(12 records)" in rows[1]["Sources"]
    assert all(r["Evidence returned"] for r in rows)
    assert fmt.llm_source_ids(entry.special_details).count("records") <= 1
    assert all(label.endswith("checks passed") for _, label in fmt.validation_summary(entry.special_details))


def test_ui_09_no_chain_of_thought_field_exists() -> None:
    forbidden = ("reasoning", "thought", "chain", "scratch", "think", "rationale")
    entry = result_for("A").entry(AgentType.LLM)
    keys = {key for row in fmt.llm_trace_rows(entry.special_details) for key in row}
    keys |= set(entry.special_details) | set(entry.special_details["execution_trace"])
    keys |= set(LLMExecutionTrace.model_fields)
    for call in entry.special_details["execution_trace"]["tool_calls"]:
        keys |= set(call)
    assert not [k for k in keys if any(word in k.lower() for word in forbidden)]


def test_ui_10_mock_badge_clearly_says_mock() -> None:
    assert fmt.llm_status_label(RuntimeMode.MOCK) == "MOCK - Real Ollama validation pending"
    entry = result_for("A").entry(AgentType.LLM)
    assert fmt.runtime_label(entry) == "MOCK" and "MOCK" in fmt.llm_detail_header(entry)
    assert "OLLAMA" not in fmt.runtime_label(entry).upper() and "LIVE" not in fmt.llm_detail_header(entry)
    assert fmt.MOCK_BANNER.startswith("LLM Agent is currently using deterministic MOCK tool-calling")
    assert "live run" in fmt.MOCK_BANNER and "local Ollama runtime" in fmt.MOCK_BANNER


def test_ui_11_live_badge_says_ollama_only_for_a_live_result() -> None:
    result = result_for("A")
    mock_entry = result.entry(AgentType.LLM)
    assert fmt.runtime_label(mock_entry) == "MOCK"
    live_entry = mock_entry.model_copy(update={"llm_runtime_mode": RuntimeMode.OLLAMA})
    assert fmt.runtime_label(live_entry) == "OLLAMA" and fmt.llm_detail_header(live_entry) == "LLM Agent - LIVE OLLAMA"
    assert fmt.llm_status_label(RuntimeMode.OLLAMA) == "LIVE OLLAMA"
    failed = AgentComparisonService(llm_mode=RuntimeMode.OLLAMA).compare(get_scenario("A")).entry(AgentType.LLM)
    assert failed.status is AgentStatus.ERROR and fmt.runtime_label(failed) == "UNAVAILABLE"
    for entry in result.results:
        if entry.agent_type is not AgentType.LLM:
            assert fmt.runtime_label(entry) == "LOCAL PYTHON"


def test_ui_12_error_messages_redact_paths_and_stack_traces() -> None:
    nasty = 'Traceback (most recent call last):\n  File "C:\\Users\\me\\app.py", line 9\nOLLAMA_API_KEY=abc123 /home/me/x'
    shown = fmt.display_error(nasty)
    for leaked in ("Traceback", "C:\\", "app.py", "abc123", "/home/me"):
        assert leaked not in shown


def test_comparison_rows_have_the_required_columns_for_all_six_agents() -> None:
    rows = fmt.comparison_rows(result_for("C"))
    assert [r["Agent"] for r in rows] == AGENT_LABELS
    assert set(rows[0]) == {"Agent", "Mechanism", "Priority", "Decision", "Runtime"}
    assert {r["Decision"] for r in rows} == {"ESCALATE_TO_SECURITY_ANALYST"}
    assert [r["Runtime"] for r in rows] == ["LOCAL PYTHON"] * 5 + ["MOCK"]


def test_agreement_and_same_input_statements() -> None:
    result = result_for("A")
    assert "same outcome: HIGH / INVESTIGATE" in fmt.agreement_summary(result)
    assert "identical scenario" in fmt.same_input_statement(result) and result.scenario_fingerprint[:12] in fmt.same_input_statement(result)
    tampered = result.model_copy(update={"scenario_fingerprint": "0" * 64})
    assert fmt.same_input_statement(tampered).startswith("Warning")


def test_scenario_summary_fields_for_scenario_a() -> None:
    fields = dict(fmt.scenario_summary_fields(get_scenario("A")))
    assert fields["Scenario"] == "A" and fields["Severity"] == "HIGH" and fields["Account"] == "PRIVILEGED"
    assert fields["Source"] == "UNFAMILIAR" and fields["Failed Logins"] == "12"
    assert fields["Asset Criticality"] == "HIGH" and fields["Confirmed Impact"] == "NO"


def test_memory_demo_formatters() -> None:
    demo, scenario = ModelBasedMemoryDemo(), get_scenario("A")
    demo.observe(scenario)
    demo.observe(scenario)
    rows = fmt.memory_demo_rows(demo.observations)
    assert [r["Decision"] for r in rows] == ["INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"]
    assert fmt.memory_state_change(demo.observations) == "Account observations: 1 -> 2"
    assert "changed the decision" in fmt.memory_demo_conclusion(demo.observations)
    assert fmt.memory_state_change(demo.observations[:1]) == ""


def test_footer_has_no_hard_coded_count_and_states_validation_is_pending() -> None:
    baseline, pending = fmt.offline_test_summary_lines()
    assert baseline == "Automated offline test suite validated before release."
    assert pending == "Pending live Ollama validation."
    assert not any(ch.isdigit() for ch in baseline)  # exact counts live in docs/test_results.md
    assert "All LLM tests passed" not in baseline + pending
    assert fmt.offline_test_summary_lines(submission_mode=True) == (baseline,)  # no pending wording once submitted


def test_scenario_labels() -> None:
    assert [fmt.scenario_label(get_scenario(s)) for s in "ABC"] == [
        "A - Privileged Account Login Failures", "B - Low-Severity Limited Evidence",
        "C - Critical Service Unusual Access"]


def test_formatters_module_does_not_import_streamlit() -> None:
    import subprocess
    import sys

    code = "import sys, ui.formatters, services.comparison_service; assert 'streamlit' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=fmt.__file__.rsplit("ui", 1)[0], timeout=120)
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------------------------ the real app (AppTest)


def test_app_loads_with_scenario_selector_summary_and_mock_banner() -> None:
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "Security Alert Triage & Prioritization Assistant"
    assert at.sidebar.selectbox(key="scenario_id").options == [
        "A - Privileged Account Login Failures", "B - Low-Severity Limited Evidence", "C - Critical Service Unusual Access"]
    assert any(w.value == fmt.MOCK_BANNER for w in at.warning)
    text = all_text(at)
    for label in ("Severity", "Failed Logins", "Asset Criticality", "Confirmed Impact", "Evidence Strength"):
        assert label in text
    assert fmt.SAFETY_NOTE in text and fmt.SUBTITLE in text
    assert any("Press **Run All Agents**" in i.value for i in at.info)


def test_run_all_agents_shows_six_rows_and_a_details_expander_per_agent() -> None:
    at = run_all_agents(run_app())
    assert not at.exception
    rows = comparison_rows(at)
    assert [r["Agent"] for r in rows] == AGENT_LABELS
    assert [r["Runtime"] for r in rows] == ["LOCAL PYTHON"] * 5 + ["MOCK"]
    assert all(r["Priority"].endswith("HIGH") and r["Decision"] == "INVESTIGATE" for r in rows)
    for label in AGENT_LABELS:
        block = expander(at, label + " - ")
        text = expander_text(block)
        for heading in REQUIRED_HEADINGS:
            assert heading in text, (label, heading)


@pytest.mark.parametrize("scenario_id", ["B", "C"])
def test_scenarios_b_and_c_compare_correctly_in_the_ui(scenario_id: str) -> None:
    at = run_all_agents(select_scenario(run_app(), scenario_id))
    priority, decision = EXPECTED[scenario_id]
    rows = comparison_rows(at)
    assert len(rows) == 6 and all(r["Priority"].endswith(priority) and r["Decision"] == decision for r in rows)
    assert fmt.MOCK_BANNER in [w.value for w in at.warning]


def test_results_for_another_scenario_are_never_shown_under_the_current_selection() -> None:
    at = run_all_agents(run_app())
    at = select_scenario(at, "C")
    assert not [e for e in at.expander if e.label.startswith("Simple Reflex - ")]
    assert any("Press **Run All Agents**" in i.value for i in at.info)
    at = run_all_agents(at)
    assert all(r["Decision"] == "ESCALATE_TO_SECURITY_ANALYST" for r in comparison_rows(at))


def test_agent_specific_details_are_displayed() -> None:
    at = run_all_agents(run_app())
    simple = expander_text(expander(at, "Simple Reflex - "))
    assert "Matched rule:" in simple and "Current percept only" in simple and "Uses memory:** No" in simple
    model_based = expander_text(expander(at, "Model-Based Reflex - "))
    assert "Observation count:** 1" in model_based and "Uses internal state:** Yes" in model_based
    goal = expander_text(expander(at, "Goal-Based - "))
    assert "GB-G01 - Safe and proportionate security triage" in goal and "Rejected alternatives" in goal
    assert "1. Open a security investigation" in goal
    utility_block = expander(at, "Utility-Based - ")
    assert "Selected expected utility:** 0.7812" in expander_text(utility_block)
    ranking = utility_block.dataframe[0].value
    assert list(ranking["Action"]) == ["INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST", "COLLECT_MORE_EVIDENCE", "MONITOR"]
    learning = expander_text(expander(at, "Learning Agent - "))
    assert "privileged_login_anomaly" in learning and fmt.EXPERIENCE_NOTE in learning
    llm_block = expander(at, "LLM Agent - ")
    llm_text = expander_text(llm_block)
    assert "LLM Agent - MOCK / Offline Validation" in llm_text and "Tool calls used" in llm_text
    assert "AUTH-A-001 ... AUTH-A-012" in llm_block.dataframe[0].value.to_string()
    assert "LIVE" not in llm_text  # a mock run never carries a live claim


def test_model_based_memory_demo_in_the_ui() -> None:
    at = run_all_agents(run_app())
    demo = expander(at, "How Model-Based Memory")
    demo.button(key="mb_run_twice").click()
    at = at.run()
    demo = expander(at, "How Model-Based Memory")
    table = demo.dataframe[0].value
    assert list(table["Decision"]) == ["INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"]
    text = expander_text(demo)
    assert "Account observations: 1 -> 2" in text and fmt.MEMORY_DEMO_NOTE in text
    # the main comparison on the page is unaffected by the demonstration
    assert [r["Decision"] for r in comparison_rows(at)] == ["INVESTIGATE"] * 6


def test_learning_demo_in_the_ui_uses_temporary_data_and_changes_the_decision() -> None:
    before = hashlib.sha256((DATA_DIR / "experience.json").read_bytes()).hexdigest()
    at = run_app()
    for _ in range(4):
        expander(at, "How Feedback").button(key="learn_failure").click()
        at = at.run()
    block = expander(at, "How Feedback")
    text = expander_text(block)
    assert "Decision at start:** `INVESTIGATE`" in text and "Current decision:** `ESCALATE_TO_SECURITY_ANALYST`" in text
    assert "Feedback events recorded:** 4" in text and fmt.LEARNING_DEMO_NOTE in text
    assert list(block.dataframe[0].value["Stage"]) == ["BEFORE", "AFTER"]
    assert hashlib.sha256((DATA_DIR / "experience.json").read_bytes()).hexdigest() == before
    expander(at, "How Feedback").button(key="learn_reset").click()
    at = at.run()
    assert "Feedback events recorded:** 0" in expander_text(expander(at, "How Feedback"))
    assert "Current decision:** `INVESTIGATE`" in expander_text(expander(at, "How Feedback"))


def test_reset_demonstrations_button_clears_both_demos() -> None:
    at = run_app()
    expander(at, "How Model-Based").button(key="mb_run_twice").click()
    expander(at, "How Feedback").button(key="learn_failure").click()
    at = at.run()
    at = at.sidebar.button(key="reset_demos").click().run()
    assert "Feedback events recorded:** 0" in expander_text(expander(at, "How Feedback"))
    assert not expander(at, "How Model-Based").dataframe


def test_one_failing_agent_does_not_crash_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    import devtools.mock_llm_runtime as mock_module

    def broken(scenario: Any) -> None:
        raise RuntimeError('File "C:\\Users\\me\\secret.py" Traceback api_key=abc123')

    monkeypatch.setattr(mock_module, "mock_llm_agent_factory", broken)
    at = run_all_agents(run_app())
    assert not at.exception
    rows = comparison_rows(at)
    assert [r["Decision"] for r in rows] == ["INVESTIGATE"] * 5 + ["ERROR"] and rows[5]["Runtime"] == "UNAVAILABLE"
    errors = " ".join(e.value for e in at.error)
    assert "LLM Agent unavailable in the selected runtime mode." in errors
    for leaked in ("Traceback", "secret.py", "C:\\", "abc123"):
        assert leaked not in all_text(at)
    assert len([e for e in at.expander if " - HIGH / INVESTIGATE" in e.label]) == 5
