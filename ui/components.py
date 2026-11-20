"""Streamlit rendering. All values come from ui.formatters / the comparison service; nothing is decided here.

The UI can only: run the six agents, replay the two demonstrations, and display results. It has no way to
block, disable, quarantine, change a firewall or contain anything, and it never writes to data/.
"""

from dataclasses import dataclass

import streamlit as st

from models.comparison_models import AgentComparisonEntry, AgentComparisonResult, AgentStatus
from models.learning_models import FeedbackOutcome
from models.llm_models import RuntimeMode
from models.schemas import AgentType, SecurityScenario
from services.comparison_service import (
    LearningFeedbackDemo,
    ModelBasedMemoryDemo,
    special_details_for,
)
from services.llm_runtime import LLMRuntimeStatus, live_unavailable_message
from ui import formatters as fmt

MB_DEMO_KEY = "mb_demo"
LEARNING_DEMO_KEY = "learning_demo"
COMPARISON_KEY = "comparison"
_OPTION_FOR_MODE = {RuntimeMode.MOCK: fmt.MOCK_MODE_OPTION, RuntimeMode.OLLAMA: fmt.LIVE_MODE_OPTION}


@dataclass(frozen=True)
class SidebarChoice:
    scenario_id: str
    llm_mode: RuntimeMode | None
    run_clicked: bool
    retry_clicked: bool
    reset_clicked: bool


# --- page frame ---------------------------------------------------------------------------------------


def render_header() -> None:
    st.title("Security Alert Triage & Prioritization Assistant")
    st.markdown(fmt.SUBTITLE)
    st.caption(fmt.SAFETY_NOTE)


def _runtime_caption(mode: RuntimeMode, status: LLMRuntimeStatus) -> None:
    if mode is RuntimeMode.OLLAMA:
        st.caption("Runtime: OLLAMA")
        st.caption(f"Model: {status.model_name or 'not set'}")
    else:
        st.caption(f"Runtime: {fmt.MOCK_MODE_OPTION}")


def _render_runtime_selector(status: LLMRuntimeStatus) -> RuntimeMode | None:
    """Show the runtime. A selector appears only in development, and only when two runtimes are really available."""
    if status.config_error:
        st.error(f"LLM runtime configuration problem: {fmt.display_error(status.config_error)}")
        return None
    if status.submission_mode:  # submission: LIVE OLLAMA only, no selector, no mock controls
        if RuntimeMode.OLLAMA not in status.available:
            st.error(live_unavailable_message(status.model_name))
            return None
        _runtime_caption(RuntimeMode.OLLAMA, status)
        return RuntimeMode.OLLAMA
    modes = list(status.available)
    mode: RuntimeMode | None = None
    if len(modes) == 1:
        mode = modes[0]
        _runtime_caption(mode, status)
    elif modes:
        options = [_OPTION_FOR_MODE[m] for m in modes]
        index = modes.index(status.default) if status.default in modes else None
        chosen = st.radio("LLM Agent runtime", options, index=index, key="llm_mode_option")
        mode = fmt.runtime_mode_for_option(chosen) if chosen else None
        if mode is not None:
            _runtime_caption(mode, status)
    else:
        st.caption("No LLM runtime is enabled.")
    for notice in status.notices:
        st.caption(fmt.display_error(notice))
    return mode


def render_sidebar(
    scenarios: list[SecurityScenario], status: LLMRuntimeStatus, retry_available: bool = False,
) -> SidebarChoice:
    labels = {s.scenario_id: fmt.scenario_label(s) for s in scenarios}
    with st.sidebar:
        st.subheader("Scenario")
        scenario_id = st.selectbox("Select a scenario", list(labels), format_func=labels.__getitem__, key="scenario_id")
        st.subheader("LLM runtime")
        mode = _render_runtime_selector(status)
        run_clicked = st.button("Run All Agents", type="primary", key="run_all")
        retry_clicked = st.button("Retry LLM Agent", key="retry_llm") if retry_available else False
        reset_clicked = st.button("Reset Demonstrations", key="reset_demos")
        st.divider()
        for line in fmt.offline_test_summary_lines(status.submission_mode, live_selected=mode is RuntimeMode.OLLAMA):
            st.caption(line)
    return SidebarChoice(scenario_id, mode, run_clicked, retry_clicked, reset_clicked)


def render_scenario_summary(scenario: SecurityScenario) -> None:
    st.subheader("Scenario")
    with st.container(border=True):
        st.caption(f"{scenario.title} - {scenario.description}")
        fields = fmt.scenario_summary_fields(scenario)
        for row in (fields[:5], fields[5:]):
            for column, (label, value) in zip(st.columns(5), row, strict=False):
                column.caption(label)
                column.markdown(f"**{value.capitalize()}**")  # capitalised so long values don't wrap mid-word


def render_llm_banner(mode: RuntimeMode | None, result: AgentComparisonResult | None, status: LLMRuntimeStatus) -> None:
    """The MOCK notice is never hidden in development. LIVE success is shown only after a real live result."""
    if mode is RuntimeMode.MOCK:
        st.warning(fmt.MOCK_BANNER)
        return
    if mode is None:
        st.info("No LLM runtime is enabled, so the LLM Agent is unavailable. The five classical agents still run.")
        return
    llm = result.entry(AgentType.LLM) if result is not None else None
    if llm is None:
        st.info("Live Ollama is selected. Press Run All Agents. CPU inference can take several minutes.")
    elif llm.status is AgentStatus.SUCCESS and llm.llm_runtime_mode is RuntimeMode.OLLAMA:
        st.success(fmt.LIVE_SUCCESS_TEXT)
    else:
        st.error(live_unavailable_message(status.model_name))


# --- comparison -------------------------------------------------------------------------------------------


def render_comparison(result: AgentComparisonResult) -> None:
    st.subheader("Six-agent comparison")
    st.dataframe(fmt.comparison_rows(result), hide_index=True, width="stretch")
    st.caption(fmt.agreement_summary(result))
    st.caption(fmt.same_input_statement(result))
    for failure in result.errors:
        st.error(f"{failure.agent_name}: {fmt.display_error(failure.message)}")


def _bullets(items: tuple[str, ...]) -> None:
    st.markdown("\n".join(f"- {item}" for item in items))


def _key_values(pairs: list[tuple[str, str]]) -> None:
    st.markdown("\n".join(f"- **{label}:** {value}" for label, value in pairs))


def _render_required_fields(entry: AgentComparisonEntry) -> None:
    left, right = st.columns(2)
    left.markdown(f"**Priority**\n\n{fmt.priority_badge(entry.priority)}")
    right.markdown(f"**Decision**\n\n`{entry.decision}`")
    st.markdown("**Why / reasoning**")
    st.write(entry.reasoning_summary)
    st.markdown("**Evidence**")
    _bullets(entry.evidence)
    st.markdown("**Uncertainty**")
    st.write(entry.uncertainty)
    st.markdown("**Likely interpretation**")
    st.write(entry.likely_interpretation)
    st.markdown("**Recommended investigation / escalation steps**")
    st.markdown("\n".join(f"{n}. {step}" for n, step in enumerate(entry.recommended_actions, start=1)))
    st.markdown("**Audit note**")
    st.caption(entry.audit_note)


def _render_goal_details(special: dict) -> None:
    details = fmt.goal_based_details(special)
    st.markdown(f"**Goal:** {details['goal']}")
    st.markdown("**Candidate actions:** " + ", ".join(f"`{c}`" for c in details["candidates"]))
    st.markdown(f"**Selected plan** ({details['plan_id']})")
    st.markdown("\n".join(f"{n}. {step}" for n, step in enumerate(details["plan_steps"], start=1)))
    st.caption(details["selection_reason"])
    if details["rejected"]:
        st.markdown("**Rejected alternatives**")
        st.markdown("\n".join(f"- `{item['decision']}`: {item['reason']}" for item in details["rejected"]))
    if details["acceptable_not_selected"]:
        st.markdown("**Acceptable but not selected**")
        st.markdown("\n".join(f"- `{i['decision']}`: {i['reason']}" for i in details["acceptable_not_selected"]))


def _render_utility_details(entry: AgentComparisonEntry) -> None:
    special = entry.special_details
    st.dataframe(
        fmt.utility_ranking_rows(special), hide_index=True, width="stretch",
        column_config={"Expected Utility": st.column_config.NumberColumn(format="%.4f")},
    )
    st.markdown(f"**Selected expected utility:** {special['selected_expected_utility']:.4f}")
    st.caption(fmt.UTILITY_NOTE)
    if trade_off := fmt.utility_trade_off_text(special):
        st.caption(trade_off)


def _render_learning_details(special: dict) -> None:
    _key_values(fmt.learning_summary(special))
    st.dataframe(fmt.learning_history_rows(special), hide_index=True, width="stretch")
    st.caption(fmt.EXPERIENCE_NOTE)


def _render_llm_details(entry: AgentComparisonEntry) -> None:
    special = entry.special_details
    st.markdown(f"**{fmt.llm_detail_header(entry)}**")
    if entry.llm_runtime_mode is RuntimeMode.OLLAMA:
        st.markdown(f"Runtime: **OLLAMA**  \nModel: `{special.get('model_name', 'unknown')}`")
        st.caption("Tools were selected by the configured Ollama model.")
    else:
        st.caption(f"{fmt.MOCK_STATUS_LABEL}. The tool sequence below is scripted by a test model, not chosen by Ollama.")
    rows = fmt.llm_trace_rows(special)
    if rows:
        st.markdown("**Tool calls used**")
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        selected = fmt.llm_selected_functions(entry)
        st.markdown("**Evidence functions selected**")
        if selected:
            st.markdown("\n".join(f"- `{name}`" + (f" ({status})" if status else "") for name, status in selected))
        else:
            st.caption("No evidence-function details were reported for this run.")
    st.markdown(f"**Source record IDs:** {fmt.llm_source_ids(special)}")
    st.markdown("**Validation**")
    for label, text, level in fmt.validation_status(special):
        line = f"{label}: {text}"
        if level == "success":
            st.success(line)
        elif level == "warning":
            st.warning(line)
        else:
            st.caption(line)
    st.caption("Only observable tool calls and results are shown; no model reasoning is recorded.")


def _render_specific_details(entry: AgentComparisonEntry) -> None:
    special = entry.special_details
    match entry.agent_type:
        case AgentType.SIMPLE_REFLEX:
            _key_values(fmt.simple_reflex_details(special))
        case AgentType.MODEL_BASED_REFLEX:
            _key_values(fmt.model_based_details(special))
        case AgentType.GOAL_BASED:
            _render_goal_details(special)
        case AgentType.UTILITY_BASED:
            _render_utility_details(entry)
        case AgentType.LEARNING:
            _render_learning_details(special)
        case AgentType.LLM:
            _render_llm_details(entry)


def render_agent_details(result: AgentComparisonResult) -> None:
    st.subheader("Agent details")
    for entry in result.results:
        with st.expander(fmt.agent_expander_label(entry)):
            if entry.status is not AgentStatus.SUCCESS:
                st.error(fmt.display_error(entry.error_message or "This agent could not complete."))
                continue
            _render_required_fields(entry)
            st.divider()
            _render_specific_details(entry)


# --- demonstrations ----------------------------------------------------------------------------------------


def _close_stored_demo(key: str) -> None:
    stored = st.session_state.pop(key, None)
    if stored is not None and hasattr(stored[1], "close"):
        stored[1].close()


def reset_demonstrations() -> None:
    _close_stored_demo(LEARNING_DEMO_KEY)
    st.session_state.pop(MB_DEMO_KEY, None)


def render_memory_demo(scenario: SecurityScenario) -> None:
    with st.expander("How Model-Based Memory Changes a Decision"):
        st.caption(fmt.MEMORY_DEMO_NOTE)
        if st.button(f"Run Scenario {scenario.scenario_id} Twice", key="mb_run_twice"):
            demo = ModelBasedMemoryDemo()  # one fresh agent, reused on purpose
            demo.observe(scenario)
            demo.observe(scenario)
            st.session_state[MB_DEMO_KEY] = (scenario.scenario_id, demo)
        stored = st.session_state.get(MB_DEMO_KEY)
        if stored is None or stored[0] != scenario.scenario_id:
            return
        observations = stored[1].observations
        st.dataframe(fmt.memory_demo_rows(observations), hide_index=True, width="stretch")
        st.markdown(f"**State change:** {fmt.memory_state_change(observations)}")
        st.caption(fmt.memory_demo_conclusion(observations))


def _learning_demo_for(scenario: SecurityScenario) -> LearningFeedbackDemo:
    stored = st.session_state.get(LEARNING_DEMO_KEY)
    if stored is None or stored[0] != scenario.scenario_id:
        _close_stored_demo(LEARNING_DEMO_KEY)
        stored = (scenario.scenario_id, LearningFeedbackDemo())
        st.session_state[LEARNING_DEMO_KEY] = stored
    return stored[1]


def render_learning_demo(scenario: SecurityScenario) -> None:
    with st.expander("How Feedback Changes the Learning Agent"):
        demo = _learning_demo_for(scenario)
        st.caption(
            "Feedback is recorded in a temporary copy of the simulated experience. "
            "The committed data/experience.json is never modified."
        )
        failure, success, reset = st.columns(3)
        if failure.button("Record FAILURE for current Learning Agent decision", key="learn_failure"):
            demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
        if success.button("Record SUCCESS", key="learn_success"):
            demo.record_feedback(scenario, FeedbackOutcome.SUCCESS)
        if reset.button("Reset Learning Demo", key="learn_reset"):
            demo.reset()

        decision = demo.decide(scenario)
        special = special_details_for(AgentType.LEARNING, decision)
        steps = demo.steps
        starting = steps[0].decision_before if steps else decision.decision
        st.markdown(f"**Decision at start:** `{starting}`  \n**Current decision:** `{decision.decision}`  \n"
                    f"**Feedback events recorded:** {demo.feedback_count}")
        if steps:
            st.markdown("**Latest feedback**")
            st.dataframe(fmt.feedback_before_after_rows(steps[-1]), hide_index=True, width="stretch")
            st.caption(fmt.LEARNING_DEMO_NOTE)
        st.markdown("**Current history for each candidate action**")
        st.dataframe(fmt.learning_history_rows(special), hide_index=True, width="stretch")
        st.caption(fmt.EXPERIENCE_NOTE)


# --- footer ------------------------------------------------------------------------------------------------


def render_assignment_coverage(submission_mode: bool = False, live_selected: bool = False) -> None:
    rows = [
        {"Assignment requirement": "Select a scenario (normal, critical, ambiguous)", "Where": "Sidebar", "Status": "Implemented"},
        {"Assignment requirement": "Same scenario given to all six agents", "Where": "Comparison caption (fingerprint)", "Status": "Implemented"},
        {"Assignment requirement": "Priority level", "Where": "Comparison table and each agent", "Status": "Implemented"},
        {"Assignment requirement": "Evidence and uncertainty", "Where": "Each agent's details", "Status": "Implemented"},
        {"Assignment requirement": "Likely interpretation", "Where": "Each agent's details", "Status": "Implemented"},
        {"Assignment requirement": "Recommended investigation / escalation steps", "Where": "Each agent's details", "Status": "Implemented"},
        {"Assignment requirement": "Audit note", "Where": "Each agent's details", "Status": "Implemented"},
        {"Assignment requirement": "Five classical agents", "Where": "Python decision rules, goals, utilities, experience", "Status": "Implemented"},
        {"Assignment requirement": "LLM agent with tool calling (local Ollama, evidence functions)", "Where": "LLM Agent details", "Status": fmt.assignment_llm_status(submission_mode, live_selected)},
    ]
    with st.expander("Assignment requirement coverage"):
        st.dataframe(rows, hide_index=True, width="stretch")
