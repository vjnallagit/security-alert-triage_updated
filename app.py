"""Streamlit entry point: streamlit run app.py.

Academic simulation only.

Run All Agents runs the five classical agents and the local-Ollama LLM Agent on the same
synthetic scenario and shows the six results side by side. The LLM Agent is part of that
comparison; there is no separate LLM panel.

No remediation or containment action is executed.
"""

from __future__ import annotations

import streamlit as st

from models.comparison_models import AgentComparisonResult, AgentStatus
from models.llm_models import RuntimeMode
from models.schemas import AgentType
from services.llm_errors import LLMConfigurationError
from services.llm_runtime import build_comparison_service, inspect_runtime
from services.scenario_service import get_all_scenarios
from ui import components as ui
from ui.formatters import LIVE_RUNNING_MESSAGE, display_error


def _retry_available(
    stored: AgentComparisonResult | None,
) -> bool:
    """Allow retry only for a failed live legacy LLM result."""

    return bool(
        stored
        and stored.llm_runtime_mode is RuntimeMode.OLLAMA
        and stored.entry(AgentType.LLM).status is AgentStatus.ERROR
    )


def main() -> None:
    st.set_page_config(
        page_title="Security Alert Triage",
        page_icon="🛡️",
        layout="wide",
    )

    scenarios = get_all_scenarios()
    status = inspect_runtime()

    stored = st.session_state.get(
        ui.COMPARISON_KEY
    )

    choice = ui.render_sidebar(
        scenarios,
        status,
        retry_available=_retry_available(
            stored
        ),
    )

    if choice.reset_clicked:
        ui.reset_demonstrations()

    scenario = next(
        scenario
        for scenario in scenarios
        if scenario.scenario_id
        == choice.scenario_id
    )

    live = (
        choice.llm_mode
        is RuntimeMode.OLLAMA
    )

    spinner_text = (
        LIVE_RUNNING_MESSAGE
        if live
        else "Running all six agents..."
    )

    ran = False

    try:
        if choice.run_clicked:
            with st.spinner(spinner_text):
                service = build_comparison_service(
                    choice.llm_mode,
                    reason=status.config_error,
                )

                st.session_state[
                    ui.COMPARISON_KEY
                ] = service.compare(
                    scenario
                )

            ran = True

        elif (
            choice.retry_clicked
            and stored is not None
            and stored.scenario.scenario_id
            == scenario.scenario_id
        ):
            with st.spinner(spinner_text):
                service = build_comparison_service(
                    choice.llm_mode
                )

                st.session_state[
                    ui.COMPARISON_KEY
                ] = service.rerun_agent(
                    stored,
                    AgentType.LLM,
                )

            ran = True

    except (LLMConfigurationError, ValueError) as exc:
        st.session_state.pop(
            ui.COMPARISON_KEY,
            None,
        )

        st.error(
            "Comparison runtime could not start: "
            f"{display_error(str(exc))}"
        )

    if ran:
        st.rerun()

    stored = st.session_state.get(
        ui.COMPARISON_KEY
    )

    current = bool(
        stored
        and stored.scenario.scenario_id
        == scenario.scenario_id
        and stored.llm_runtime_mode
        is choice.llm_mode
    )

    result = stored if current else None

    ui.render_header()
    ui.render_scenario_summary(scenario)
    ui.render_llm_banner(
        choice.llm_mode,
        result,
        status,
    )

    if result is None:
        st.info(
            "Press **Run All Agents** in the sidebar "
            "to run the comparison."
        )
    else:
        ui.render_comparison(result)
        ui.render_agent_details(result)

    st.divider()

    st.subheader("Demonstrations")

    ui.render_memory_demo(scenario)
    ui.render_learning_demo(scenario)
    ui.render_assignment_coverage(
        status.submission_mode,
        live_selected=live,
    )


if __name__ == "__main__":
    main()
