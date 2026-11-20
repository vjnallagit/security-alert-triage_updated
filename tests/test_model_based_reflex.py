import builtins
from pathlib import Path

import pytest
from agents.base import BaseAgent
from agents.model_based_reflex import ModelBasedReflexAgent
from agents.simple_reflex import SimpleReflexAgent
from models.agent_state import ModelBasedState
from models.schemas import AgentDecision, Priority, SecurityScenario
from services.data_loader import DATA_DIR
from services.scenario_service import get_all_scenarios, get_scenario

ALL_IDS = ["A", "B", "C"]
ESCALATE = "ESCALATE_TO_SECURITY_ANALYST"


def test_mb01_inherits_from_base_agent():
    assert issubclass(ModelBasedReflexAgent, BaseAgent)
    assert isinstance(ModelBasedReflexAgent(), BaseAgent)


def test_mb02_new_agent_starts_with_default_state():
    state = ModelBasedReflexAgent().get_state_snapshot()
    assert state == ModelBasedState()
    assert state.processed_scenario_count == 0
    assert state.account_observations == {}
    assert state.last_decision is None and state.last_priority is None


def test_mb03_processing_scenario_a_updates_state():
    scenario = get_scenario("A")
    agent = ModelBasedReflexAgent()
    agent.decide(scenario)
    state = agent.get_state_snapshot()
    assert state.processed_scenario_count == 1
    assert state.account_observations == {scenario.target_account: 1}
    assert state.asset_observations == {scenario.asset_id: 1}
    assert state.alert_type_observations == {scenario.alert_type: 1}
    assert state.unfamiliar_source_observations == 1
    assert (state.last_scenario_id, state.last_alert_id) == ("A", scenario.alert_id)
    assert (state.last_decision, state.last_priority) == ("INVESTIGATE", Priority.HIGH)

    agent.decide(scenario)
    state = agent.get_state_snapshot()
    assert state.processed_scenario_count == 2
    assert state.account_observations[scenario.target_account] == 2


def test_mb04_first_scenario_a_is_high_investigate():
    result = ModelBasedReflexAgent().decide(get_scenario("A"))
    assert (result.priority, result.decision) == (Priority.HIGH, "INVESTIGATE")
    assert result.metadata["matched_rule_id"] == "MB-03"


def test_mb05_second_scenario_a_is_critical_escalate():
    agent = ModelBasedReflexAgent()
    scenario = get_scenario("A")
    agent.decide(scenario)
    second = agent.decide(scenario)
    assert (second.priority, second.decision) == (Priority.CRITICAL, ESCALATE)
    assert second.metadata["matched_rule_id"] == "MB-02"
    assert second.metadata["state_summary"]["previous_decision"] == "INVESTIGATE"
    assert second.metadata["state_summary"]["pattern_seen_previously"] is True


def test_mb06_state_is_instance_specific():
    scenario = get_scenario("A")
    used = ModelBasedReflexAgent()
    used.decide(scenario)
    used.decide(scenario)
    fresh = ModelBasedReflexAgent()
    assert fresh.get_state_snapshot() == ModelBasedState()
    result = fresh.decide(scenario)
    assert (result.priority, result.decision) == (Priority.HIGH, "INVESTIGATE")


def test_mb07_simple_reflex_stays_same_where_model_based_changes():
    scenario = get_scenario("A")
    simple, model_based = SimpleReflexAgent(), ModelBasedReflexAgent()
    simple_runs = [simple.decide(scenario) for _ in range(2)]
    model_runs = [model_based.decide(scenario) for _ in range(2)]
    assert [(r.priority, r.decision) for r in simple_runs] == [(Priority.HIGH, "INVESTIGATE")] * 2
    assert (model_runs[0].priority, model_runs[0].decision) == (Priority.HIGH, "INVESTIGATE")
    assert (model_runs[1].priority, model_runs[1].decision) == (Priority.CRITICAL, ESCALATE)


def test_mb08_snapshot_cannot_mutate_private_state():
    scenario = get_scenario("A")
    agent = ModelBasedReflexAgent()
    agent.decide(scenario)
    snapshot = agent.get_state_snapshot()
    snapshot.processed_scenario_count = 99
    snapshot.account_observations[scenario.target_account] = 99
    snapshot.account_observations["injected"] = 1
    again = agent.get_state_snapshot()
    assert again.processed_scenario_count == 1
    assert again.account_observations == {scenario.target_account: 1}
    # The next decision is based on the real state (2nd observation), not the tampered one.
    assert agent.decide(scenario).metadata["matched_rule_id"] == "MB-02"


def test_mb09_scenario_b_first_observation_is_collect_more_evidence():
    result = ModelBasedReflexAgent().decide(get_scenario("B"))
    assert (result.priority, result.decision) == (Priority.LOW, "COLLECT_MORE_EVIDENCE")
    assert result.metadata["matched_rule_id"] == "MB-05"


def test_mb10_scenario_b_repeated_observation_is_monitor():
    agent = ModelBasedReflexAgent()
    scenario = get_scenario("B")
    agent.decide(scenario)
    second = agent.decide(scenario)
    assert (second.priority, second.decision) == (Priority.LOW, "MONITOR")
    assert second.metadata["matched_rule_id"] == "MB-04"


@pytest.mark.parametrize("repeat", [1, 2])
def test_mb11_scenario_c_is_critical_escalate_but_unconfirmed(repeat):
    agent = ModelBasedReflexAgent()
    scenario = get_scenario("C")
    for _ in range(repeat):
        result = agent.decide(scenario)
    assert (result.priority, result.decision) == (Priority.CRITICAL, ESCALATE)
    assert result.metadata["matched_rule_id"] == "MB-01"
    assert "Confirmed impact: False" in result.evidence
    assert "NOT confirmed" in result.uncertainty


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_mb12_result_validates_as_agent_decision(scenario_id):
    result = ModelBasedReflexAgent().decide(get_scenario(scenario_id))
    assert AgentDecision.model_validate(result.model_dump()) == result
    assert result.agent_type.value == "MODEL_BASED_REFLEX"
    assert result.reasoning and result.evidence and result.uncertainty
    assert result.likely_interpretation and result.recommended_actions and result.audit_note
    assert result.tool_calls == [] and result.utility_score is None
    assert result.experience_used is None


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_mb13_metadata_describes_mechanism_and_state(scenario_id):
    metadata = ModelBasedReflexAgent().decide(get_scenario(scenario_id)).metadata
    assert metadata["matched_rule_id"].startswith("MB-")
    assert metadata["decision_mechanism"] == "internal_state_plus_condition_action_rules"
    assert metadata["uses_internal_state"] is True
    summary = metadata["state_summary"]
    assert summary["processed_scenario_count"] == 1
    assert summary["account_count"] == summary["asset_count"] == summary["alert_type_count"] == 1
    assert summary["pattern_seen_previously"] is False
    assert summary["previous_decision"] is None


def test_mb14_agent_does_not_read_or_write_files(monkeypatch):
    """Any file access during decide() fails the test; experience.json must stay byte-identical."""
    scenarios = get_all_scenarios()  # load before blocking file access
    experience = DATA_DIR / "experience.json"
    before = (experience.read_bytes(), experience.stat().st_mtime_ns)

    def blocked(*args, **kwargs):
        raise AssertionError("agent attempted file access")

    monkeypatch.setattr(builtins, "open", blocked)
    for name in ("read_text", "read_bytes", "write_text", "write_bytes", "open"):
        monkeypatch.setattr(Path, name, blocked)

    agent = ModelBasedReflexAgent()
    for scenario in scenarios + scenarios:
        assert agent.decide(scenario).experience_used is None

    monkeypatch.undo()
    assert (experience.read_bytes(), experience.stat().st_mtime_ns) == before


def test_mb14_agent_modules_do_not_depend_on_data_services():
    import agents.model_based_reflex as module

    assert not hasattr(module, "get_experience")
    assert not any(name.startswith("services") for name in vars(module))


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_mb15_scenario_is_not_mutated(scenario_id):
    scenario = get_scenario(scenario_id)
    snapshot = scenario.model_copy(deep=True)
    agent = ModelBasedReflexAgent()
    agent.decide(scenario)
    agent.decide(scenario)
    assert scenario == snapshot


def test_state_accumulates_across_different_scenarios():
    agent = ModelBasedReflexAgent()
    for scenario in get_all_scenarios():
        agent.decide(scenario)
    state = agent.get_state_snapshot()
    assert state.processed_scenario_count == 3
    assert state.last_scenario_id == "C"
    assert state.correlated_risk_observations == 2  # A (2 alerts) and C (3 alerts)
    assert len(state.asset_observations) == 3


def test_failed_decision_does_not_corrupt_state(monkeypatch):
    import agents.model_based_reflex as module

    agent = ModelBasedReflexAgent()
    monkeypatch.setattr(module, "RULES", ())
    with pytest.raises(RuntimeError):
        agent.decide(get_scenario("A"))
    assert agent.get_state_snapshot() == ModelBasedState()


def test_rule_order_and_unmatched_fallbacks():
    from agents.model_based_reflex import RULES

    assert [r.rule_id for r in RULES] == [
        "MB-01", "MB-02", "MB-03", "MB-04", "MB-05", "MB-06", "MB-DEFAULT",
    ]
    medium = SecurityScenario.model_validate(
        {**get_scenario("B").model_dump(mode="json"), "severity": "MEDIUM", "evidence_strength": "MODERATE"}
    )
    result = ModelBasedReflexAgent().decide(medium)
    assert result.metadata["matched_rule_id"] == "MB-DEFAULT"
    assert (result.priority, result.decision) == (Priority.MEDIUM, "COLLECT_MORE_EVIDENCE")
