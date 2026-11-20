import pytest
from agents.base import BaseAgent
from agents.simple_reflex import SimpleReflexAgent
from models.schemas import AgentDecision, Priority, SecurityScenario
from services.scenario_service import get_all_scenarios, get_scenario

EXPECTED = {
    "A": ("SR-02", Priority.HIGH, "INVESTIGATE"),
    "B": ("SR-03", Priority.LOW, "COLLECT_MORE_EVIDENCE"),
    "C": ("SR-01", Priority.CRITICAL, "ESCALATE_TO_SECURITY_ANALYST"),
}
ALL_IDS = ["A", "B", "C"]


def test_sr01_inherits_from_base_agent():
    assert issubclass(SimpleReflexAgent, BaseAgent)
    assert isinstance(SimpleReflexAgent(), BaseAgent)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_sr02_to_sr04_expected_outcomes(scenario_id):
    rule_id, priority, decision = EXPECTED[scenario_id]
    result = SimpleReflexAgent().decide(get_scenario(scenario_id))
    assert (result.priority, result.decision) == (priority, decision)
    assert result.metadata["matched_rule_id"] == rule_id  # SR-07


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_sr05_result_validates_as_agent_decision(scenario_id):
    result = SimpleReflexAgent().decide(get_scenario(scenario_id))
    assert AgentDecision.model_validate(result.model_dump()) == result
    assert result.agent_type.value == "SIMPLE_REFLEX"


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_sr06_result_has_all_explanation_fields(scenario_id):
    result = SimpleReflexAgent().decide(get_scenario(scenario_id))
    assert result.reasoning and result.uncertainty and result.likely_interpretation
    assert result.evidence and result.recommended_actions and result.audit_note
    assert EXPECTED[scenario_id][0] in result.reasoning
    assert f"Failed login count: {get_scenario(scenario_id).failed_login_count}" in result.evidence


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_sr07_metadata_describes_mechanism(scenario_id):
    metadata = SimpleReflexAgent().decide(get_scenario(scenario_id)).metadata
    assert metadata["decision_mechanism"] == "ordered_condition_action_rules"
    assert metadata["uses_internal_state"] is False
    assert metadata["current_percept_only"] is True
    assert metadata["matched_rule_name"]


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_sr08_states_no_action_was_executed(scenario_id):
    result = SimpleReflexAgent().decide(get_scenario(scenario_id))
    assert result.metadata["executed_actions"] == []
    assert "No disruptive action was executed" in result.audit_note
    assert "human authorization" in result.audit_note


def test_sr09_repeated_calls_give_same_result():
    agent = SimpleReflexAgent()
    scenario = get_scenario("A")
    results = [agent.decide(scenario) for _ in range(3)]
    assert all(r == results[0] for r in results)
    assert (results[0].priority, results[0].decision) == (Priority.HIGH, "INVESTIGATE")


def test_sr10_separate_instances_agree():
    for scenario in get_all_scenarios():
        assert SimpleReflexAgent().decide(scenario) == SimpleReflexAgent().decide(scenario)


def test_sr11_agent_gains_no_state_from_calls():
    agent = SimpleReflexAgent()
    before = dict(vars(agent))
    for scenario in get_all_scenarios():
        agent.decide(scenario)
    assert dict(vars(agent)) == before == {}


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_sr12_scenario_is_not_mutated(scenario_id):
    scenario = get_scenario(scenario_id)
    snapshot = scenario.model_copy(deep=True)
    SimpleReflexAgent().decide(scenario)
    assert scenario == snapshot


def test_scenario_c_is_not_described_as_confirmed_compromise():
    result = SimpleReflexAgent().decide(get_scenario("C"))
    assert "Confirmed impact: False" in result.evidence
    assert "NOT confirmed" in result.uncertainty


def test_rule_order_is_first_match_wins():
    """Scenario A also satisfies SR-04 (HIGH severity), but the more specific SR-02 is first."""
    from agents.simple_reflex import RULES, _high_or_critical_severity

    scenario = get_scenario("A")
    assert _high_or_critical_severity(scenario)
    assert [r.rule_id for r in RULES] == ["SR-01", "SR-02", "SR-03", "SR-04", "SR-DEFAULT"]
    assert SimpleReflexAgent().decide(scenario).metadata["matched_rule_id"] == "SR-02"


@pytest.mark.parametrize(
    "changes, rule_id, priority, decision",
    [
        ({"severity": "MEDIUM", "failed_login_count": 0, "correlated_alert_count": 0,
          "unusual_access": False, "evidence_strength": "MODERATE",
          "account_privilege": "STANDARD"}, "SR-DEFAULT", Priority.MEDIUM, "COLLECT_MORE_EVIDENCE"),
        ({"account_privilege": "STANDARD", "failed_login_count": 0}, "SR-04", Priority.HIGH, "INVESTIGATE"),
    ],
)
def test_fallback_rules(changes, rule_id, priority, decision):
    scenario = SecurityScenario.model_validate({**get_scenario("A").model_dump(mode="json"), **changes})
    result = SimpleReflexAgent().decide(scenario)
    assert result.metadata["matched_rule_id"] == rule_id
    assert (result.priority, result.decision) == (priority, decision)
