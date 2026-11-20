import pytest
from agents.base import BaseAgent
from models.schemas import (
    AgentDecision,
    AgentType,
    AlertDetail,
    ExperienceOutcome,
    Priority,
    ResponsePolicy,
    SecurityScenario,
)
from pydantic import ValidationError
from services.scenario_service import get_policy, get_scenario


@pytest.fixture
def scenario_data() -> dict:
    """A valid scenario as a plain dict, ready to be corrupted by individual tests."""
    return get_scenario("A").model_dump(mode="json")


def test_valid_scenario_round_trips(scenario_data):
    assert SecurityScenario.model_validate(scenario_data).scenario_id == "A"


@pytest.mark.parametrize("field", ["failed_login_count", "correlated_alert_count"])
def test_negative_counts_are_rejected(scenario_data, field):
    scenario_data[field] = -1
    with pytest.raises(ValidationError):
        SecurityScenario.model_validate(scenario_data)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("severity", "SEVERE"),
        ("scenario_category", "UNKNOWN_CATEGORY"),
        ("account_privilege", "ROOT"),
        ("source_familiarity", "MAYBE"),
        ("asset_criticality", "EXTREME"),
        ("evidence_strength", "WEAKISH"),
    ],
)
def test_invalid_enum_values_are_rejected(scenario_data, field, bad_value):
    scenario_data[field] = bad_value
    with pytest.raises(ValidationError):
        SecurityScenario.model_validate(scenario_data)


@pytest.mark.parametrize("field", ["scenario_id", "alert_id", "asset_id", "policy_id"])
@pytest.mark.parametrize("bad_value", ["", "   "])
def test_empty_identifiers_are_rejected(scenario_data, field, bad_value):
    scenario_data[field] = bad_value
    with pytest.raises(ValidationError):
        SecurityScenario.model_validate(scenario_data)


def test_unknown_field_is_rejected(scenario_data):
    scenario_data["agent_decision"] = "escalate"
    with pytest.raises(ValidationError):
        SecurityScenario.model_validate(scenario_data)


def test_scenario_has_no_decision_field():
    assert not {"decision", "priority", "recommended_actions"} & set(SecurityScenario.model_fields)


def test_alert_timestamp_must_be_timezone_aware():
    with pytest.raises(ValidationError):
        AlertDetail.model_validate(
            {
                "alert_id": "X",
                "timestamp": "2026-03-10T02:14:00",
                "alert_type": "t",
                "severity": "LOW",
                "summary": "s",
                "target": "t",
                "source": "s",
                "evidence": ["e"],
            }
        )


def test_policy_that_permits_automation_is_rejected():
    data = get_policy("POL-PRIV-LOGIN-001").model_dump(mode="json")
    data["automated_actions_permitted"] = True
    with pytest.raises(ValidationError):
        ResponsePolicy.model_validate(data)


def test_experience_success_rate_is_derived():
    assert ExperienceOutcome(success=8, failure=2).success_rate == pytest.approx(0.8)
    assert ExperienceOutcome(success=0, failure=0).success_rate is None


def test_experience_rejects_negative_counts():
    with pytest.raises(ValidationError):
        ExperienceOutcome(success=-1, failure=0)


def _decision_data() -> dict:
    return {
        "agent_name": "Example",
        "agent_type": AgentType.SIMPLE_REFLEX,
        "priority": Priority.HIGH,
        "decision": "investigate",
        "reasoning": "r",
        "uncertainty": "u",
        "likely_interpretation": "l",
        "recommended_actions": ["Review authentication activity"],
        "audit_note": "a",
    }


def test_agent_decision_accepts_required_fields_and_defaults_optionals():
    decision = AgentDecision.model_validate(_decision_data())
    assert decision.tool_calls == []
    assert decision.utility_score is None
    assert decision.experience_used is None
    assert decision.metadata == {}


def test_agent_decision_rejects_bad_priority_and_empty_actions():
    with pytest.raises(ValidationError):
        AgentDecision.model_validate({**_decision_data(), "priority": "URGENT"})
    with pytest.raises(ValidationError):
        AgentDecision.model_validate({**_decision_data(), "recommended_actions": []})


def test_base_agent_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseAgent()
