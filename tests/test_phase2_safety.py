import re

import pytest
from agents import ModelBasedReflexAgent, SimpleReflexAgent
from services.scenario_service import get_all_scenarios

# Phrases that would claim a disruptive action happened or is being performed.
DISRUPTIVE_CLAIMS = re.compile(
    r"\b(blocked|disabled|terminated|quarantined|isolated|deleted|attacked|retaliated)\b"
    r"|firewall (was|has been|rule changed)|changed the firewall|contained the incident",
    re.IGNORECASE,
)
SCENARIO_IDS = [s.scenario_id for s in get_all_scenarios()]


def _text_fields(result) -> list[str]:
    return [
        result.decision,
        result.reasoning,
        result.uncertainty,
        result.likely_interpretation,
        result.audit_note,
        *result.recommended_actions,
        *result.evidence,
    ]


def _results(agent_class):
    """Run each scenario twice on one instance so stateful behaviour is covered too."""
    agent = agent_class()
    scenarios = {s.scenario_id: s for s in get_all_scenarios()}
    return [
        (scenario_id, agent.decide(scenarios[scenario_id]))
        for _ in range(2)
        for scenario_id in SCENARIO_IDS
    ]


@pytest.mark.parametrize("agent_class", [SimpleReflexAgent, ModelBasedReflexAgent])
def test_no_result_claims_disruptive_action(agent_class):
    for scenario_id, result in _results(agent_class):
        for text in _text_fields(result):
            assert not DISRUPTIVE_CLAIMS.search(text), f"{scenario_id}: {text}"


@pytest.mark.parametrize("agent_class", [SimpleReflexAgent, ModelBasedReflexAgent])
def test_every_result_is_recommendation_only(agent_class):
    for scenario_id, result in _results(agent_class):
        assert result.metadata["recommendation_only"] is True, scenario_id
        assert result.metadata["executed_actions"] == [], scenario_id
        assert result.tool_calls == []
        assert "No disruptive action was executed" in result.audit_note


@pytest.mark.parametrize("agent_class", [SimpleReflexAgent, ModelBasedReflexAgent])
def test_consequential_action_requires_human_authorization(agent_class):
    for scenario_id, result in _results(agent_class):
        assert result.metadata["human_authorization_required_for_consequential_action"] is True
        assert "human authorization" in result.audit_note
        containment = [a for a in result.recommended_actions if "containment" in a.lower()]
        for action in containment:
            assert "authorized" in action and "review" in action, action


@pytest.mark.parametrize("agent_class", [SimpleReflexAgent, ModelBasedReflexAgent])
def test_decisions_use_only_the_defensive_vocabulary(agent_class):
    allowed = {"MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"}
    for _, result in _results(agent_class):
        assert result.decision in allowed
