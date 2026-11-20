import re

import pytest
from agents import GoalBasedAgent
from services.scenario_service import get_all_scenarios, get_scenario

SCENARIO_IDS = [s.scenario_id for s in get_all_scenarios()]

# Phrases claiming a disruptive action happened or is being performed.
DISRUPTIVE_CLAIMS = re.compile(
    r"\b(blocked|disabled|terminated|quarantined|isolated|deleted|attacked|retaliated)\b"
    r"|firewall (was|has been|rule changed)|changed the firewall|contained the incident"
    r"|containment (was|has been) (performed|executed|applied)",
    re.IGNORECASE,
)
# Claims that compromise is established. Negated forms ("not confirmed") must not match.
COMPROMISE_CLAIMS = re.compile(
    r"(?<!not )(?<!un)confirmed (compromise|breach)"
    r"|compromise (is|was|has been) confirmed"
    r"|(account|system|asset|user) (is|was|has been) compromised",
    re.IGNORECASE,
)


def _text_fields(result) -> list[str]:
    return [
        result.decision, result.reasoning, result.uncertainty, result.likely_interpretation,
        result.audit_note, *result.recommended_actions, *result.evidence,
    ]


def _all_results():
    agent = GoalBasedAgent()
    return [(s.scenario_id, agent.decide(s)) for s in get_all_scenarios() for _ in range(2)]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_executed_actions_remain_empty_and_result_is_recommendation_only(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert result.metadata["executed_actions"] == []
    assert result.metadata["recommendation_only"] is True
    assert result.tool_calls == []
    assert "No disruptive action was executed" in result.audit_note


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_result_claims_blocking_disabling_quarantine_firewall_or_offense(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    for text in _text_fields(result):
        assert not DISRUPTIVE_CLAIMS.search(text), text


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_offensive_or_automatic_step_is_recommended(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    forbidden = re.compile(r"\b(block|disable|quarantine|terminate|attack|retaliate|delete|hack)\b", re.I)
    for step in result.recommended_actions:
        assert not forbidden.search(step), step


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_containment_is_only_ever_an_authorized_review(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    for step in result.recommended_actions:
        if "containment" in step.lower():
            assert "authorized" in step and "review" in step and "only if" in step


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_consequential_action_requires_human_authorization(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert result.metadata["human_authorization_required_for_consequential_action"] is True
    assert "human authorization" in result.audit_note
    prohibited = result.metadata["goal"]["prohibited_conditions"]
    assert "CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION" in prohibited
    assert result.metadata["goal_evaluation"]["violated_prohibited_conditions"] == []


def test_no_unconfirmed_scenario_is_described_as_confirmed_compromise():
    for scenario in get_all_scenarios():
        assert scenario.confirmed_impact is False
    for scenario_id, result in _all_results():
        for text in _text_fields(result):
            assert not COMPROMISE_CLAIMS.search(text), f"{scenario_id}: {text}"
        assert "NOT confirmed" in result.uncertainty


def test_selected_decision_is_always_in_the_defensive_vocabulary():
    allowed = {"MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"}
    assert {r.decision for _, r in _all_results()} <= allowed


def test_compromise_claim_pattern_detects_claims_but_not_negations():
    assert COMPROMISE_CLAIMS.search("The account was compromised.")
    assert COMPROMISE_CLAIMS.search("This is a confirmed compromise.")
    assert not COMPROMISE_CLAIMS.search("Compromise is not confirmed.")
    assert not COMPROMISE_CLAIMS.search("an unconfirmed compromise")
