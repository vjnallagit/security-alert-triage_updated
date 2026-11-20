import re

import pytest
from agents import UtilityBasedAgent
from services.scenario_service import get_all_scenarios, get_scenario

SCENARIO_IDS = [s.scenario_id for s in get_all_scenarios()]

# Phrases claiming a disruptive action happened or is being performed.
DISRUPTIVE_CLAIMS = re.compile(
    r"\b(blocked|disabled|terminated|quarantined|isolated|deleted|attacked|retaliated)\b"
    r"|firewall (was|has been|rule changed)|changed the firewall|contained the incident"
    r"|containment (was|has been) (performed|executed|applied)",
    re.IGNORECASE,
)
OFFENSIVE_STEPS = re.compile(r"\b(block|disable|quarantine|terminate|attack|retaliate|delete|hack)\b", re.I)
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
    agent = UtilityBasedAgent()
    return [(s.scenario_id, agent.decide(s)) for s in get_all_scenarios() for _ in range(2)]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_executed_actions_remain_empty_and_output_is_recommendation_only(scenario_id):
    result = UtilityBasedAgent().decide(get_scenario(scenario_id))
    assert result.metadata["executed_actions"] == []
    assert result.metadata["recommendation_only"] is True
    assert result.tool_calls == []
    assert "No disruptive action was executed" in result.audit_note


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_result_claims_blocking_disabling_quarantine_firewall_or_containment(scenario_id):
    result = UtilityBasedAgent().decide(get_scenario(scenario_id))
    for text in _text_fields(result):
        assert not DISRUPTIVE_CLAIMS.search(text), text


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_offensive_or_automatic_step_is_proposed(scenario_id):
    result = UtilityBasedAgent().decide(get_scenario(scenario_id))
    for step in result.recommended_actions:
        assert not OFFENSIVE_STEPS.search(step), step


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_containment_is_only_ever_an_authorized_review(scenario_id):
    for step in UtilityBasedAgent().decide(get_scenario(scenario_id)).recommended_actions:
        if "containment" in step.lower():
            assert "authorized" in step and "review" in step and "only if" in step


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_human_authorization_is_required_for_consequential_action(scenario_id):
    result = UtilityBasedAgent().decide(get_scenario(scenario_id))
    assert result.metadata["human_authorization_required_for_consequential_action"] is True
    assert "human authorization" in result.audit_note
    assert result.metadata["ineligible_actions"] == []  # every standard candidate is human-controlled


def test_unconfirmed_compromise_is_never_described_as_confirmed():
    for scenario in get_all_scenarios():
        assert scenario.confirmed_impact is False
    for scenario_id, result in _all_results():
        for text in _text_fields(result):
            assert not COMPROMISE_CLAIMS.search(text), f"{scenario_id}: {text}"
        assert "NOT confirmed" in result.uncertainty


def test_selected_decision_is_always_in_the_defensive_vocabulary():
    allowed = {"MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"}
    assert {r.decision for _, r in _all_results()} <= allowed


def test_simulation_assumptions_are_disclosed_in_every_result():
    for _, result in _all_results():
        assert "simulated" in result.reasoning.lower()
        assert "not real-world statistical estimates" in result.metadata["assumptions_note"]


def test_compromise_claim_pattern_detects_claims_but_not_negations():
    assert COMPROMISE_CLAIMS.search("The account was compromised.")
    assert COMPROMISE_CLAIMS.search("This is a confirmed compromise.")
    assert not COMPROMISE_CLAIMS.search("Compromise remains unconfirmed.")
    assert not COMPROMISE_CLAIMS.search("Compromise is not confirmed.")
