import re

import pytest
from agents import LearningAgent
from agents import learning_agent as learning_module
from agents.learning_agent import LEARNING_CANDIDATES, NoSafeActionError
from models.goal_models import ProhibitedCondition
from models.learning_models import FeedbackOutcome, LearningCandidate
from services.experience_repository import ExperienceRepository
from services.scenario_service import get_all_scenarios, get_scenario

SCENARIO_IDS = [s.scenario_id for s in get_all_scenarios()]
MONITOR = "MONITOR"
INVESTIGATE = "INVESTIGATE"
ESCALATE = "ESCALATE_TO_SECURITY_ANALYST"

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


def _all_results(repository):
    agent = LearningAgent(repository)
    return [(s.scenario_id, agent.decide(s)) for s in get_all_scenarios() for _ in range(2)]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_output_is_recommendation_only_with_no_executed_actions(seed_repository, scenario_id):
    result = LearningAgent(seed_repository).decide(get_scenario(scenario_id))
    assert result.metadata["executed_actions"] == [] and result.metadata["recommendation_only"] is True
    assert result.tool_calls == []
    assert "No disruptive action was executed" in result.audit_note


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_no_claim_of_blocking_disabling_quarantine_firewall_offense_or_containment(seed_repository, scenario_id):
    result = LearningAgent(seed_repository).decide(get_scenario(scenario_id))
    for text in _text_fields(result):
        assert not DISRUPTIVE_CLAIMS.search(text), text
    for step in result.recommended_actions:
        assert not OFFENSIVE_STEPS.search(step), step
        if "containment" in step.lower():
            assert "authorized" in step and "review" in step and "only if" in step


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_consequential_action_requires_human_authorization(seed_repository, scenario_id):
    result = LearningAgent(seed_repository).decide(get_scenario(scenario_id))
    assert result.metadata["human_authorization_required_for_consequential_action"] is True
    assert "human authorization" in result.audit_note
    chosen = next(c for c in LEARNING_CANDIDATES if c.decision == result.decision)
    if result.decision in (INVESTIGATE, ESCALATE):
        assert chosen.requires_human_authorization is True


def test_unconfirmed_compromise_is_never_described_as_confirmed(seed_repository):
    for scenario in get_all_scenarios():
        assert scenario.confirmed_impact is False
    for scenario_id, result in _all_results(seed_repository):
        for text in _text_fields(result):
            assert not COMPROMISE_CLAIMS.search(text), f"{scenario_id}: {text}"
        assert "NOT confirmed" in result.uncertainty


def test_learning_disclaims_real_world_meaning_in_every_result(seed_repository):
    for _, result in _all_results(seed_repository):
        note = result.metadata["limitation_note"]
        assert "simulated academic data" in note and "not real SOC statistics" in note and "no causal claim" in note


def unsafe(decision: str, side_effect: ProhibitedCondition) -> LearningCandidate:
    return LearningCandidate(action_id="LA-ACT-UNSAFE", decision=decision, side_effects=(side_effect,))


@pytest.mark.parametrize("side_effect", [c for c in ProhibitedCondition
                                         if c is not ProhibitedCondition.CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION])
def test_an_unsafe_action_with_perfect_history_is_never_selectable(tmp_path, monkeypatch, side_effect):
    # MONITOR is adequate for Scenario B, and its history is a flawless 100/0; but this MONITOR is unsafe.
    repository = ExperienceRepository.create(tmp_path / "e.json", {
        "low_severity_limited_evidence": {MONITOR: (100, 0), "COLLECT_MORE_EVIDENCE": (1, 9), INVESTIGATE: (1, 9)},
    })
    candidates = tuple(unsafe(MONITOR, side_effect) if c.decision == MONITOR else c for c in LEARNING_CANDIDATES)
    monkeypatch.setattr(learning_module, "LEARNING_CANDIDATES", candidates)
    result = LearningAgent(repository).decide(get_scenario("B"))
    by_id = {c["action_id"]: c for c in result.metadata["candidate_experience"]}
    assert by_id["LA-ACT-UNSAFE"]["eligible"] is False and by_id["LA-ACT-UNSAFE"]["success"] == 100
    assert any(side_effect.value in reason for reason in by_id["LA-ACT-UNSAFE"]["ineligibility_reasons"])
    assert result.metadata["selected_experience"]["action_id"] != "LA-ACT-UNSAFE"
    assert result.decision != MONITOR


def test_consequential_candidate_without_human_authorization_stays_ineligible(tmp_path, monkeypatch):
    repository = ExperienceRepository.create(tmp_path / "e.json", {
        "low_severity_limited_evidence": {MONITOR: (100, 0), "COLLECT_MORE_EVIDENCE": (1, 9), INVESTIGATE: (1, 9)}})
    risky = LearningCandidate(action_id="LA-ACT-RISKY", decision=MONITOR, is_consequential=True,
                              requires_human_authorization=False)
    monkeypatch.setattr(learning_module, "LEARNING_CANDIDATES",
                        tuple(risky if c.decision == MONITOR else c for c in LEARNING_CANDIDATES))
    result = LearningAgent(repository).decide(get_scenario("B"))
    chosen = next(c for c in result.metadata["candidate_experience"] if c["action_id"] == "LA-ACT-RISKY")
    assert chosen["eligible"] is False and result.metadata["selected_experience"]["action_id"] != "LA-ACT-RISKY"


def test_when_every_candidate_is_unsafe_the_agent_refuses_instead_of_guessing(tmp_path, monkeypatch):
    repository = ExperienceRepository.create(tmp_path / "e.json", {
        "critical_service_correlated_access": {ESCALATE: (100, 0)}})
    monkeypatch.setattr(learning_module, "LEARNING_CANDIDATES",
                        (unsafe(ESCALATE, ProhibitedCondition.AUTOMATIC_USER_BLOCKING),))
    with pytest.raises(NoSafeActionError):
        LearningAgent(repository).decide(get_scenario("C"))


@pytest.mark.parametrize("scenario_id, inadequate", [("A", MONITOR), ("A", "COLLECT_MORE_EVIDENCE"),
                                                       ("C", MONITOR), ("C", "COLLECT_MORE_EVIDENCE"),
                                                       ("C", INVESTIGATE), ("B", ESCALATE)])
def test_historical_success_never_overrides_current_adequacy_or_proportionality(tmp_path, scenario_id, inadequate):
    pattern = {"A": "privileged_login_anomaly", "B": "low_severity_limited_evidence",
               "C": "critical_service_correlated_access"}[scenario_id]
    repository = ExperienceRepository.create(tmp_path / "e.json", {pattern: {inadequate: (100, 0)}})
    result = LearningAgent(repository).decide(get_scenario(scenario_id))
    assert result.decision != inadequate
    assert next(c for c in result.metadata["candidate_experience"] if c["decision"] == inadequate)["eligible"] is False


def test_feedback_cannot_make_a_critical_scenario_selectable_as_monitor(tmp_path):
    repository = ExperienceRepository.create(tmp_path / "e.json", {
        "critical_service_correlated_access": {MONITOR: (0, 0), ESCALATE: (0, 0)}})
    agent = LearningAgent(repository)
    for _ in range(25):  # an enormous run of MONITOR "successes" and ESCALATE "failures"
        agent.record_feedback(get_scenario("C"), MONITOR, FeedbackOutcome.SUCCESS)
        agent.record_feedback(get_scenario("C"), ESCALATE, FeedbackOutcome.FAILURE)
    result = agent.decide(get_scenario("C"))
    assert result.decision == ESCALATE and result.priority.value == "CRITICAL"


def test_feedback_does_not_execute_or_authorize_any_security_action(seed_repository):
    update = LearningAgent(seed_repository).record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.SUCCESS)
    assert set(update.model_dump()) == {
        "pattern", "action", "outcome", "before_success", "before_failure", "after_success", "after_failure",
        "before_raw_rate", "after_raw_rate", "before_smoothed_rate", "after_smoothed_rate", "changed",
        "critic_assessment", "learning_explanation",
    }
    assert "Only this pattern/action entry was updated" in update.learning_explanation
