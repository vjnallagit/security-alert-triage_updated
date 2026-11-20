import ast
import builtins
import hashlib
import random
import sys
from pathlib import Path

import pytest
from agents import (
    GoalBasedAgent,
    LearningAgent,
    ModelBasedReflexAgent,
    SimpleReflexAgent,
    UtilityBasedAgent,
)
from agents import learning_agent as learning_module
from agents.base import BaseAgent
from agents.learning_agent import (
    LEARNING_CANDIDATES,
    MIN_EXPERIENCE_COUNT,
    Critic,
    InvalidFeedbackError,
    LearningElement,
    PerformanceElement,
    classify_pattern,
)
from models.learning_models import (
    CriticAssessment,
    ExperienceStats,
    FeedbackOutcome,
    LearningFeedback,
    LearningPattern,
    LearningUpdate,
    ProblemGeneratorGuidance,
)
from models.schemas import AgentDecision, Priority, SecurityScenario
from pydantic import ValidationError
from services.data_loader import DATA_DIR
from services.experience_repository import ExperienceRepository
from services.scenario_service import get_all_scenarios, get_scenario

PATTERN = "privileged_login_anomaly"
INVESTIGATE = "INVESTIGATE"
ESCALATE = "ESCALATE_TO_SECURITY_ANALYST"
COLLECT = "COLLECT_MORE_EVIDENCE"
MONITOR = "MONITOR"
EXPECTED = {
    "A": (Priority.HIGH, INVESTIGATE),
    "B": (Priority.LOW, COLLECT),
    "C": (Priority.CRITICAL, ESCALATE),
}


def smoothed(success: int, failure: int) -> float:
    return (success + 1) / (success + failure + 2)


def controlled_repository(tmp_path: Path, investigate=(4, 1), escalate=(3, 2)) -> ExperienceRepository:
    """The controlled starting experience for Scenario A's pattern (temporary store only)."""
    return ExperienceRepository.create(
        tmp_path / "controlled.json", {PATTERN: {INVESTIGATE: investigate, ESCALATE: escalate}}
    )


def store_bytes(repository: ExperienceRepository) -> bytes:
    return repository.path.read_bytes()


def variant(base_id: str = "A", **changes) -> SecurityScenario:
    return SecurityScenario.model_validate({**get_scenario(base_id).model_dump(mode="json"), **changes})


def candidates_by_decision(result: AgentDecision) -> dict[str, dict]:
    return {c["decision"]: c for c in result.metadata["candidate_experience"]}


# ===========================================================================
# Learning model tests (LM-01 .. LM-10)
# ===========================================================================


@pytest.mark.parametrize("success, failure", [(-1, 0), (0, -1), (-5, -5)])
def test_lm01_counts_cannot_be_negative(success, failure):
    with pytest.raises(ValidationError):
        ExperienceStats(success=success, failure=failure)


@pytest.mark.parametrize("outcome", ["MAYBE", "success", "", "WIN", 1])
def test_lm02_invalid_feedback_values_are_rejected(outcome):
    with pytest.raises(ValidationError):
        LearningFeedback(pattern=PATTERN, action=INVESTIGATE, outcome=outcome)


@pytest.mark.parametrize("pattern", ["", "   "])
def test_lm03_empty_pattern_is_rejected(pattern):
    with pytest.raises(ValidationError):
        LearningFeedback(pattern=pattern, action=INVESTIGATE, outcome=FeedbackOutcome.SUCCESS)


@pytest.mark.parametrize("action", ["BLOCK_ACCOUNT", "investigate", "", "ESCALATE", "QUARANTINE_HOST"])
def test_lm04_invalid_action_is_rejected(action):
    with pytest.raises(ValidationError):
        LearningFeedback(pattern=PATTERN, action=action, outcome=FeedbackOutcome.SUCCESS)


@pytest.mark.parametrize("success, failure, rate", [(8, 2, 0.8), (3, 1, 0.75), (0, 5, 0.0), (5, 0, 1.0)])
def test_lm05_raw_success_rate(success, failure, rate):
    assert ExperienceStats(success=success, failure=failure).raw_success_rate == pytest.approx(rate)


def test_lm06_zero_observations_are_handled_safely():
    stats = ExperienceStats(success=0, failure=0)
    assert stats.observation_count == 0
    assert stats.raw_success_rate is None  # no division by zero, and not a misleading 0.0
    assert stats.smoothed_success_rate == pytest.approx(0.5)


@pytest.mark.parametrize("success, failure, expected", [
    (8, 2, 9 / 12), (0, 0, 1 / 2), (1, 0, 2 / 3), (4, 1, 5 / 7), (3, 2, 4 / 7), (4, 2, 5 / 8),
])
def test_lm07_beta_one_one_smoothing(success, failure, expected):
    assert ExperienceStats(success=success, failure=failure).smoothed_success_rate == pytest.approx(expected)
    assert ExperienceStats(success=8, failure=2).smoothed_success_rate == pytest.approx(0.75)


def test_lm07_smoothing_prevents_extreme_preferences_from_tiny_samples():
    one_win = ExperienceStats(success=1, failure=0)
    assert one_win.raw_success_rate == 1.0 and one_win.smoothed_success_rate < 0.7


def test_stats_after_outcome():
    stats = ExperienceStats(success=4, failure=1)
    assert stats.after(FeedbackOutcome.SUCCESS) == ExperienceStats(success=5, failure=1)
    assert stats.after(FeedbackOutcome.FAILURE) == ExperienceStats(success=4, failure=2)
    assert stats.after(FeedbackOutcome.UNKNOWN) == stats


def test_lm08_learning_update_reports_before_and_after(tmp_path):
    agent = LearningAgent(controlled_repository(tmp_path))
    update = agent.record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.FAILURE)
    assert (update.before_success, update.before_failure) == (4, 1)
    assert (update.after_success, update.after_failure) == (4, 2)
    assert update.before_raw_rate == pytest.approx(0.8) and update.after_raw_rate == pytest.approx(4 / 6)
    assert update.before_smoothed_rate == pytest.approx(5 / 7) and update.after_smoothed_rate == pytest.approx(5 / 8)
    assert update.changed is True and update.critic_assessment.usable is True
    assert "FAILURE" in update.learning_explanation and "4 -> 4" in update.learning_explanation


def test_lm08_update_model_rejects_inconsistent_counts():
    usable = CriticAssessment(usable=True, outcome=FeedbackOutcome.SUCCESS, reason="ok")
    base = dict(pattern=PATTERN, action=INVESTIGATE, outcome=FeedbackOutcome.SUCCESS, before_success=1,
                before_failure=1, before_raw_rate=0.5, after_raw_rate=0.5, before_smoothed_rate=0.5,
                after_smoothed_rate=0.5, critic_assessment=usable, learning_explanation="x")
    assert LearningUpdate(after_success=2, after_failure=1, changed=True, **base)
    with pytest.raises(ValidationError):  # a SUCCESS cannot raise the failure count
        LearningUpdate(after_success=1, after_failure=2, changed=True, **base)
    with pytest.raises(ValidationError):  # a jump of two is not one piece of feedback
        LearningUpdate(after_success=3, after_failure=1, changed=True, **base)
    with pytest.raises(ValidationError):  # "unchanged" must mean unchanged
        LearningUpdate(after_success=2, after_failure=1, changed=False, **base)


def test_lm09_unknown_feedback_reports_changed_false(tmp_path):
    repository = controlled_repository(tmp_path)
    update = LearningAgent(repository).record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.UNKNOWN)
    assert update.changed is False and update.critic_assessment.usable is False
    assert (update.before_success, update.before_failure) == (update.after_success, update.after_failure) == (4, 1)
    assert "not counted" in update.learning_explanation


def test_lm10_learning_models_reject_unexpected_fields_and_are_immutable():
    with pytest.raises(ValidationError):
        LearningFeedback(pattern=PATTERN, action=INVESTIGATE, outcome="SUCCESS", success_rate=0.9)
    with pytest.raises(ValidationError):
        ExperienceStats(success=1, failure=1, rate=0.5)
    with pytest.raises(ValidationError):
        ExperienceStats(success=1, failure=1).success = 5


def test_problem_generator_guidance_model_is_consistent():
    with pytest.raises(ValidationError):
        ProblemGeneratorGuidance(min_experience_count=3, experience_sufficient=True, used=True, guidance="g")
    with pytest.raises(ValidationError):
        ProblemGeneratorGuidance(min_experience_count=3, experience_sufficient=False, used=True, guidance="g")


# ===========================================================================
# Core Learning Agent (LA-01 .. LA-17)
# ===========================================================================


def test_la01_inherits_directly_from_base_agent():
    assert LearningAgent.__bases__ == (BaseAgent,)
    assert isinstance(LearningAgent(), BaseAgent)


def test_la02_is_not_a_subclass_of_any_other_classical_agent():
    others = (SimpleReflexAgent, ModelBasedReflexAgent, GoalBasedAgent, UtilityBasedAgent)
    assert not issubclass(LearningAgent, others)
    assert not any(issubclass(other, LearningAgent) for other in others)


def test_la03_accepts_an_injected_repository(tmp_path):
    repository = controlled_repository(tmp_path)
    assert LearningAgent(repository)._repository is repository
    assert LearningAgent()._repository.path == DATA_DIR / "experience.json"  # default: committed seed


def test_la04_la05_la06_la07_patterns_come_from_facts(seed_repository):
    assert classify_pattern(get_scenario("A")) is LearningPattern.PRIVILEGED_LOGIN_ANOMALY
    assert classify_pattern(get_scenario("B")) is LearningPattern.LOW_SEVERITY_LIMITED_EVIDENCE
    assert classify_pattern(get_scenario("C")) is LearningPattern.CRITICAL_SERVICE_CORRELATED_ACCESS
    agent = LearningAgent(seed_repository)
    assert [agent._classify_pattern(get_scenario(i)) for i in "ABC"] == [
        "privileged_login_anomaly", "low_severity_limited_evidence", "critical_service_correlated_access",
    ]


def test_la08_renaming_a_scenario_does_not_change_its_pattern():
    for original in "ABC":
        reference = classify_pattern(get_scenario(original))
        for new_id in ("A", "B", "C", "ZZZ", "scenario-9"):
            assert classify_pattern(variant(original, scenario_id=new_id)) == reference


def test_pattern_changes_with_facts_and_has_a_safe_fallback():
    assert classify_pattern(variant("A", source_familiarity="KNOWN")) is LearningPattern.GENERAL_SECURITY_ALERT
    assert classify_pattern(variant("A", failed_login_count=1)) is LearningPattern.GENERAL_SECURITY_ALERT
    assert classify_pattern(variant("B", severity="MEDIUM")) is LearningPattern.GENERAL_SECURITY_ALERT
    assert classify_pattern(variant("C", correlated_alert_count=1)) is LearningPattern.GENERAL_SECURITY_ALERT
    # A critical service with a privileged-login pattern is classified as the more serious critical pattern.
    both = variant("A", asset_criticality="CRITICAL", correlated_alert_count=3)
    assert classify_pattern(both) is LearningPattern.CRITICAL_SERVICE_CORRELATED_ACCESS


@pytest.mark.parametrize("scenario_id", "ABC")
def test_la09_la10_la11_la12_la13_results_validate_and_declare_learning(seed_repository, scenario_id):
    result = LearningAgent(seed_repository).decide(get_scenario(scenario_id))
    assert AgentDecision.model_validate(result.model_dump()) == result
    assert result.agent_type.value == "LEARNING" and result.agent_name == "Learning Agent"
    assert result.experience_used and result.experience_used["pattern"] == result.metadata["learning_pattern"]
    assert result.utility_score is None and result.tool_calls == []
    assert result.metadata["decision_mechanism"] == "feedback_driven_experience_learning"
    assert result.metadata["learning_components"] == {
        "performance_element": True, "critic": True, "learning_element": True, "problem_generator": True,
    }
    assert result.metadata["uses_learning"] is True
    assert result.metadata["uses_utility_scoring"] is False and result.metadata["uses_llm"] is False
    assert result.reasoning and result.evidence and result.uncertainty
    assert result.likely_interpretation and result.recommended_actions and result.audit_note


@pytest.mark.parametrize("scenario_id", "ABC")
def test_la14_la15_candidate_histories_carry_counts_and_rates(seed_repository, scenario_id):
    result = LearningAgent(seed_repository).decide(get_scenario(scenario_id))
    history = seed_repository.get_pattern_history(result.metadata["learning_pattern"])
    assert [c["decision"] for c in result.metadata["candidate_experience"]] == [
        MONITOR, COLLECT, INVESTIGATE, ESCALATE,
    ]
    for candidate in result.metadata["candidate_experience"]:
        stats = history[candidate["decision"]]
        assert (candidate["success"], candidate["failure"]) == (stats.success, stats.failure)
        assert candidate["observation_count"] == stats.observation_count
        assert candidate["raw_success_rate"] == pytest.approx(stats.raw_success_rate)
        assert candidate["smoothed_success_rate"] == pytest.approx(smoothed(stats.success, stats.failure))
    assert result.metadata["selected_experience"]["decision"] == result.decision


def test_la16_reasoning_explains_exactly_how_experience_affected_selection(seed_repository):
    result = LearningAgent(seed_repository).decide(get_scenario("A"))
    reasoning = result.reasoning
    assert "Pattern privileged_login_anomaly matched" in reasoning
    assert "INVESTIGATE has 8 successes and 2 failures" in reasoning
    assert "raw historical success rate is 80.0%" in reasoning and "Beta(1,1)-smoothed estimate is 75.0%" in reasoning
    assert "ESCALATE_TO_SECURITY_ANALYST has 5 successes and 3 failures" in reasoning
    assert "excluded by current-scenario constraints" in reasoning  # MONITOR / COLLECT_MORE_EVIDENCE
    assert "Historical experience therefore supports INVESTIGATE (75.0%) over the other eligible alternatives" in reasoning
    assert "Historical experience is sufficient" in reasoning
    assert "did not change experience" in reasoning


@pytest.mark.parametrize("scenario_id", "ABC")
def test_la17_decide_alone_never_modifies_experience(seed_repository, scenario_id):
    agent = LearningAgent(seed_repository)
    before = store_bytes(seed_repository)
    for _ in range(3):
        agent.decide(get_scenario(scenario_id))
    assert store_bytes(seed_repository) == before


# ===========================================================================
# Baseline scenarios with a clean copy of the seed experience
# ===========================================================================


@pytest.mark.parametrize("scenario_id", "ABC")
def test_baseline_decisions_come_from_facts_eligibility_and_seed_experience(seed_repository, scenario_id):
    result = LearningAgent(seed_repository).decide(get_scenario(scenario_id))
    assert (result.priority, result.decision) == EXPECTED[scenario_id]
    assert "NOT confirmed" in result.uncertainty and "Confirmed impact: False" in result.evidence
    assert "Compromise is not confirmed" in result.reasoning


def test_baseline_a_is_chosen_by_history_among_eligible_actions(seed_repository):
    result = LearningAgent(seed_repository).decide(get_scenario("A"))
    candidates = candidates_by_decision(result)
    assert {d for d, c in candidates.items() if c["eligible"]} == {INVESTIGATE, ESCALATE}
    assert result.metadata["selection_basis"] == "safe_action_then_smoothed_historical_success"
    assert candidates[INVESTIGATE]["smoothed_success_rate"] > candidates[ESCALATE]["smoothed_success_rate"]
    assert result.metadata["experience_changed_default"] is False and result.metadata["problem_generator_used"] is False


def test_baseline_b_prefers_collect_more_evidence_from_history(seed_repository):
    result = LearningAgent(seed_repository).decide(get_scenario("B"))
    candidates = candidates_by_decision(result)
    assert {d for d, c in candidates.items() if c["eligible"]} == {MONITOR, COLLECT, INVESTIGATE}
    assert candidates[ESCALATE]["eligible"] is False
    assert any("Disproportionate" in r for r in candidates[ESCALATE]["ineligibility_reasons"])
    assert result.metadata["ranked_decisions"][0] == COLLECT


def test_baseline_c_has_a_single_eligible_action_and_history_cannot_change_it(seed_repository):
    result = LearningAgent(seed_repository).decide(get_scenario("C"))
    assert result.metadata["selection_basis"] == "only_eligible_safe_action"
    assert [d for d, c in candidates_by_decision(result).items() if c["eligible"]] == [ESCALATE]
    assert "only eligible action" in result.reasoning


def test_decisions_do_not_depend_on_scenario_ids(seed_repository):
    agent = LearningAgent(seed_repository)
    for original in "ABC":
        reference = agent.decide(get_scenario(original))
        for new_id in ("ZZZ", "A", "B", "C"):
            renamed = agent.decide(variant(original, scenario_id=new_id))
            assert (renamed.priority, renamed.decision) == (reference.priority, reference.decision)
            assert renamed.metadata["candidate_experience"] == reference.metadata["candidate_experience"]


def test_priority_is_grounded_in_current_risk_not_in_history(tmp_path):
    # History that loves INVESTIGATE for a low-risk pattern must not raise a LOW alert's priority.
    repository = ExperienceRepository.create(
        tmp_path / "e.json",
        {"low_severity_limited_evidence": {INVESTIGATE: (50, 0), COLLECT: (0, 50), MONITOR: (0, 50)}},
    )
    result = LearningAgent(repository).decide(get_scenario("B"))
    assert result.decision == INVESTIGATE and result.priority is Priority.LOW
    # ... and a critical scenario cannot be pulled down by history.
    critical = ExperienceRepository.create(
        tmp_path / "c.json", {"critical_service_correlated_access": {ESCALATE: (0, 50), MONITOR: (50, 0)}}
    )
    low = LearningAgent(critical).decide(get_scenario("C"))
    assert low.decision == ESCALATE and low.priority is Priority.CRITICAL


def test_general_fallback_pattern_works_with_seed_experience(seed_repository):
    result = LearningAgent(seed_repository).decide(variant("B", severity="MEDIUM", evidence_strength="MODERATE"))
    assert result.metadata["learning_pattern"] == "general_security_alert"
    assert result.priority is Priority.MEDIUM and result.decision == INVESTIGATE


# ===========================================================================
# Feedback and learning (LF-01 .. LF-10)
# ===========================================================================


def test_lf01_lf02_success_and_failure_each_increment_exactly_one_count(tmp_path):
    agent = LearningAgent(controlled_repository(tmp_path))
    success = agent.record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.SUCCESS)
    assert (success.after_success, success.after_failure) == (5, 1)
    failure = agent.record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.FAILURE)
    assert (failure.after_success, failure.after_failure) == (5, 2)


def test_lf03_feedback_affects_only_the_matching_pattern_and_action(tmp_path):
    repository = ExperienceRepository.create(tmp_path / "e.json", {
        PATTERN: {INVESTIGATE: (4, 1), ESCALATE: (3, 2)},
        "low_severity_limited_evidence": {INVESTIGATE: (9, 9)},
    })
    before = repository.load().model_dump()
    LearningAgent(repository).record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.FAILURE)
    after = repository.load().model_dump()
    assert after["records"][PATTERN][ESCALATE] == before["records"][PATTERN][ESCALATE]
    assert after["records"]["low_severity_limited_evidence"] == before["records"]["low_severity_limited_evidence"]
    assert after["records"][PATTERN][INVESTIGATE] == {"success": 4, "failure": 2}


def test_lf04_lf05_lf06_rates_change_correctly_and_update_returns_before_and_after(tmp_path):
    agent = LearningAgent(controlled_repository(tmp_path))
    update = agent.record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.SUCCESS)
    assert update.before_raw_rate == pytest.approx(4 / 5) and update.after_raw_rate == pytest.approx(5 / 6)
    assert update.before_smoothed_rate == pytest.approx(5 / 7) and update.after_smoothed_rate == pytest.approx(6 / 8)
    assert update.after_smoothed_rate > update.before_smoothed_rate


def test_lf07_unknown_feedback_does_not_modify_the_store(tmp_path):
    repository = controlled_repository(tmp_path)
    before = store_bytes(repository)
    update = LearningAgent(repository).record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.UNKNOWN)
    assert update.changed is False and store_bytes(repository) == before


def test_lf08_invalid_feedback_does_not_modify_the_store(tmp_path):
    repository = controlled_repository(tmp_path)
    agent = LearningAgent(repository)
    before = store_bytes(repository)
    with pytest.raises(ValidationError):
        agent.record_feedback(get_scenario("A"), "BLOCK_ACCOUNT", FeedbackOutcome.SUCCESS)
    with pytest.raises(ValidationError):
        agent.record_feedback(get_scenario("A"), INVESTIGATE, "MAYBE")  # type: ignore[arg-type]
    with pytest.raises(InvalidFeedbackError, match="Unknown learning pattern"):
        agent.learn(LearningFeedback(pattern="typo_pattern", action=INVESTIGATE, outcome=FeedbackOutcome.SUCCESS))
    assert store_bytes(repository) == before


def test_lf09_decide_after_feedback_reads_the_updated_experience(tmp_path):
    agent = LearningAgent(controlled_repository(tmp_path))
    before = candidates_by_decision(agent.decide(get_scenario("A")))[INVESTIGATE]
    agent.record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.FAILURE)
    after = candidates_by_decision(agent.decide(get_scenario("A")))[INVESTIGATE]
    assert (before["failure"], after["failure"]) == (1, 2)
    assert after["smoothed_success_rate"] < before["smoothed_success_rate"]


def test_lf10_experience_survives_a_new_agent_instance(tmp_path):
    repository = controlled_repository(tmp_path)
    LearningAgent(repository).record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.FAILURE)
    fresh = LearningAgent(ExperienceRepository(repository.path))
    assert candidates_by_decision(fresh.decide(get_scenario("A")))[INVESTIGATE]["failure"] == 2


def test_feedback_for_a_new_pattern_action_pair_is_initialised_safely(tmp_path):
    repository = controlled_repository(tmp_path)
    agent = LearningAgent(repository)
    update = agent.record_feedback(get_scenario("A"), COLLECT, FeedbackOutcome.FAILURE)
    assert (update.before_success, update.before_failure, update.after_failure) == (0, 0, 1)
    assert update.before_raw_rate is None and update.before_smoothed_rate == pytest.approx(0.5)


def test_critic_never_modifies_experience_and_learning_element_obeys_it(tmp_path):
    repository = controlled_repository(tmp_path)
    before = store_bytes(repository)
    feedback = LearningFeedback(pattern=PATTERN, action=INVESTIGATE, outcome=FeedbackOutcome.FAILURE)
    assessment = Critic.evaluate(feedback, {PATTERN})
    assert assessment.usable and store_bytes(repository) == before  # the critic only judges
    unusable = CriticAssessment(usable=False, outcome=FeedbackOutcome.FAILURE, reason="rejected")
    update = LearningElement(repository).apply(feedback, unusable)
    assert update.changed is False and store_bytes(repository) == before
    assert LearningElement(repository).apply(feedback, assessment).changed is True


def test_feedback_is_recorded_only_through_explicit_calls(tmp_path, monkeypatch):
    repository = controlled_repository(tmp_path)
    monkeypatch.setattr(ExperienceRepository, "record_outcome",
                        lambda *a, **k: pytest.fail("decide() must never record an outcome"))
    agent = LearningAgent(repository)
    for scenario in get_all_scenarios():
        agent.decide(scenario)


# ===========================================================================
# Mandatory behavior-change tests
# ===========================================================================


def test_feedback_eventually_changes_future_decision(tmp_path):
    repository = controlled_repository(tmp_path)  # INVESTIGATE 4/1 (5/7) vs ESCALATE 3/2 (4/7)
    agent = LearningAgent(repository)
    scenario = get_scenario("A")
    scenario_before = scenario.model_copy(deep=True)
    code_before = (PerformanceElement.select.__code__.co_code, LEARNING_CANDIDATES)
    escalate_before = repository.get_stats(PATTERN, ESCALATE)

    # Derived from the mathematics, not hard-coded: the first failure count at which
    # ESCALATE's smoothed estimate overtakes INVESTIGATE's.
    expected_switch = next(f for f in range(1, 11) if smoothed(4, 1 + f) < smoothed(3, 2))

    initial = agent.decide(scenario)
    assert (initial.priority, initial.decision) == (Priority.HIGH, INVESTIGATE)

    switched_after = None
    for count in range(1, 11):  # safe maximum of 10 feedback events
        before = repository.get_stats(PATTERN, INVESTIGATE)
        update = agent.record_feedback(scenario, INVESTIGATE, FeedbackOutcome.FAILURE)
        assert update.after_failure == before.failure + 1 and update.after_success == before.success
        assert update.after_smoothed_rate < update.before_smoothed_rate  # INVESTIGATE gets worse each time
        assert repository.get_stats(PATTERN, INVESTIGATE).failure == 1 + count
        result = agent.decide(scenario)  # the SAME scenario object every time
        if result.decision == ESCALATE:
            switched_after = count
            break
        assert result.decision == INVESTIGATE
    assert switched_after is not None, "decision never changed within 10 failure events"
    assert switched_after == expected_switch == 2

    assert (result.priority, result.decision) == (Priority.HIGH, ESCALATE)  # priority stays grounded in risk
    assert result.metadata["experience_changed_default"] is True
    assert "accumulated outcome feedback changed the preference" in result.reasoning
    # Only experience changed.
    assert scenario == scenario_before
    assert (PerformanceElement.select.__code__.co_code, LEARNING_CANDIDATES) == code_before
    assert repository.get_stats(PATTERN, ESCALATE) == escalate_before  # the competitor was never touched


def test_behavior_change_with_the_committed_seed_needs_four_failures_and_uses_the_tie_breaker(seed_repository):
    agent = LearningAgent(seed_repository)
    scenario = get_scenario("A")
    needed = next(f for f in range(1, 11) if smoothed(8, 2 + f) < smoothed(5, 3))
    assert needed == 4
    decisions = []
    for _ in range(needed):
        agent.record_feedback(scenario, INVESTIGATE, FeedbackOutcome.FAILURE)
        decisions.append(agent.decide(scenario))
    assert [d.decision for d in decisions] == [INVESTIGATE, INVESTIGATE, INVESTIGATE, ESCALATE]
    # After 3 failures the estimates are exactly tied (9/15 == 6/10); the documented tie-breaker keeps the default.
    assert decisions[2].metadata["tie_breaker_used"] is True
    assert "tie-breaker" in decisions[2].reasoning


def test_success_feedback_strengthens_future_preference(tmp_path):
    repository = controlled_repository(tmp_path)
    agent = LearningAgent(repository)
    scenario = get_scenario("A")
    previous = candidates_by_decision(agent.decide(scenario))
    previous_margin = previous[INVESTIGATE]["smoothed_success_rate"] - previous[ESCALATE]["smoothed_success_rate"]
    for _ in range(3):
        update = agent.record_feedback(scenario, INVESTIGATE, FeedbackOutcome.SUCCESS)
        assert update.after_success == update.before_success + 1 and update.after_failure == update.before_failure
        assert update.after_smoothed_rate > update.before_smoothed_rate
        result = agent.decide(scenario)
        current = candidates_by_decision(result)
        margin = current[INVESTIGATE]["smoothed_success_rate"] - current[ESCALATE]["smoothed_success_rate"]
        assert current[INVESTIGATE]["smoothed_success_rate"] > previous[INVESTIGATE]["smoothed_success_rate"]
        assert margin > previous_margin  # the preference strengthens
        assert result.decision == INVESTIGATE
        previous, previous_margin = current, margin


def test_success_feedback_for_the_runner_up_can_also_change_the_decision(tmp_path):
    repository = controlled_repository(tmp_path)  # INVESTIGATE 5/7 vs ESCALATE 4/7
    agent = LearningAgent(repository)
    needed = next(n for n in range(1, 11) if smoothed(3 + n, 2) > smoothed(4, 1))
    for count in range(1, needed + 1):
        assert agent.decide(get_scenario("A")).decision == (INVESTIGATE if count <= needed else ESCALATE)
        agent.record_feedback(get_scenario("A"), ESCALATE, FeedbackOutcome.SUCCESS)
    assert needed == 4
    assert agent.decide(get_scenario("A")).decision == ESCALATE


# ===========================================================================
# Problem Generator (PG-01 .. PG-06)
# ===========================================================================


def sparse_repository(tmp_path: Path, records=None) -> ExperienceRepository:
    return ExperienceRepository.create(tmp_path / "sparse.json", records or {"unrelated": {MONITOR: (0, 0)}})


def test_pg01_experience_below_the_minimum_is_marked_insufficient(tmp_path):
    result = LearningAgent(sparse_repository(tmp_path)).decide(get_scenario("B"))
    assert result.metadata["experience_sufficient"] is False
    assert all(c["sufficient"] is False for c in result.metadata["candidate_experience"])
    assert MIN_EXPERIENCE_COUNT == 3
    two_each = ExperienceRepository.create(tmp_path / "two.json", {
        "low_severity_limited_evidence": {COLLECT: (1, 1), MONITOR: (2, 0), INVESTIGATE: (0, 2)}})
    assert LearningAgent(two_each).decide(get_scenario("B")).metadata["experience_sufficient"] is False


def test_pg02_guidance_is_populated(tmp_path):
    result = LearningAgent(sparse_repository(tmp_path)).decide(get_scenario("B"))
    guidance = result.metadata["problem_generator"]
    assert result.metadata["problem_generator_used"] is True and guidance["used"] is True
    assert "Historical experience is insufficient" in guidance["guidance"]
    assert "rather than treating weak experience as reliable" in guidance["guidance"]
    assert guidance["insufficient_decisions"] and guidance["min_experience_count"] == 3
    assert guidance["guidance"] in result.reasoning


def test_pg03_weak_experience_encourages_collect_more_evidence_where_safe(tmp_path):
    result = LearningAgent(sparse_repository(tmp_path)).decide(get_scenario("B"))
    assert (result.priority, result.decision) == (Priority.LOW, COLLECT)
    assert result.metadata["selection_basis"] == "problem_generator_evidence_gathering"
    assert result.metadata["problem_generator"]["recommended_decision"] == COLLECT


def test_pg05_current_critical_risk_overrides_exploration(tmp_path):
    repository = sparse_repository(tmp_path)
    critical = LearningAgent(repository).decide(get_scenario("C"))
    assert critical.decision == ESCALATE and critical.decision != COLLECT
    assert (critical.priority) is Priority.CRITICAL
    assert candidates_by_decision(critical)[COLLECT]["eligible"] is False
    # Scenario A under cold start: exploration cannot replace an active response either.
    privileged = LearningAgent(repository).decide(get_scenario("A"))
    assert privileged.decision == INVESTIGATE
    guidance = privileged.metadata["problem_generator"]
    assert guidance["recommended_decision"] is None and guidance["overridden_by_current_risk"] is True
    assert privileged.metadata["selection_basis"] == "deterministic_safe_fallback"


def test_pg04_problem_generator_only_ever_recommends_defensive_actions(tmp_path):
    allowed = {MONITOR, COLLECT, INVESTIGATE, ESCALATE}
    for scenario in get_all_scenarios():
        guidance = LearningAgent(sparse_repository(tmp_path)).decide(scenario).metadata["problem_generator"]
        assert guidance["recommended_decision"] in allowed | {None}
        assert not any(word in guidance["guidance"].lower() for word in ("block", "disable", "quarantine", "attack", "firewall"))


def test_pg06_problem_generator_does_not_use_randomness(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("randomness used")

    for name in ("random", "choice", "uniform", "shuffle", "sample", "randint"):
        monkeypatch.setattr(random, name, forbidden)
    repository = sparse_repository(tmp_path)
    runs = [[LearningAgent(repository).decide(s) for s in get_all_scenarios()] for _ in range(2)]
    assert runs[0] == runs[1]


def test_sufficient_overall_but_thin_candidates_are_reported_not_penalised(tmp_path):
    repository = ExperienceRepository.create(tmp_path / "mixed.json", {
        PATTERN: {INVESTIGATE: (8, 2), ESCALATE: (1, 0)}})  # ESCALATE has 1 observation
    result = LearningAgent(repository).decide(get_scenario("A"))
    guidance = result.metadata["problem_generator"]
    assert guidance["experience_sufficient"] is True and guidance["used"] is False
    assert guidance["insufficient_decisions"] == [ESCALATE]
    assert result.decision == INVESTIGATE  # 0.75 vs smoothed 2/3


# ===========================================================================
# Architectural differentiation (LD-01 .. LD-10)
# ===========================================================================


def committed_digest() -> str:
    return hashlib.sha256((DATA_DIR / "experience.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("agent_class", [SimpleReflexAgent, ModelBasedReflexAgent, GoalBasedAgent, UtilityBasedAgent])
def test_ld01_to_ld04_other_agents_do_not_use_or_update_experience_feedback(agent_class):
    before = committed_digest()
    agent = agent_class()
    for scenario in get_all_scenarios() * 2:
        result = agent.decide(scenario)
        assert result.experience_used is None
        assert result.metadata.get("uses_learning", False) is False
        assert "learning_pattern" not in result.metadata and "candidate_experience" not in result.metadata
    assert committed_digest() == before


def test_ld05_ld06_learning_agent_calls_no_other_agent_and_computes_no_utility(seed_repository, monkeypatch):
    def forbidden(self, scenario):
        raise AssertionError("another agent's decide() was called")

    for other in (SimpleReflexAgent, ModelBasedReflexAgent, GoalBasedAgent, UtilityBasedAgent):
        monkeypatch.setattr(other, "decide", forbidden)
    for scenario in get_all_scenarios():
        result = LearningAgent(seed_repository).decide(scenario)
        assert result.utility_score is None
        assert not {"action_analyses", "expected_utility_formula", "utility_profile"} & set(result.metadata)
    for name in ("UtilityBasedAgent", "GoalBasedAgent", "SimpleReflexAgent", "ModelBasedReflexAgent",
                 "DEFAULT_UTILITY_PROFILE"):
        assert not hasattr(learning_module, name)


def test_ld07_no_llm_or_ui_dependency_is_introduced(seed_repository):
    # LangChain is now a legitimate dependency of the separate LLM layer and may already be loaded by other
    # tests, so check what THIS agent imports and loads rather than the global module table.
    before = set(sys.modules)
    LearningAgent(seed_repository).decide(get_scenario("A"))
    forbidden = {"langchain", "langchain_core", "langchain_ollama", "ollama", "streamlit"}
    assert not forbidden & (set(sys.modules) - before)
    tree = ast.parse((Path(__file__).resolve().parents[1] / "agents" / "learning_agent.py").read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not forbidden & imported
    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8").lower()
    # Streamlit is now the declared UI dependency (separate from the agents); the ML libraries stay excluded.
    assert not any(n in requirements for n in ("scikit", "numpy", "pandas", "torch", "tensorflow"))


def test_ld08_no_random_exploration(seed_repository, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("randomness used")

    for name in ("random", "choice", "uniform", "shuffle", "sample", "randint", "gauss"):
        monkeypatch.setattr(random, name, forbidden)
    assert not any(hasattr(learning_module, n) for n in ("random", "secrets", "time"))
    for scenario in get_all_scenarios():
        LearningAgent(seed_repository).decide(scenario)
        LearningAgent(seed_repository).record_feedback(scenario, INVESTIGATE, FeedbackOutcome.UNKNOWN)


def test_ld09_decision_changes_only_when_facts_or_experience_change(tmp_path):
    repository = controlled_repository(tmp_path)
    agent = LearningAgent(repository)
    scenario = get_scenario("A")
    first = agent.decide(scenario)
    assert agent.decide(scenario) == first == LearningAgent(repository).decide(scenario)  # nothing changed
    assert agent.decide(variant("A", scenario_id="OTHER")).decision == first.decision

    different_facts = agent.decide(variant("A", source_familiarity="KNOWN"))  # facts change the pattern
    assert different_facts.metadata["learning_pattern"] != first.metadata["learning_pattern"]

    agent.record_feedback(scenario, INVESTIGATE, FeedbackOutcome.FAILURE)
    agent.record_feedback(scenario, INVESTIGATE, FeedbackOutcome.FAILURE)  # experience changes
    assert agent.decide(scenario).decision != first.decision


def test_ld10_different_instances_using_the_same_repository_observe_learning(tmp_path):
    repository = controlled_repository(tmp_path)
    teacher, student = LearningAgent(repository), LearningAgent(ExperienceRepository(repository.path))
    assert student.decide(get_scenario("A")).decision == INVESTIGATE
    for _ in range(2):
        teacher.record_feedback(get_scenario("A"), INVESTIGATE, FeedbackOutcome.FAILURE)
    assert student.decide(get_scenario("A")).decision == ESCALATE


def test_agent_holds_no_decision_history_only_its_repository_and_components(seed_repository):
    agent = LearningAgent(seed_repository)
    before = set(vars(agent))
    for scenario in get_all_scenarios() * 2:
        agent.decide(scenario)
    assert set(vars(agent)) == before == {
        "_repository", "_performance_element", "_critic", "_learning_element", "_problem_generator",
    }


def test_ld_decide_does_not_touch_the_filesystem_for_writing(seed_repository, monkeypatch):
    scenarios = get_all_scenarios()
    agent = LearningAgent(seed_repository)
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("open() used")))
    for name in ("write_text", "write_bytes"):
        monkeypatch.setattr(Path, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("write used")))
    import os

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(AssertionError("replace used")))
    for scenario in scenarios:
        agent.decide(scenario)  # reads through the repository; any write would fail the test
