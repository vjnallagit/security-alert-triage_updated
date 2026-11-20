import ast
import builtins
import math
import random
import sys
from pathlib import Path

import pytest
from agents import GoalBasedAgent, ModelBasedReflexAgent, SimpleReflexAgent, UtilityBasedAgent
from agents.base import BaseAgent
from agents.utility_based import (
    CANDIDATE_ACTIONS,
    DEFAULT_UTILITY_PROFILE,
    NoEligibleActionError,
)
from models.goal_models import ProhibitedCondition
from models.schemas import AgentDecision, Priority, SecurityScenario
from models.utility_models import (
    TOLERANCE,
    ActionExpectedUtility,
    CriterionContribution,
    CriterionName,
    DecisionContext,
    OutcomeCategory,
    OutcomeState,
    OutcomeUtilityBreakdown,
    ProbabilisticOutcome,
    UtilityAction,
    UtilityCriterion,
    UtilityDecisionAnalysis,
    UtilityProfile,
    UtilityRanking,
    UtilityStep,
)
from pydantic import ValidationError
from services.data_loader import DATA_DIR
from services.scenario_service import get_all_scenarios, get_scenario

WEIGHTS = {
    CriterionName.RISK_REDUCTION: 0.25,
    CriterionName.ASSET_PROTECTION: 0.20,
    CriterionName.EVIDENCE_ALIGNMENT: 0.20,
    CriterionName.PROPORTIONALITY: 0.15,
    CriterionName.HUMAN_CONTROL: 0.10,
    CriterionName.OPERATIONAL_CONTINUITY: 0.10,
}


# ---------------------------------------------------------------------------
# Builders: valid instances that individual tests corrupt
# ---------------------------------------------------------------------------


def make_criteria(weights=None) -> tuple[UtilityCriterion, ...]:
    weights = weights or WEIGHTS
    return tuple(
        UtilityCriterion(criterion_id=f"UC-0{i}", name=name, description="d", weight=weight)
        for i, (name, weight) in enumerate(weights.items(), start=1)
    )


def make_profile(weights=None, **overrides) -> UtilityProfile:
    data = {"profile_id": "UP-TEST", "name": "n", "description": "d", "criteria": make_criteria(weights)}
    return UtilityProfile(**{**data, **overrides})


def make_state(name=OutcomeCategory.FAVORABLE, score=0.5, **overrides) -> OutcomeState:
    data = {
        "outcome_id": f"OUT-{name.value}", "name": name, "description": "d",
        "risk_reduction": score, "asset_protection": score, "evidence_alignment": score,
        "proportionality": score, "human_control": score, "operational_continuity": score,
    }
    return OutcomeState(**{**data, **overrides})


def make_breakdown(state: OutcomeState, profile: UtilityProfile | None = None) -> OutcomeUtilityBreakdown:
    profile = profile or make_profile()
    parts = tuple(
        CriterionContribution(
            criterion=c.name, raw_score=state.score(c.name), weight=c.weight,
            weighted_contribution=state.score(c.name) * c.weight,
        )
        for c in profile.criteria
    )
    return OutcomeUtilityBreakdown(
        outcome_id=state.outcome_id, contributions=parts,
        outcome_utility=math.fsum(p.weighted_contribution for p in parts),
    )


def make_eu(action_id="ACT-1", decision="INVESTIGATE", probabilities=(0.7, 0.3), scores=(0.9, 0.4),
            eligible=True, rank=None, **overrides) -> ActionExpectedUtility:
    names = (OutcomeCategory.FAVORABLE, OutcomeCategory.UNFAVORABLE)
    states = [make_state(n, s, outcome_id=f"{action_id}-{n.value}") for n, s in zip(names, scores)]
    outcomes = tuple(
        ProbabilisticOutcome(outcome_id=s.outcome_id, action_id=action_id, probability=p, outcome_state=s)
        for s, p in zip(states, probabilities)
    )
    breakdowns = tuple(make_breakdown(s) for s in states)
    expected = math.fsum(o.probability * b.outcome_utility for o, b in zip(outcomes, breakdowns))
    data = {
        "action_id": action_id, "decision": decision, "eligible": eligible,
        "eligibility_reasons": () if eligible else ("Prohibited",),
        "outcomes": outcomes, "outcome_utilities": breakdowns,
        "expected_utility": expected, "rank": rank,
    }
    return ActionExpectedUtility(**{**data, **overrides})


def make_context() -> DecisionContext:
    return DecisionContext(
        scenario_id="X", risk_need=0.5, asset_stake=0.5, evidence_support=0.5, required_intensity=0.5,
        confirmed_impact=False, failed_logins_present=False, source_not_known=False,
    )


def make_analysis(actions, selected_id, ranked_ids=None, **overrides) -> UtilityDecisionAnalysis:
    ranked_ids = ranked_ids if ranked_ids is not None else tuple(a.action_id for a in actions if a.eligible)
    data = {
        "profile": make_profile(), "context": make_context(), "actions": tuple(actions),
        "ranking": UtilityRanking(ranked_action_ids=tuple(ranked_ids), tie_breaker_used=False),
        "selected_action_id": selected_id, "priority": Priority.HIGH,
    }
    return UtilityDecisionAnalysis(**{**data, **overrides})


# ---------------------------------------------------------------------------
# UB-M01 .. UB-M15
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight", [-0.01, 1.01, 2.0])
def test_ub_m01_criterion_rejects_weights_outside_unit_range(weight):
    with pytest.raises(ValidationError):
        UtilityCriterion(criterion_id="UC-01", name=CriterionName.RISK_REDUCTION, description="d", weight=weight)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_ub_m01_criterion_accepts_boundary_weights(weight):
    assert UtilityCriterion(criterion_id="UC-01", name=CriterionName.RISK_REDUCTION, description="d", weight=weight)


def test_ub_m02_profile_rejects_weights_that_do_not_sum_to_one():
    heavy = {**WEIGHTS, CriterionName.RISK_REDUCTION: 0.30}
    with pytest.raises(ValidationError, match="sum to 1.0"):
        make_profile(heavy)
    light = {**WEIGHTS, CriterionName.RISK_REDUCTION: 0.20}
    with pytest.raises(ValidationError, match="sum to 1.0"):
        make_profile(light)


def test_ub_m03_profile_accepts_the_documented_six_criterion_profile():
    profile = make_profile()
    assert len(profile.criteria) == 6
    assert abs(profile.weight_total - 1.0) <= 1e-9


def test_profile_requires_every_criterion_exactly_once():
    five = {k: v for k, v in WEIGHTS.items() if k is not CriterionName.HUMAN_CONTROL}
    five[CriterionName.RISK_REDUCTION] = 0.35
    with pytest.raises(ValidationError, match="six criteria"):
        make_profile(five)
    with pytest.raises(ValidationError):
        make_profile(criteria=())


@pytest.mark.parametrize("field", ["risk_reduction", "asset_protection", "evidence_alignment",
                                   "proportionality", "human_control", "operational_continuity"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_ub_m04_outcome_scores_reject_values_outside_unit_range(field, value):
    with pytest.raises(ValidationError):
        make_state(**{field: value})


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_ub_m05_probability_rejects_values_outside_unit_range(probability):
    with pytest.raises(ValidationError):
        ProbabilisticOutcome(outcome_id="O", action_id="A", probability=probability, outcome_state=make_state())


def test_ub_m06_action_rejects_probabilities_that_do_not_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        make_eu(probabilities=(0.7, 0.2))
    with pytest.raises(ValidationError, match="sum to 1.0"):
        make_eu(probabilities=(0.7, 0.4))


def test_ub_m07_action_requires_at_least_two_outcomes():
    valid = make_eu()
    with pytest.raises(ValidationError):
        ActionExpectedUtility(
            **{**valid.model_dump(), "outcomes": valid.outcomes[:1], "outcome_utilities": valid.outcome_utilities[:1]}
        )


def test_ub_m08_outcome_utility_equals_sum_of_weighted_contributions():
    breakdown = make_breakdown(make_state(score=0.8))
    assert breakdown.outcome_utility == pytest.approx(sum(c.weighted_contribution for c in breakdown.contributions))
    assert breakdown.outcome_utility == pytest.approx(0.8)  # all scores 0.8 and weights sum to 1
    with pytest.raises(ValidationError, match="sum of weighted contributions"):
        OutcomeUtilityBreakdown(
            outcome_id="O", contributions=breakdown.contributions, outcome_utility=breakdown.outcome_utility - 0.1
        )


def test_contribution_must_equal_score_times_weight():
    with pytest.raises(ValidationError, match="raw score times weight"):
        CriterionContribution(
            criterion=CriterionName.RISK_REDUCTION, raw_score=0.5, weight=0.25, weighted_contribution=0.2
        )


def test_ub_m09_expected_utility_equals_probability_weighted_sum():
    action = make_eu(probabilities=(0.7, 0.3), scores=(0.9, 0.4))
    assert action.expected_utility == pytest.approx(0.7 * 0.9 + 0.3 * 0.4)
    with pytest.raises(ValidationError, match="probability-weighted"):
        make_eu(expected_utility=0.1)


def test_ub_m10_expected_utility_stays_between_zero_and_one():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            make_eu(expected_utility=bad)
    assert 0.0 <= make_eu(scores=(1.0, 1.0)).expected_utility <= 1.0
    assert 0.0 <= make_eu(scores=(0.0, 0.0)).expected_utility <= 1.0


def test_ub_m11_unexpected_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_state(risk_score=0.5)
    with pytest.raises(ValidationError):
        UtilityCriterion(criterion_id="UC-01", name=CriterionName.RISK_REDUCTION, description="d", weight=0.1, bonus=1)
    with pytest.raises(ValidationError):
        make_eu(learned_bias=0.2)


def test_ub_m12_models_are_immutable_and_copy_safe():
    profile = make_profile()
    with pytest.raises(ValidationError):
        profile.profile_id = "changed"
    with pytest.raises(ValidationError):
        make_eu().expected_utility = 0.0
    copy = profile.model_copy(deep=True)
    assert copy == profile and copy is not profile


@pytest.mark.parametrize("rank", [0, -1])
def test_ub_m13_rank_must_be_positive(rank):
    with pytest.raises(ValidationError):
        make_eu(rank=rank)
    assert make_eu(rank=1).rank == 1


def test_ineligible_action_must_explain_and_cannot_be_ranked():
    with pytest.raises(ValidationError, match="must state why"):
        make_eu(eligible=False, eligibility_reasons=())
    with pytest.raises(ValidationError, match="cannot be ranked"):
        make_eu(eligible=False, rank=1)


def test_ub_m14_selected_action_must_be_eligible():
    unsafe = make_eu("ACT-UNSAFE", scores=(1.0, 1.0), eligible=False)
    safe = make_eu("ACT-SAFE", rank=1)
    assert make_analysis([unsafe, safe], "ACT-SAFE", ranked_ids=("ACT-SAFE",))
    with pytest.raises(ValidationError, match="must be eligible"):
        make_analysis([unsafe, safe], "ACT-UNSAFE", ranked_ids=("ACT-SAFE",))


def test_ub_m15_selected_action_must_have_highest_eligible_expected_utility():
    best = make_eu("ACT-BEST", scores=(0.9, 0.5), rank=1)
    worse = make_eu("ACT-WORSE", scores=(0.5, 0.2), rank=2)
    assert make_analysis([best, worse], "ACT-BEST", ranked_ids=("ACT-BEST", "ACT-WORSE"))
    # Ranking the lower-utility action first is inconsistent and must be refused.
    swapped_best = make_eu("ACT-BEST", scores=(0.9, 0.5), rank=2)
    swapped_worse = make_eu("ACT-WORSE", scores=(0.5, 0.2), rank=1)
    with pytest.raises(ValidationError, match="highest eligible expected utility"):
        make_analysis([swapped_best, swapped_worse], "ACT-WORSE", ranked_ids=("ACT-WORSE", "ACT-BEST"))


def test_analysis_requires_consistent_ranks_and_ranking():
    a = make_eu("ACT-A", scores=(0.9, 0.5), rank=1)
    b = make_eu("ACT-B", scores=(0.5, 0.2), rank=3)
    with pytest.raises(ValidationError, match="consecutive"):
        make_analysis([a, b], "ACT-A", ranked_ids=("ACT-A", "ACT-B"))
    b2 = make_eu("ACT-B", scores=(0.5, 0.2), rank=2)
    with pytest.raises(ValidationError, match="rank order"):
        make_analysis([a, b2], "ACT-A", ranked_ids=("ACT-B", "ACT-A"))
    unranked = make_eu("ACT-B", scores=(0.5, 0.2), rank=None)
    with pytest.raises(ValidationError, match="eligible actions must be ranked"):
        make_analysis([a, unranked], "ACT-A", ranked_ids=("ACT-A",))


def test_analysis_breakdown_weights_must_match_profile():
    a = make_eu("ACT-A", rank=1)
    other = make_profile({**WEIGHTS, CriterionName.RISK_REDUCTION: 0.15, CriterionName.HUMAN_CONTROL: 0.20})
    with pytest.raises(ValidationError, match="match the utility profile"):
        make_analysis([a], "ACT-A", profile=other)


def test_utility_action_requires_unit_range_characteristics_and_steps():
    base = {
        "action_id": "A", "decision": "MONITOR", "description": "d", "risk_handling": 0.1,
        "asset_protection": 0.1, "evidence_development": 0.1, "intensity": 0.1, "human_control": 0.1,
        "plan_steps": (UtilityStep(text="Do something safe."),),
    }
    assert UtilityAction(**base)
    with pytest.raises(ValidationError):
        UtilityAction(**{**base, "intensity": 1.5})
    with pytest.raises(ValidationError):
        UtilityAction(**{**base, "plan_steps": ()})
    with pytest.raises(ValidationError):
        UtilityAction(**{**base, "decision": "BLOCK_ACCOUNT"})


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

ALL_IDS = ["A", "B", "C"]
EXPECTED = {
    "A": (Priority.HIGH, "INVESTIGATE"),
    "B": (Priority.LOW, "COLLECT_MORE_EVIDENCE"),
    "C": (Priority.CRITICAL, "ESCALATE_TO_SECURITY_ANALYST"),
}
FOUR_DECISIONS = ["MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"]


def run(scenario_id: str) -> AgentDecision:
    return UtilityBasedAgent().decide(get_scenario(scenario_id))


def variant(base_id: str = "A", **changes) -> SecurityScenario:
    return SecurityScenario.model_validate({**get_scenario(base_id).model_dump(mode="json"), **changes})


def analyses(result: AgentDecision) -> dict[str, dict]:
    return {a["decision"]: a for a in result.metadata["action_analyses"]}


def ranks(result: AgentDecision) -> dict[str, int]:
    return {d: a["rank"] for d, a in analyses(result).items()}


def recompute_expected_utility(action_meta: dict) -> float:
    """Recomputes EU from probabilities, raw criterion scores and weights only."""
    total = 0.0
    for outcome in action_meta["outcomes"]:
        utility = sum(part["raw_score"] * part["weight"] for part in outcome["criterion_breakdown"])
        total += outcome["probability"] * utility
    return total


# --- Core agent ---------------------------------------------------------------


def test_ub01_inherits_directly_from_base_agent():
    assert UtilityBasedAgent.__bases__ == (BaseAgent,)
    assert isinstance(UtilityBasedAgent(), BaseAgent)


def test_ub02_is_not_a_subclass_of_any_other_classical_agent():
    others = (SimpleReflexAgent, ModelBasedReflexAgent, GoalBasedAgent)
    assert not issubclass(UtilityBasedAgent, others)
    assert not any(issubclass(other, UtilityBasedAgent) for other in others)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub03_to_ub05_expected_priority_and_decision(scenario_id):
    result = run(scenario_id)
    assert (result.priority, result.decision) == EXPECTED[scenario_id]


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub06_ub07_result_validates_and_has_all_explanation_fields(scenario_id):
    result = run(scenario_id)
    assert AgentDecision.model_validate(result.model_dump()) == result
    assert result.agent_type.value == "UTILITY_BASED" and result.agent_name == "Utility-Based Agent"
    assert result.reasoning and result.evidence and result.uncertainty
    assert result.likely_interpretation and result.recommended_actions and result.audit_note
    assert result.tool_calls == [] and result.experience_used is None


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub08_ub09_utility_score_is_populated_and_matches_selected_expected_utility(scenario_id):
    result = run(scenario_id)
    assert isinstance(result.utility_score, float) and 0.0 < result.utility_score <= 1.0
    assert result.utility_score == result.metadata["selected_expected_utility"]


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub10_all_four_candidates_are_evaluated(scenario_id):
    result = run(scenario_id)
    assert result.metadata["candidate_actions_considered"] == FOUR_DECISIONS
    assert set(analyses(result)) == set(FOUR_DECISIONS)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub11_ub12_each_candidate_has_a_proper_probability_distribution(scenario_id):
    for action in run(scenario_id).metadata["action_analyses"]:
        assert len(action["outcomes"]) >= 2
        assert {o["name"] for o in action["outcomes"]} == {"FAVORABLE", "PARTIAL", "UNFAVORABLE"}
        assert all(0.0 <= o["probability"] <= 1.0 for o in action["outcomes"])
        assert abs(sum(o["probability"] for o in action["outcomes"]) - 1.0) <= TOLERANCE


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub13_each_outcome_has_a_complete_utility_breakdown(scenario_id):
    for action in run(scenario_id).metadata["action_analyses"]:
        for outcome in action["outcomes"]:
            breakdown = outcome["criterion_breakdown"]
            assert {p["criterion"] for p in breakdown} == {c.value for c in CriterionName}
            assert all(0.0 <= p["raw_score"] <= 1.0 for p in breakdown)
            assert all(abs(p["weighted_contribution"] - p["raw_score"] * p["weight"]) <= TOLERANCE for p in breakdown)
            assert abs(sum(p["weighted_contribution"] for p in breakdown) - outcome["outcome_utility"]) <= TOLERANCE


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub14_expected_utility_is_independently_reproducible(scenario_id):
    for action in run(scenario_id).metadata["action_analyses"]:
        assert recompute_expected_utility(action) == pytest.approx(action["expected_utility"], abs=1e-9)
        assert 0.0 <= action["expected_utility"] <= 1.0
        contributions = sum(o["contribution_to_expected_utility"] for o in action["outcomes"])
        assert contributions == pytest.approx(action["expected_utility"], abs=1e-9)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub15_to_ub17_every_eligible_action_is_ranked_and_the_best_is_selected(scenario_id):
    result = run(scenario_id)
    actions = result.metadata["action_analyses"]
    eligible = [a for a in actions if a["eligible"]]
    assert all(a["rank"] is not None for a in eligible)
    assert sorted(a["rank"] for a in eligible) == list(range(1, len(eligible) + 1))
    selected = next(a for a in actions if a["rank"] == 1)
    assert selected["decision"] == result.decision == result.metadata["selected_action"]
    assert selected["expected_utility"] == max(a["expected_utility"] for a in eligible)
    by_rank = sorted(eligible, key=lambda a: a["rank"])
    assert [a["expected_utility"] for a in by_rank] == sorted((a["expected_utility"] for a in eligible), reverse=True)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub18_to_ub20_metadata_declares_mechanism_formula_and_boundaries(scenario_id):
    metadata = run(scenario_id).metadata
    assert metadata["decision_mechanism"] == "expected_utility_maximization"
    assert metadata["expected_utility_formula"] == "EU(a)=sum(P(outcome|a,scenario)*U(outcome))"
    assert metadata["uses_learning"] is False and metadata["uses_llm"] is False
    assert metadata["uses_goal_satisfaction_selection"] is False
    profile = metadata["utility_profile"]
    assert profile["profile_id"] == "UP-01" and abs(profile["weight_total"] - 1.0) <= TOLERANCE
    assert [c["weight"] for c in profile["criteria"]] == [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]


# --- Scenario A ---------------------------------------------------------------


def test_ub_a01_to_a03_scenario_a_ranking():
    result = run("A")
    rank = ranks(result)
    assert rank["INVESTIGATE"] == 1
    assert rank["MONITOR"] > rank["INVESTIGATE"]
    assert rank["ESCALATE_TO_SECURITY_ANALYST"] > rank["INVESTIGATE"]  # more intensive than necessary
    assert rank["MONITOR"] == 4  # inadequate for repeated privileged-account activity


def test_ub_a04_scenario_a_does_not_claim_confirmed_compromise():
    result = run("A")
    assert "NOT confirmed" in result.uncertainty and "Confirmed impact: False" in result.evidence
    assert "Compromise remains unconfirmed" in result.reasoning


def test_scenario_a_plan_reviews_authentication_and_validates_source():
    steps = " ".join(run("A").recommended_actions).lower()
    assert "authentication" in steps and "source is expected" in steps and "escalate if stronger evidence" in steps


# --- Scenario B ---------------------------------------------------------------


def test_ub_b01_b02_scenario_b_ranking():
    rank = ranks(run("B"))
    assert rank["COLLECT_MORE_EVIDENCE"] == 1
    assert rank["ESCALATE_TO_SECURITY_ANALYST"] > rank["COLLECT_MORE_EVIDENCE"]
    assert rank["ESCALATE_TO_SECURITY_ANALYST"] == 4  # disproportionate for weak evidence


def test_ub_b03_scenario_b_is_driven_by_evidence_alignment_and_proportionality():
    result = run("B")
    trade_off = result.metadata["key_trade_off"]
    assert trade_off["advantages"][0]["criterion"] == "EVIDENCE_ALIGNMENT"
    assert "EVIDENCE_ALIGNMENT" in result.reasoning and "PROPORTIONALITY" in result.reasoning

    def favorable_score(decision, criterion):
        outcome = next(o for o in analyses(result)[decision]["outcomes"] if o["name"] == "FAVORABLE")
        return next(p["raw_score"] for p in outcome["criterion_breakdown"] if p["criterion"] == criterion)

    for criterion in ("PROPORTIONALITY", "EVIDENCE_ALIGNMENT", "OPERATIONAL_CONTINUITY"):
        assert favorable_score("COLLECT_MORE_EVIDENCE", criterion) > favorable_score(
            "ESCALATE_TO_SECURITY_ANALYST", criterion
        )


# --- Scenario C ---------------------------------------------------------------


def test_ub_c01_c02_scenario_c_ranking():
    rank = ranks(run("C"))
    assert rank["ESCALATE_TO_SECURITY_ANALYST"] == 1
    assert rank["MONITOR"] > rank["ESCALATE_TO_SECURITY_ANALYST"]
    assert rank["MONITOR"] == 4


def test_ub_c03_scenario_c_reflects_strong_asset_protection_value():
    result = run("C")
    advantages = [a["criterion"] for a in result.metadata["key_trade_off"]["advantages"]]
    assert advantages[0] == "ASSET_PROTECTION" and "RISK_REDUCTION" in advantages
    weighted = {
        d: sum(
            o["probability"] * next(p["weighted_contribution"] for p in o["criterion_breakdown"]
                                    if p["criterion"] == "ASSET_PROTECTION")
            for o in a["outcomes"]
        )
        for d, a in analyses(result).items()
    }
    assert max(weighted, key=weighted.get) == "ESCALATE_TO_SECURITY_ANALYST"
    # It still pays an operational cost, which is outweighed.
    assert "OPERATIONAL_CONTINUITY" in [c["criterion"] for c in result.metadata["key_trade_off"]["concessions"]]


def test_ub_c04_c05_scenario_c_is_unconfirmed_and_requires_human_review():
    result = run("C")
    assert "NOT confirmed" in result.uncertainty and "Compromise remains unconfirmed" in result.reasoning
    assert "human investigation and review" in result.reasoning
    assert any("authorized SOC or security-analyst" in step for step in result.recommended_actions)
    assert result.metadata["human_authorization_required_for_consequential_action"] is True
    assert result.metadata["executed_actions"] == []


# --- Calculation --------------------------------------------------------------


def test_ub_calc01_default_weights_sum_to_one_and_match_the_documented_profile():
    assert abs(DEFAULT_UTILITY_PROFILE.weight_total - 1.0) <= TOLERANCE
    assert {c.name.value: c.weight for c in DEFAULT_UTILITY_PROFILE.criteria} == {
        "RISK_REDUCTION": 0.25, "ASSET_PROTECTION": 0.20, "EVIDENCE_ALIGNMENT": 0.20,
        "PROPORTIONALITY": 0.15, "HUMAN_CONTROL": 0.10, "OPERATIONAL_CONTINUITY": 0.10,
    }


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub_calc02_calc03_calc04_sums_match_everywhere(scenario_id):
    result = run(scenario_id)
    for action in result.metadata["action_analyses"]:
        for outcome in action["outcomes"]:
            weighted = math.fsum(p["weighted_contribution"] for p in outcome["criterion_breakdown"])
            assert weighted == pytest.approx(outcome["outcome_utility"], abs=1e-9)
        assert recompute_expected_utility(action) == pytest.approx(action["expected_utility"], abs=1e-9)
    top = next(a for a in result.metadata["action_analyses"] if a["rank"] == 1)
    assert result.utility_score == top["expected_utility"]


def test_ub_calc05_changing_a_relevant_fact_changes_the_calculation():
    base = run("A")
    changed = UtilityBasedAgent().decide(variant("A", asset_criticality="CRITICAL", asset_id="ASSET-PAY-01"))
    before, after = analyses(base), analyses(changed)
    assert any(
        before[d]["expected_utility"] != after[d]["expected_utility"]
        or [o["probability"] for o in before[d]["outcomes"]] != [o["probability"] for o in after[d]["outcomes"]]
        for d in FOUR_DECISIONS
    )
    for fact in ({"severity": "LOW"}, {"evidence_strength": "LIMITED"}, {"confirmed_impact": True},
                 {"failed_login_count": 1}, {"unusual_access": False}):
        other = UtilityBasedAgent().decide(variant("A", **fact))
        assert other.metadata["action_analyses"] != base.metadata["action_analyses"], fact


def test_ub_calc06_calc07_scenario_ids_do_not_control_decisions():
    agent = UtilityBasedAgent()
    for original in ALL_IDS:
        reference = agent.decide(get_scenario(original))
        for new_id in ("ZZZ", "A", "B", "C", "scenario-42"):
            renamed = agent.decide(variant(original, scenario_id=new_id))
            assert renamed.decision == reference.decision and renamed.priority == reference.priority
            assert renamed.metadata["action_analyses"] == reference.metadata["action_analyses"]
            assert renamed.utility_score == reference.utility_score
            assert ranks(renamed) == ranks(reference)


def test_ub_calc08_calc09_deterministic_for_repeats_and_separate_instances():
    agent = UtilityBasedAgent()
    for scenario in get_all_scenarios():
        first, second = agent.decide(scenario), agent.decide(scenario)
        assert first == second == UtilityBasedAgent().decide(scenario)


def test_alternative_profile_changes_the_decision_proving_utility_drives_selection():
    weights = {
        CriterionName.RISK_REDUCTION: 0.05, CriterionName.ASSET_PROTECTION: 0.05,
        CriterionName.EVIDENCE_ALIGNMENT: 0.05, CriterionName.PROPORTIONALITY: 0.05,
        CriterionName.HUMAN_CONTROL: 0.05, CriterionName.OPERATIONAL_CONTINUITY: 0.75,
    }
    cautious_about_disruption = make_profile(weights, profile_id="UP-TEST-CONTINUITY")
    result = UtilityBasedAgent(cautious_about_disruption).decide(get_scenario("C"))
    assert result.decision != "ESCALATE_TO_SECURITY_ANALYST"
    assert result.metadata["utility_profile"]["profile_id"] == "UP-TEST-CONTINUITY"
    assert UtilityBasedAgent().decide(get_scenario("C")).decision == "ESCALATE_TO_SECURITY_ANALYST"


# --- Ranking and tie-breaking ----------------------------------------------------


def custom_eu(action_id: str, **score_overrides) -> ActionExpectedUtility:
    """Two outcomes with identical scores (0.5 unless overridden), probability 0.5 each."""
    states = [
        make_state(name, 0.5, outcome_id=f"{action_id}-{name.value}", **score_overrides)
        for name in (OutcomeCategory.FAVORABLE, OutcomeCategory.UNFAVORABLE)
    ]
    outcomes = tuple(
        ProbabilisticOutcome(outcome_id=s.outcome_id, action_id=action_id, probability=0.5, outcome_state=s)
        for s in states
    )
    breakdowns = tuple(make_breakdown(s) for s in states)
    expected = math.fsum(o.probability * b.outcome_utility for o, b in zip(outcomes, breakdowns))
    return ActionExpectedUtility(
        action_id=action_id, decision="INVESTIGATE", eligible=True, eligibility_reasons=("ok",),
        outcomes=outcomes, outcome_utilities=breakdowns, expected_utility=expected,
    )


def test_ub_calc10_tie_break_prefers_less_disruptive_action_first():
    plain = custom_eu("ACT-PLAIN")
    # Same expected utility (0.5), but higher continuity paid for by lower risk reduction.
    gentle = custom_eu("ACT-GENTLE", operational_continuity=0.8, risk_reduction=0.38)
    assert abs(plain.expected_utility - gentle.expected_utility) <= TOLERANCE
    for order in ((plain, gentle), (gentle, plain)):  # input order must not matter
        ranked, ranking = UtilityBasedAgent._rank_actions(order)
        assert ranking.ranked_action_ids[0] == "ACT-GENTLE"
        assert ranking.tie_breaker_used is True
        assert ranking.tie_groups == (tuple(ranking.ranked_action_ids),) or set(ranking.tie_groups[0]) == {
            "ACT-PLAIN", "ACT-GENTLE"
        }
        assert {a.action_id: a.rank for a in ranked} == {"ACT-GENTLE": 1, "ACT-PLAIN": 2}


def test_ub_calc10_tie_break_then_prefers_evidence_alignment_then_human_control():
    plain = custom_eu("ACT-PLAIN")
    evidence = custom_eu("ACT-EVIDENCE", evidence_alignment=0.7, risk_reduction=0.34)
    control = custom_eu("ACT-CONTROL", human_control=0.7, risk_reduction=0.42)
    assert abs(plain.expected_utility - evidence.expected_utility) <= TOLERANCE
    assert abs(plain.expected_utility - control.expected_utility) <= TOLERANCE
    _, ranking = UtilityBasedAgent._rank_actions((plain, control, evidence))
    assert ranking.ranked_action_ids == ("ACT-EVIDENCE", "ACT-CONTROL", "ACT-PLAIN")


def test_ub_calc10_final_tie_breaker_is_stable_candidate_order():
    first, second = custom_eu("ACT-FIRST"), custom_eu("ACT-SECOND")
    assert UtilityBasedAgent._rank_actions((first, second))[1].ranked_action_ids == ("ACT-FIRST", "ACT-SECOND")
    assert UtilityBasedAgent._rank_actions((second, first))[1].ranked_action_ids == ("ACT-SECOND", "ACT-FIRST")


def test_clear_utility_difference_is_not_a_tie():
    better = custom_eu("ACT-BETTER", risk_reduction=0.9)
    worse = custom_eu("ACT-WORSE")
    _, ranking = UtilityBasedAgent._rank_actions((worse, better))
    assert ranking.ranked_action_ids == ("ACT-BETTER", "ACT-WORSE") and ranking.tie_breaker_used is False
    assert ranking.tie_groups == ()


def test_tie_breaker_order_is_documented_in_every_result():
    order = run("A").metadata["tie_breaker_order"]
    assert order[0].startswith("higher expected OPERATIONAL_CONTINUITY") and order[-1] == "stable candidate order"
    assert run("A").metadata["tie_breaker_used"] is False


# --- Architectural boundaries ---------------------------------------------------


def test_ub_d01_does_not_call_any_other_agents_decide(monkeypatch):
    def forbidden(self, scenario):
        raise AssertionError("another agent's decide() was called")

    for other in (SimpleReflexAgent, ModelBasedReflexAgent, GoalBasedAgent):
        monkeypatch.setattr(other, "decide", forbidden)
    for scenario in get_all_scenarios():
        UtilityBasedAgent().decide(scenario)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub_d02_goal_based_output_remains_utility_free(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert result.utility_score is None
    assert result.metadata["uses_utility_scoring"] is False
    assert "action_analyses" not in result.metadata and "expected_utility_formula" not in result.metadata


def test_ub_d03_d04_no_experience_and_no_mutable_history():
    agent = UtilityBasedAgent()
    before = dict(vars(agent))
    for scenario in get_all_scenarios() * 2:
        assert agent.decide(scenario).experience_used is None
    assert dict(vars(agent)) == before == {"_profile": DEFAULT_UTILITY_PROFILE}


def test_ub_d05_d06_no_file_access_and_experience_json_unchanged(monkeypatch):
    scenarios = get_all_scenarios()  # load before file access is blocked
    experience = DATA_DIR / "experience.json"
    before = (experience.read_bytes(), experience.stat().st_mtime_ns)

    def blocked(*args, **kwargs):
        raise AssertionError("agent attempted file access")

    monkeypatch.setattr(builtins, "open", blocked)
    for name in ("read_text", "read_bytes", "write_text", "write_bytes", "open"):
        monkeypatch.setattr(Path, name, blocked)
    for scenario in scenarios:
        UtilityBasedAgent().decide(scenario)
    monkeypatch.undo()
    assert (experience.read_bytes(), experience.stat().st_mtime_ns) == before


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_ub_d07_scenario_is_not_mutated(scenario_id):
    scenario = get_scenario(scenario_id)
    snapshot = scenario.model_copy(deep=True)
    UtilityBasedAgent().decide(scenario)
    assert scenario == snapshot


def test_ub_d08_no_randomness_is_used(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("randomness used")

    for name in ("random", "choice", "uniform", "shuffle", "sample", "randint", "gauss"):
        monkeypatch.setattr(random, name, forbidden)
    import agents.utility_based as module

    assert not any(hasattr(module, name) for name in ("random", "secrets", "time", "datetime"))
    for scenario in get_all_scenarios():
        UtilityBasedAgent().decide(scenario)


def test_ub_d09_no_llm_or_ui_dependency_is_introduced():
    # LangChain is now a legitimate dependency of the separate LLM layer and may already be loaded by other
    # tests, so check what THIS agent imports and loads rather than the global module table.
    before = set(sys.modules)
    for scenario in get_all_scenarios():
        UtilityBasedAgent().decide(scenario)
    forbidden = {"langchain", "langchain_core", "langchain_ollama", "ollama", "streamlit"}
    assert not forbidden & (set(sys.modules) - before)
    tree = ast.parse((Path(__file__).resolve().parents[1] / "agents" / "utility_based.py").read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not forbidden & imported
    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8").lower()
    # Streamlit is now the declared UI dependency (separate from the agents); the numeric libraries stay excluded.
    assert not any(name in requirements for name in ("numpy", "pandas", "scipy"))


def test_agent_module_does_not_depend_on_data_services_or_other_agents():
    import agents.utility_based as module

    assert not any(name.startswith("services") for name in vars(module))
    for other in ("SimpleReflexAgent", "ModelBasedReflexAgent", "GoalBasedAgent"):
        assert not hasattr(module, other)


# --- Eligibility and safety override --------------------------------------------


def unsafe_dominant_action(side_effect=ProhibitedCondition.AUTOMATIC_USER_BLOCKING) -> UtilityAction:
    """Looks superb on every capability, but would perform a prohibited action."""
    return UtilityAction(
        action_id="UB-ACT-UNSAFE", decision="ESCALATE_TO_SECURITY_ANALYST", description="Unsafe test candidate.",
        risk_handling=1.0, asset_protection=1.0, evidence_development=1.0, intensity=0.85, human_control=1.0,
        plan_steps=(UtilityStep(text="Placeholder step."),), side_effects=(side_effect,),
    )


@pytest.mark.parametrize("prohibited", list(ProhibitedCondition)[:-1])
def test_higher_utility_cannot_override_a_safety_prohibition(monkeypatch, prohibited):
    import agents.utility_based as module

    monkeypatch.setattr(module, "CANDIDATE_ACTIONS", (*CANDIDATE_ACTIONS, unsafe_dominant_action(prohibited)))
    result = UtilityBasedAgent().decide(get_scenario("C"))
    by_id = {a["action_id"]: a for a in result.metadata["action_analyses"]}
    unsafe = by_id["UB-ACT-UNSAFE"]
    best_safe = max(a["expected_utility"] for a in by_id.values() if a["eligible"])
    assert unsafe["expected_utility"] > best_safe  # genuinely higher utility ...
    assert unsafe["eligible"] is False and unsafe["rank"] is None  # ... yet ineligible and unranked
    assert any(prohibited.value in reason for reason in unsafe["eligibility_reasons"])
    assert result.metadata["selected_action_id"] == "UB-ACT-ESCALATE"
    assert result.metadata["ineligible_actions"][0]["decision"] == "ESCALATE_TO_SECURITY_ANALYST"


def test_consequential_action_without_human_authorization_is_ineligible(monkeypatch):
    import agents.utility_based as module

    risky = unsafe_dominant_action().model_copy(
        update={"side_effects": (), "is_consequential": True, "requires_human_authorization": False}
    )
    monkeypatch.setattr(module, "CANDIDATE_ACTIONS", (*CANDIDATE_ACTIONS, risky))
    result = UtilityBasedAgent().decide(get_scenario("C"))
    unsafe = next(a for a in result.metadata["action_analyses"] if a["action_id"] == "UB-ACT-UNSAFE")
    assert not unsafe["eligible"] and result.metadata["selected_action_id"] != "UB-ACT-UNSAFE"


def test_authorized_consequential_action_is_eligible():
    safe = unsafe_dominant_action().model_copy(
        update={"side_effects": (), "is_consequential": True, "requires_human_authorization": True}
    )
    context = UtilityBasedAgent._build_decision_context(get_scenario("C"))
    assert UtilityBasedAgent._check_action_eligibility(context, safe)[0] is True


def test_unsupported_claim_is_ineligible_unless_the_scenario_confirms_impact():
    claim = unsafe_dominant_action(ProhibitedCondition.UNSUPPORTED_COMPROMISE_CLAIM)
    unconfirmed = UtilityBasedAgent._build_decision_context(get_scenario("A"))
    confirmed = UtilityBasedAgent._build_decision_context(variant("A", confirmed_impact=True))
    assert UtilityBasedAgent._check_action_eligibility(unconfirmed, claim)[0] is False
    assert UtilityBasedAgent._check_action_eligibility(confirmed, claim)[0] is True


def test_when_no_action_is_eligible_the_agent_refuses_instead_of_guessing(monkeypatch):
    import agents.utility_based as module

    monkeypatch.setattr(module, "CANDIDATE_ACTIONS", (unsafe_dominant_action(),))
    with pytest.raises(NoEligibleActionError):
        UtilityBasedAgent().decide(get_scenario("A"))


def test_ranking_never_includes_an_ineligible_action_even_with_the_top_utility():
    unsafe = custom_eu("ACT-UNSAFE", risk_reduction=1.0, asset_protection=1.0).model_copy(
        update={"eligible": False, "eligibility_reasons": ("Prohibited",)}
    )
    safe = custom_eu("ACT-SAFE")
    ranked, ranking = UtilityBasedAgent._rank_actions((unsafe, safe))
    assert ranking.ranked_action_ids == ("ACT-SAFE",)
    assert {a.action_id: a.rank for a in ranked} == {"ACT-UNSAFE": None, "ACT-SAFE": 1}
    assert UtilityBasedAgent._select_best_action(ranked, ranking).action_id == "ACT-SAFE"


def test_catalogue_actions_are_all_eligible_and_every_plan_has_two_to_five_steps():
    for scenario in get_all_scenarios():
        context = UtilityBasedAgent._build_decision_context(scenario)
        for action in CANDIDATE_ACTIONS:
            assert UtilityBasedAgent._check_action_eligibility(context, action)[0] is True
    for scenario in get_all_scenarios():
        assert 2 <= len(run(scenario.scenario_id).recommended_actions) <= 5


def test_probability_model_is_a_pure_function_of_scenario_facts():
    estimate = UtilityBasedAgent._estimate_outcome_probabilities
    context = UtilityBasedAgent._build_decision_context(get_scenario("C"))
    twin = UtilityBasedAgent._build_decision_context(variant("C", scenario_id="OTHER"))
    for action in CANDIDATE_ACTIONS:
        assert estimate(context, action) == estimate(twin, action)
        assert abs(sum(estimate(context, action).values()) - 1.0) <= TOLERANCE
