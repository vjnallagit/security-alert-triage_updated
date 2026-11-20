"""Typed models for the Utility-Based Agent.

The agent compares actions by expected utility:

    U(outcome) = sum over criteria of  weight_c * score_c(outcome)
    EU(action) = sum over outcomes of  P(outcome | action, scenario) * U(outcome)

All numbers are transparent simulation assumptions for an academic exercise, not
statistical estimates. Floats are used with the documented tolerances below.
"""

import math
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, model_validator

from models.goal_models import DecisionName, FrozenModel, ProhibitedCondition
from models.schemas import NonEmptyStr, Priority

# Absolute tolerance for "sums to 1.0", "equals the weighted sum" and "tied" comparisons.
TOLERANCE = 1e-9

UnitScore = Annotated[float, Field(ge=0.0, le=1.0)]


class CriterionName(StrEnum):
    RISK_REDUCTION = "RISK_REDUCTION"
    ASSET_PROTECTION = "ASSET_PROTECTION"
    EVIDENCE_ALIGNMENT = "EVIDENCE_ALIGNMENT"
    PROPORTIONALITY = "PROPORTIONALITY"
    HUMAN_CONTROL = "HUMAN_CONTROL"
    OPERATIONAL_CONTINUITY = "OPERATIONAL_CONTINUITY"


class OutcomeCategory(StrEnum):
    FAVORABLE = "FAVORABLE"
    PARTIAL = "PARTIAL"
    UNFAVORABLE = "UNFAVORABLE"


# --------------------------------------------------------------------------
# Utility profile (criteria and weights)
# --------------------------------------------------------------------------


class UtilityCriterion(FrozenModel):
    criterion_id: str = NonEmptyStr
    name: CriterionName
    description: str = NonEmptyStr
    weight: UnitScore
    direction: str = "HIGHER_IS_BETTER"  # 0.0 = worst outcome, 1.0 = best outcome


class UtilityProfile(FrozenModel):
    profile_id: str = NonEmptyStr
    name: str = NonEmptyStr
    description: str = NonEmptyStr
    criteria: tuple[UtilityCriterion, ...] = Field(min_length=1)

    @property
    def weight_total(self) -> float:
        return math.fsum(c.weight for c in self.criteria)

    def weight_of(self, name: CriterionName) -> float:
        return next(c.weight for c in self.criteria if c.name == name)

    @model_validator(mode="after")
    def _weights_must_form_a_complete_distribution(self) -> Self:
        names = [c.name for c in self.criteria]
        if set(names) != set(CriterionName) or len(names) != len(CriterionName):
            raise ValueError("profile must contain each of the six criteria exactly once")
        if len({c.criterion_id for c in self.criteria}) != len(self.criteria):
            raise ValueError("criterion IDs must be unique")
        if abs(self.weight_total - 1.0) > TOLERANCE:
            raise ValueError(f"criterion weights must sum to 1.0, got {self.weight_total}")
        return self


# --------------------------------------------------------------------------
# Decision context and candidate actions
# --------------------------------------------------------------------------


class DecisionContext(FrozenModel):
    """Concise numeric reading of one scenario. Scenario IDs never enter calculations."""

    scenario_id: str = NonEmptyStr  # label for output only
    risk_need: UnitScore  # how much risk reduction the situation calls for
    asset_stake: UnitScore  # how much is at stake on the affected asset
    evidence_support: UnitScore  # how strongly the evidence supports acting
    required_intensity: UnitScore  # response intensity the situation calls for
    confirmed_impact: bool
    failed_logins_present: bool
    source_not_known: bool
    privileged_account_pattern: bool = False
    critical_service_pattern: bool = False
    adjustments_applied: tuple[str, ...] = ()  # human-readable audit of each adjustment


class UtilityStep(FrozenModel):
    text: str = NonEmptyStr
    only_if: str | None = None  # name of a DecisionContext boolean; step shown only if true


class UtilityAction(FrozenModel):
    """A recommendation to evaluate. It is never executed."""

    action_id: str = NonEmptyStr
    decision: DecisionName
    description: str = NonEmptyStr
    # Documented capability assumptions, each 0.0 (none) to 1.0 (maximum).
    risk_handling: UnitScore
    asset_protection: UnitScore
    evidence_development: UnitScore
    intensity: UnitScore  # disruption and effort; also drives operational continuity
    human_control: UnitScore
    plan_steps: tuple[UtilityStep, ...] = Field(min_length=1)
    is_consequential: bool = False
    requires_human_authorization: bool = False
    side_effects: tuple[ProhibitedCondition, ...] = ()


# --------------------------------------------------------------------------
# Outcomes and utility calculation
# --------------------------------------------------------------------------


class OutcomeState(FrozenModel):
    """A plausible consequence of recommending an action, scored per criterion."""

    outcome_id: str = NonEmptyStr
    name: OutcomeCategory
    description: str = NonEmptyStr
    risk_reduction: UnitScore
    asset_protection: UnitScore
    evidence_alignment: UnitScore
    proportionality: UnitScore
    human_control: UnitScore
    operational_continuity: UnitScore
    limitations: tuple[str, ...] = ()

    def score(self, criterion: CriterionName) -> float:
        return getattr(self, criterion.value.lower())


class ProbabilisticOutcome(FrozenModel):
    outcome_id: str = NonEmptyStr
    action_id: str = NonEmptyStr
    probability: UnitScore
    outcome_state: OutcomeState


class CriterionContribution(FrozenModel):
    criterion: CriterionName
    raw_score: UnitScore
    weight: UnitScore
    weighted_contribution: UnitScore

    @model_validator(mode="after")
    def _contribution_is_score_times_weight(self) -> Self:
        if abs(self.weighted_contribution - self.raw_score * self.weight) > TOLERANCE:
            raise ValueError("weighted contribution must equal raw score times weight")
        return self


class OutcomeUtilityBreakdown(FrozenModel):
    outcome_id: str = NonEmptyStr
    contributions: tuple[CriterionContribution, ...] = Field(min_length=1)
    outcome_utility: UnitScore

    @model_validator(mode="after")
    def _utility_is_sum_of_contributions(self) -> Self:
        total = math.fsum(c.weighted_contribution for c in self.contributions)
        if abs(total - self.outcome_utility) > TOLERANCE:
            raise ValueError("outcome utility must equal the sum of weighted contributions")
        if len({c.criterion for c in self.contributions}) != len(self.contributions):
            raise ValueError("each criterion may appear only once in a breakdown")
        return self


class ActionExpectedUtility(FrozenModel):
    action_id: str = NonEmptyStr
    decision: DecisionName
    eligible: bool
    eligibility_reasons: tuple[str, ...] = ()
    outcomes: tuple[ProbabilisticOutcome, ...] = Field(min_length=2)
    outcome_utilities: tuple[OutcomeUtilityBreakdown, ...]
    expected_utility: UnitScore
    rank: Annotated[int, Field(ge=1)] | None = None

    def expected_score(self, criterion: CriterionName) -> float:
        """Probability-weighted raw score for one criterion (used for tie-breaking)."""
        return math.fsum(o.probability * o.outcome_state.score(criterion) for o in self.outcomes)

    @model_validator(mode="after")
    def _distribution_and_expected_utility_must_be_consistent(self) -> Self:
        if abs(math.fsum(o.probability for o in self.outcomes) - 1.0) > TOLERANCE:
            raise ValueError("outcome probabilities must sum to 1.0")
        if any(o.action_id != self.action_id for o in self.outcomes):
            raise ValueError("every outcome must belong to this action")
        if len({o.outcome_state.name for o in self.outcomes}) != len(self.outcomes):
            raise ValueError("outcome categories must be unique within an action")
        if [u.outcome_id for u in self.outcome_utilities] != [o.outcome_id for o in self.outcomes]:
            raise ValueError("each outcome needs exactly one utility breakdown, in order")
        expected = math.fsum(
            o.probability * u.outcome_utility
            for o, u in zip(self.outcomes, self.outcome_utilities, strict=True)
        )
        if abs(expected - self.expected_utility) > TOLERANCE:
            raise ValueError("expected utility must equal the probability-weighted sum of outcome utilities")
        if not self.eligible and not self.eligibility_reasons:
            raise ValueError("an ineligible action must state why")
        if not self.eligible and self.rank is not None:
            raise ValueError("an ineligible action cannot be ranked")
        return self


class UtilityRanking(FrozenModel):
    """Eligible actions, best first, and how any ties were resolved."""

    ranked_action_ids: tuple[str, ...]
    tie_breaker_used: bool  # True if the selected action was tied with another within TOLERANCE
    tie_groups: tuple[tuple[str, ...], ...] = ()  # every group of actions tied within TOLERANCE
    tie_breaker_order: tuple[str, ...] = ()  # documented order in which ties were resolved


class UtilityDecisionAnalysis(FrozenModel):
    profile: UtilityProfile
    context: DecisionContext
    actions: tuple[ActionExpectedUtility, ...] = Field(min_length=1)
    ranking: UtilityRanking
    selected_action_id: str = NonEmptyStr
    priority: Priority

    @property
    def selected(self) -> ActionExpectedUtility:
        return next(a for a in self.actions if a.action_id == self.selected_action_id)

    @model_validator(mode="after")
    def _selection_must_be_the_best_eligible_action(self) -> Self:
        by_id = {a.action_id: a for a in self.actions}
        if len(by_id) != len(self.actions):
            raise ValueError("action IDs must be unique")
        eligible = [a for a in self.actions if a.eligible]
        if any(a.eligible != (a.rank is not None) for a in self.actions):
            raise ValueError("exactly the eligible actions must be ranked")

        ranked = sorted(eligible, key=lambda a: a.rank or 0)
        if [a.rank for a in ranked] != list(range(1, len(ranked) + 1)):
            raise ValueError("ranks must be consecutive positive integers starting at 1")
        if tuple(a.action_id for a in ranked) != self.ranking.ranked_action_ids:
            raise ValueError("ranking must list the eligible actions in rank order")

        selected = by_id.get(self.selected_action_id)
        if selected is None:
            raise ValueError("selected action is not one of the candidates analysed")
        if not selected.eligible:
            raise ValueError("selected action must be eligible")
        if selected.rank != 1:
            raise ValueError("selected action must have rank 1")
        if selected.expected_utility < max(a.expected_utility for a in eligible) - TOLERANCE:
            raise ValueError("selected action must have the highest eligible expected utility")

        for action in self.actions:
            for utility in action.outcome_utilities:
                for part in utility.contributions:
                    if abs(part.weight - self.profile.weight_of(part.criterion)) > TOLERANCE:
                        raise ValueError("breakdown weights must match the utility profile")
        return self
