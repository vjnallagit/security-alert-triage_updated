"""Typed models for the Goal-Based Agent: goal, candidate actions, outcomes, plans.

Everything here is qualitative. There are deliberately no numeric scores, weights,
probabilities or utilities: the Goal-Based Agent asks "does this action achieve the
goal?", not "which action scores highest?" (that is the future Utility-Based Agent).
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from models.schemas import NonEmptyStr, Priority, StrictModel

# Mirrors agents.common.TriageDecision. It is repeated here (and checked by a test)
# so that models/ never has to import from agents/.
DecisionName = Literal[
    "MONITOR",
    "COLLECT_MORE_EVIDENCE",
    "INVESTIGATE",
    "ESCALATE_TO_SECURITY_ANALYST",
]


class FrozenModel(StrictModel):
    """Immutable, so the goal and candidate catalogue cannot be altered after creation."""

    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class GoalCondition(StrEnum):
    """Conditions a predicted outcome must satisfy to achieve the goal."""

    ACTIVE_RISK_ADDRESSED = "ACTIVE_RISK_ADDRESSED"
    CRITICAL_ASSET_PROTECTED = "CRITICAL_ASSET_PROTECTED"
    EVIDENCE_GAP_ADDRESSED = "EVIDENCE_GAP_ADDRESSED"
    PROPORTIONATE_RESPONSE = "PROPORTIONATE_RESPONSE"
    UNCERTAINTY_PRESERVED = "UNCERTAINTY_PRESERVED"
    HUMAN_CONTROL_PRESERVED = "HUMAN_CONTROL_PRESERVED"


class ProhibitedCondition(StrEnum):
    """Things no acceptable plan may bring about."""

    AUTOMATIC_USER_BLOCKING = "AUTOMATIC_USER_BLOCKING"
    AUTOMATIC_ACCOUNT_DISABLING = "AUTOMATIC_ACCOUNT_DISABLING"
    AUTOMATIC_MACHINE_QUARANTINE = "AUTOMATIC_MACHINE_QUARANTINE"
    AUTOMATIC_FIREWALL_MODIFICATION = "AUTOMATIC_FIREWALL_MODIFICATION"
    OFFENSIVE_ACTION = "OFFENSIVE_ACTION"
    UNSUPPORTED_COMPROMISE_CLAIM = "UNSUPPORTED_COMPROMISE_CLAIM"
    CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION = (
        "CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION"
    )


class StateFact(StrEnum):
    """Facts about the current state; values are CurrentState field names."""

    ACTIVE_RISK_INDICATED = "active_risk_indicated"
    FAILED_LOGINS_PRESENT = "failed_logins_present"
    SOURCE_NOT_KNOWN = "source_not_known"


class PredictedEffect(StrEnum):
    """What recommending an action is expected to achieve."""

    PASSIVE_WATCH = "PASSIVE_WATCH"
    REDUCES_EVIDENCE_GAP = "REDUCES_EVIDENCE_GAP"
    ACTIVE_INVESTIGATION = "ACTIVE_INVESTIGATION"
    ANALYST_REVIEW = "ANALYST_REVIEW"


class RiskAttention(StrEnum):
    """Qualitative level of attention the risk is predicted to receive."""

    WATCHING = "WATCHING"
    EVIDENCE_REVIEW = "EVIDENCE_REVIEW"
    ACTIVE_INVESTIGATION = "ACTIVE_INVESTIGATION"
    ANALYST_REVIEW = "ANALYST_REVIEW"


# --------------------------------------------------------------------------
# Current state, goal and candidates
# --------------------------------------------------------------------------


class CurrentState(FrozenModel):
    """Concise interpretation of one SecurityScenario (no history, no counters)."""

    scenario_id: str = NonEmptyStr
    target_account: str = NonEmptyStr
    asset_name: str = NonEmptyStr
    privileged_account_at_risk: bool
    critical_asset_at_risk: bool
    high_severity: bool
    weak_evidence: bool
    impact_confirmed: bool
    evidence_gap_exists: bool
    failed_logins_present: bool
    source_not_known: bool
    active_risk_indicated: bool
    analyst_review_needed: bool

    def holds(self, fact: StateFact) -> bool:
        return bool(getattr(self, fact.value))


class SecurityGoal(FrozenModel):
    goal_id: str = NonEmptyStr
    name: str = NonEmptyStr
    description: str = NonEmptyStr
    required_conditions: tuple[GoalCondition, ...] = Field(min_length=1)
    prohibited_conditions: tuple[ProhibitedCondition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _goal_must_keep_every_safety_prohibition(self) -> Self:
        missing = set(ProhibitedCondition) - set(self.prohibited_conditions)
        if missing:
            names = ", ".join(sorted(c.value for c in missing))
            raise ValueError(f"goal must prohibit every unsafe condition; missing: {names}")
        if len(set(self.required_conditions)) != len(self.required_conditions):
            raise ValueError("required_conditions must not contain duplicates")
        return self


class PlanStep(FrozenModel):
    text: str = NonEmptyStr
    only_if: StateFact | None = None  # step is included only when this fact holds


class CandidateAction(FrozenModel):
    """A possible recommendation. It is evaluated, never executed."""

    action_id: str = NonEmptyStr
    decision: DecisionName
    description: str = NonEmptyStr
    preconditions: tuple[StateFact, ...] = ()
    predicted_effects: tuple[PredictedEffect, ...] = Field(min_length=1)
    plan_steps: tuple[PlanStep, ...] = Field(min_length=1)
    is_consequential: bool = False
    requires_human_authorization: bool = False
    # Prohibited conditions this action would cause; every catalogue action has none.
    side_effects: tuple[ProhibitedCondition, ...] = ()
    limitations: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Predictions, evaluation and plan
# --------------------------------------------------------------------------


class PredictedOutcome(FrozenModel):
    action_id: str = NonEmptyStr
    decision: DecisionName
    predicted_state: str = NonEmptyStr
    evidence_gap_addressed: bool
    risk_attention_level: RiskAttention
    active_risk_addressed: bool
    critical_asset_protected: bool
    preserves_uncertainty: bool
    requires_human_review: bool
    unsafe_automation_avoided: bool
    unsupported_claim_avoided: bool
    limitations: tuple[str, ...] = ()


class GoalEvaluation(FrozenModel):
    action_id: str = NonEmptyStr
    goal_id: str = NonEmptyStr
    satisfies_goal: bool
    satisfied_conditions: tuple[GoalCondition, ...] = ()
    unsatisfied_conditions: tuple[GoalCondition, ...] = ()
    violated_prohibited_conditions: tuple[ProhibitedCondition, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _verdict_must_match_findings(self) -> Self:
        failed = bool(self.unsatisfied_conditions or self.violated_prohibited_conditions)
        if self.satisfies_goal and failed:
            raise ValueError("an action that fails conditions or violates a prohibition cannot satisfy the goal")
        if not self.satisfies_goal and not self.rejection_reasons:
            raise ValueError("a rejected action must state at least one reason")
        if self.satisfies_goal and self.rejection_reasons:
            raise ValueError("an acceptable action must not carry rejection reasons")
        return self


class ActionPlan(FrozenModel):
    plan_id: str = NonEmptyStr
    goal_id: str = NonEmptyStr
    selected_action_id: str = NonEmptyStr
    primary_decision: DecisionName
    steps: tuple[str, ...] = Field(min_length=2, max_length=5)
    selection_reason: str = NonEmptyStr
    predicted_outcome: PredictedOutcome
    goal_evaluation: GoalEvaluation

    @model_validator(mode="after")
    def _plan_must_be_consistent_and_satisfy_goal(self) -> Self:
        ids = {self.selected_action_id, self.predicted_outcome.action_id, self.goal_evaluation.action_id}
        if len(ids) != 1:
            raise ValueError("plan, predicted outcome and evaluation must refer to the same action")
        if self.goal_evaluation.goal_id != self.goal_id:
            raise ValueError("evaluation was made against a different goal")
        if not self.goal_evaluation.satisfies_goal:
            raise ValueError("a selected plan must satisfy the goal")
        if self.predicted_outcome.decision != self.primary_decision:
            raise ValueError("primary decision must match the predicted outcome")
        return self


class GoalDeliberation(FrozenModel):
    """Everything the agent considered for one scenario, in one inspectable object."""

    goal: SecurityGoal
    state: CurrentState
    candidates: tuple[CandidateAction, ...] = Field(min_length=1)
    outcomes: tuple[PredictedOutcome, ...]
    evaluations: tuple[GoalEvaluation, ...]
    plan: ActionPlan
    priority: Priority

    @model_validator(mode="after")
    def _selected_plan_must_reference_a_considered_candidate(self) -> Self:
        candidate_ids = [c.action_id for c in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate action IDs must be unique")
        if [o.action_id for o in self.outcomes] != candidate_ids:
            raise ValueError("every candidate needs exactly one predicted outcome, in order")
        if [e.action_id for e in self.evaluations] != candidate_ids:
            raise ValueError("every candidate needs exactly one goal evaluation, in order")
        if self.plan.selected_action_id not in candidate_ids:
            raise ValueError("selected action is not one of the candidates considered")
        if self.plan.goal_id != self.goal.goal_id:
            raise ValueError("plan was selected for a different goal")
        return self
