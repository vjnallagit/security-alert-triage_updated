"""Typed models for the Learning Agent: experience statistics, feedback and updates.

Experience is stored as plain success/failure counts. Every rate is derived from the
counts and never stored. All values are SIMULATED academic data, not real SOC statistics.
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from models.goal_models import DecisionName, FrozenModel, ProhibitedCondition
from models.schemas import NonEmptyStr

# Beta(1,1) / Laplace smoothing: one imaginary success and one imaginary failure.
SMOOTHING_PRIOR_SUCCESSES = 1
SMOOTHING_PRIOR_FAILURES = 1
RATE_TOLERANCE = 1e-9


class FeedbackOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"  # never counted as success or failure


class LearningPattern(StrEnum):
    """Fact-derived situation types. Scenario IDs are never learning keys."""

    PRIVILEGED_LOGIN_ANOMALY = "privileged_login_anomaly"
    CRITICAL_SERVICE_CORRELATED_ACCESS = "critical_service_correlated_access"
    LOW_SEVERITY_LIMITED_EVIDENCE = "low_severity_limited_evidence"
    GENERAL_SECURITY_ALERT = "general_security_alert"  # safe fallback


class ExperienceStats(FrozenModel):
    """Success/failure counts for one pattern and action, with derived rates."""

    success: int = Field(ge=0)
    failure: int = Field(ge=0)

    @property
    def observation_count(self) -> int:
        return self.success + self.failure

    @property
    def raw_success_rate(self) -> float | None:
        """success / (success + failure); None (not 0.0) when nothing was observed."""
        return self.success / self.observation_count if self.observation_count else None

    @property
    def smoothed_success_rate(self) -> float:
        """(S + 1) / (S + F + 2): an academic preference estimate, not a real probability."""
        return (self.success + SMOOTHING_PRIOR_SUCCESSES) / (
            self.observation_count + SMOOTHING_PRIOR_SUCCESSES + SMOOTHING_PRIOR_FAILURES
        )

    def after(self, outcome: FeedbackOutcome) -> Self:
        """Counts after one piece of feedback. UNKNOWN changes nothing."""
        if outcome is FeedbackOutcome.SUCCESS:
            return type(self)(success=self.success + 1, failure=self.failure)
        if outcome is FeedbackOutcome.FAILURE:
            return type(self)(success=self.success, failure=self.failure + 1)
        return self


class LearningFeedback(FrozenModel):
    """Explicit, observed outcome of a previously recommended action."""

    pattern: str = NonEmptyStr
    action: DecisionName
    outcome: FeedbackOutcome
    note: str | None = None


class CriticAssessment(FrozenModel):
    """The Critic's verdict on feedback. The Critic never changes experience."""

    usable: bool
    outcome: FeedbackOutcome
    reason: str = NonEmptyStr


class LearningUpdate(FrozenModel):
    """What the Learning Element did with one piece of feedback."""

    pattern: str = NonEmptyStr
    action: DecisionName
    outcome: FeedbackOutcome
    before_success: int = Field(ge=0)
    before_failure: int = Field(ge=0)
    after_success: int = Field(ge=0)
    after_failure: int = Field(ge=0)
    before_raw_rate: float | None
    after_raw_rate: float | None
    before_smoothed_rate: float
    after_smoothed_rate: float
    changed: bool
    critic_assessment: CriticAssessment
    learning_explanation: str = NonEmptyStr

    @model_validator(mode="after")
    def _update_must_match_the_feedback(self) -> Self:
        success_step = self.after_success - self.before_success
        failure_step = self.after_failure - self.before_failure
        expected = {
            FeedbackOutcome.SUCCESS: (1, 0),
            FeedbackOutcome.FAILURE: (0, 1),
            FeedbackOutcome.UNKNOWN: (0, 0),
        }[self.outcome]
        if self.changed:
            if (success_step, failure_step) != expected or expected == (0, 0):
                raise ValueError("a changed update must add exactly one success or one failure matching the outcome")
        elif (success_step, failure_step) != (0, 0):
            raise ValueError("an unchanged update must leave the counts as they were")
        if self.changed and not self.critic_assessment.usable:
            raise ValueError("experience cannot change when the critic found the feedback unusable")
        return self


class LearningCandidate(FrozenModel):
    """A recommendation the agent may select. It is never executed."""

    action_id: str = NonEmptyStr
    decision: DecisionName
    is_consequential: bool = False
    requires_human_authorization: bool = False
    side_effects: tuple[ProhibitedCondition, ...] = ()


class SituationFacts(FrozenModel):
    """Current-scenario facts that constrain which actions are acceptable."""

    pattern: str = NonEmptyStr
    privileged_account_pattern: bool
    critical_service_pattern: bool
    high_severity: bool
    confirmed_impact: bool
    weak_low_severity_alert: bool
    active_risk_indicated: bool
    analyst_review_needed: bool


class ProblemGeneratorGuidance(FrozenModel):
    """Exploration advice when history is thin. It never overrides current risk or safety."""

    min_experience_count: int = Field(ge=1)
    experience_sufficient: bool
    insufficient_decisions: tuple[str, ...] = ()
    recommended_decision: DecisionName | None = None
    used: bool  # True when thin history made the guidance, not the history ranking, decide
    overridden_by_current_risk: bool = False
    guidance: str = NonEmptyStr

    @model_validator(mode="after")
    def _guidance_must_be_consistent(self) -> Self:
        if self.experience_sufficient and self.used:
            raise ValueError("guidance is used only when experience is insufficient")
        if not self.experience_sufficient and not self.insufficient_decisions:
            raise ValueError("insufficient experience must name the candidates that lack history")
        return self
