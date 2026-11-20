"""Learning Agent: experience + outcome feedback -> updated preferences -> changed behavior.

The four classical learning-agent components are separate classes:

  PerformanceElement  chooses the current recommendation from the scenario and history
  Critic              judges whether explicit outcome feedback is usable (never edits data)
  LearningElement     turns usable feedback into a persisted success/failure update
  ProblemGenerator    advises evidence gathering when history is too thin to trust

DECIDING AND LEARNING ARE SEPARATE. decide() only reads experience. Experience changes
only through record_feedback()/learn(), because producing a recommendation does not tell
the agent whether it worked.

How this differs from the other agents: the Model-Based Reflex Agent remembers observations
about the ENVIRONMENT; this agent updates ACTION preferences from SUCCESS/FAILURE feedback.
The Utility-Based Agent evaluates fixed utilities and probabilities; this agent computes no
utility and no probabilities, it ranks safe candidates by smoothed historical success.

All counts are simulated academic data. Safety and current-scenario adequacy are hard
constraints applied BEFORE any history is consulted, so no success rate can make an unsafe
or inadequate action selectable.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from agents.base import BaseAgent
from agents.common import (
    HUMAN_AUTHORIZATION_STATEMENT,
    MULTIPLE_CORRELATED_ALERTS_THRESHOLD,
    NO_ACTION_STATEMENT,
    RECOMMENDED_ACTIONS,
    REPEATED_FAILED_LOGIN_THRESHOLD,
    TriageDecision,
    safety_metadata,
    scenario_evidence,
    uncertainty_statement,
)
from models.goal_models import ProhibitedCondition
from models.learning_models import (
    RATE_TOLERANCE,
    CriticAssessment,
    ExperienceStats,
    FeedbackOutcome,
    LearningCandidate,
    LearningFeedback,
    LearningPattern,
    LearningUpdate,
    ProblemGeneratorGuidance,
    SituationFacts,
)
from models.schemas import (
    AccountPrivilege,
    AgentDecision,
    AgentType,
    AssetCriticality,
    EvidenceStrength,
    Priority,
    SecurityScenario,
    Severity,
    SourceFamiliarity,
)
from services.experience_repository import ExperienceRepository

AGENT_NAME = "Learning Agent"
DECISION_MECHANISM = "feedback_driven_experience_learning"

# ---------------------------------------------------------------------------
# Learning policy: the only place the thresholds live
# ---------------------------------------------------------------------------
MIN_EXPERIENCE_COUNT = 3  # observations an action needs before its history is trusted
SMOOTHING_DESCRIPTION = "Beta(1,1) / Laplace: (S + 1) / (S + F + 2)"
SELECTION_BASIS_HISTORY = "safe_action_then_smoothed_historical_success"
SELECTION_BASIS_SINGLE = "only_eligible_safe_action"
SELECTION_BASIS_PROBLEM_GENERATOR = "problem_generator_evidence_gathering"
SELECTION_BASIS_FALLBACK = "deterministic_safe_fallback"
LIMITATION_NOTE = (
    "Experience counts are simulated academic data, not real SOC statistics; the smoothed "
    "value is a preference estimate, not a probability, and implies no causal claim."
)

# Candidates ordered least to most disruptive; that order is the final deterministic tie-breaker.
LEARNING_CANDIDATES: tuple[LearningCandidate, ...] = (
    LearningCandidate(action_id="LA-ACT-MONITOR", decision=TriageDecision.MONITOR.value),
    LearningCandidate(action_id="LA-ACT-COLLECT", decision=TriageDecision.COLLECT_MORE_EVIDENCE.value),
    LearningCandidate(action_id="LA-ACT-INVESTIGATE", decision=TriageDecision.INVESTIGATE.value,
                      requires_human_authorization=True),
    LearningCandidate(action_id="LA-ACT-ESCALATE", decision=TriageDecision.ESCALATE_TO_SECURITY_ANALYST.value,
                      requires_human_authorization=True),
)


class InvalidFeedbackError(ValueError):
    """Feedback cannot be used (for example, an unknown pattern); experience is left unchanged."""


class NoSafeActionError(RuntimeError):
    """No candidate is both safe and adequate for the current scenario."""


@dataclass(frozen=True)
class CandidateView:
    """One candidate with its eligibility verdict and its historical counts."""

    candidate: LearningCandidate
    eligible: bool
    reasons: tuple[str, ...]
    stats: ExperienceStats

    @property
    def decision(self) -> str:
        return self.candidate.decision


@dataclass(frozen=True)
class Selection:
    view: CandidateView
    basis: str
    ranked_decisions: tuple[str, ...]
    default_decision: str
    tie_breaker_used: bool
    margin: float | None  # smoothed-rate lead over the next-best eligible action

    @property
    def experience_changed_default(self) -> bool:
        return self.view.decision != self.default_decision


# ---------------------------------------------------------------------------
# Pattern identification: from facts only, never from scenario IDs
# ---------------------------------------------------------------------------


def classify_pattern(scenario: SecurityScenario) -> LearningPattern:
    return LearningPattern(_situation_facts(scenario).pattern)


def _situation_facts(scenario: SecurityScenario) -> SituationFacts:
    privileged = (
        scenario.account_privilege == AccountPrivilege.PRIVILEGED
        and scenario.source_familiarity == SourceFamiliarity.UNFAMILIAR
        and scenario.failed_login_count >= REPEATED_FAILED_LOGIN_THRESHOLD
    )
    critical = (
        scenario.asset_criticality == AssetCriticality.CRITICAL
        and scenario.unusual_access
        and scenario.correlated_alert_count >= MULTIPLE_CORRELATED_ALERTS_THRESHOLD
    )
    high_severity = scenario.severity in (Severity.HIGH, Severity.CRITICAL)
    weak_low = (
        scenario.severity == Severity.LOW
        and scenario.evidence_strength == EvidenceStrength.LIMITED
        and not scenario.confirmed_impact
    )
    if critical:
        pattern = LearningPattern.CRITICAL_SERVICE_CORRELATED_ACCESS
    elif privileged:
        pattern = LearningPattern.PRIVILEGED_LOGIN_ANOMALY
    elif weak_low:
        pattern = LearningPattern.LOW_SEVERITY_LIMITED_EVIDENCE
    else:
        pattern = LearningPattern.GENERAL_SECURITY_ALERT
    return SituationFacts(
        pattern=pattern.value,
        privileged_account_pattern=privileged,
        critical_service_pattern=critical,
        high_severity=high_severity,
        confirmed_impact=scenario.confirmed_impact,
        weak_low_severity_alert=weak_low,
        active_risk_indicated=privileged or critical or high_severity or scenario.confirmed_impact,
        analyst_review_needed=critical or scenario.confirmed_impact,
    )


# ---------------------------------------------------------------------------
# Problem Generator
# ---------------------------------------------------------------------------


class ProblemGenerator:
    """Deterministic and safe: it only ever advises evidence gathering, never experiments."""

    @staticmethod
    def assess(situation: SituationFacts, views: Sequence[CandidateView], default_decision: str) -> ProblemGeneratorGuidance:
        eligible = [v for v in views if v.eligible]
        below_minimum = tuple(v.decision for v in eligible if v.stats.observation_count < MIN_EXPERIENCE_COUNT)
        sufficient = any(v.stats.observation_count >= MIN_EXPERIENCE_COUNT for v in eligible)

        if sufficient:
            note = (
                f"Fewer than {MIN_EXPERIENCE_COUNT} observations exist for {', '.join(below_minimum)}; "
                "their smoothed estimates stay near 0.50, so they are neither favored nor penalised unfairly."
                if below_minimum
                else f"Every eligible action has at least {MIN_EXPERIENCE_COUNT} observations."
            )
            return ProblemGeneratorGuidance(
                min_experience_count=MIN_EXPERIENCE_COUNT, experience_sufficient=True,
                insufficient_decisions=below_minimum, used=False,
                guidance=f"Historical experience is sufficient. {note}",
            )

        collect = TriageDecision.COLLECT_MORE_EVIDENCE.value
        can_explore = any(v.decision == collect for v in eligible)
        if can_explore:
            advice = (f"{collect} is acceptable for the current situation, so it is recommended as safe "
                      "information gathering.")
        else:
            advice = ("Current risk requires an active response, so evidence-gathering alone cannot replace "
                      f"it; using the deterministic safe fallback ({default_decision}).")
        return ProblemGeneratorGuidance(
            min_experience_count=MIN_EXPERIENCE_COUNT, experience_sufficient=False,
            insufficient_decisions=below_minimum,
            recommended_decision=collect if can_explore else None,
            used=True, overridden_by_current_risk=not can_explore and situation.active_risk_indicated,
            guidance=(f"Historical experience is insufficient (fewer than {MIN_EXPERIENCE_COUNT} observations "
                      "for every eligible action). Gather additional evidence or use the deterministic safe "
                      f"fallback rather than treating weak experience as reliable. {advice}"),
        )


# ---------------------------------------------------------------------------
# Performance Element
# ---------------------------------------------------------------------------


class PerformanceElement:
    """Chooses among safe, adequate candidates; history only ranks what constraints allow."""

    @staticmethod
    def default_decision(situation: SituationFacts) -> str:
        """Cold-start response when history cannot decide: the one the situation itself calls for."""
        if situation.analyst_review_needed:
            return TriageDecision.ESCALATE_TO_SECURITY_ANALYST.value
        if situation.active_risk_indicated:
            return TriageDecision.INVESTIGATE.value
        return TriageDecision.COLLECT_MORE_EVIDENCE.value

    @staticmethod
    def check_eligibility(situation: SituationFacts, candidate: LearningCandidate) -> tuple[bool, tuple[str, ...]]:
        """Hard constraints. History is never consulted here, so it can never override them."""
        reasons: list[str] = []
        violations = [
            effect for effect in candidate.side_effects
            # Describing impact the scenario itself confirms is not an unsupported claim.
            if not (effect is ProhibitedCondition.UNSUPPORTED_COMPROMISE_CLAIM and situation.confirmed_impact)
        ]
        if candidate.is_consequential and not candidate.requires_human_authorization:
            violations.append(ProhibitedCondition.CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION)
        reasons.extend(f"Prohibited: would bring about {v.value}." for v in dict.fromkeys(violations))

        decision = candidate.decision
        if decision in (TriageDecision.MONITOR.value, TriageDecision.COLLECT_MORE_EVIDENCE.value):
            if situation.active_risk_indicated:
                reasons.append(f"Inadequate: {decision} alone does not address the active risk in the current scenario.")
        elif decision == TriageDecision.INVESTIGATE.value:
            if situation.analyst_review_needed:
                reasons.append("Inadequate: the current scenario needs analyst review, which INVESTIGATE alone does not provide.")
        elif decision == TriageDecision.ESCALATE_TO_SECURITY_ANALYST.value:
            if not situation.active_risk_indicated:
                reasons.append("Disproportionate: no active risk indicator justifies analyst escalation.")
        return not reasons, tuple(reasons)

    @classmethod
    def select(cls, situation: SituationFacts, views: Sequence[CandidateView], guidance: ProblemGeneratorGuidance) -> Selection:
        eligible = [v for v in views if v.eligible]  # views are in least-disruptive-first order
        if not eligible:
            raise NoSafeActionError("No candidate action is safe and adequate for the current scenario.")
        default = cls.default_decision(situation)
        default_view = next((v for v in eligible if v.decision == default), eligible[0])

        if len(eligible) == 1:
            return Selection(eligible[0], SELECTION_BASIS_SINGLE, (eligible[0].decision,), default_view.decision, False, None)

        if guidance.used:
            recommended = next((v for v in eligible if v.decision == guidance.recommended_decision), None)
            chosen = recommended or default_view
            basis = SELECTION_BASIS_PROBLEM_GENERATOR if recommended else SELECTION_BASIS_FALLBACK
            return Selection(chosen, basis, tuple(v.decision for v in eligible), default_view.decision, False, None)

        # Sufficient history: strongest smoothed success first. Ties within RATE_TOLERANCE go to the
        # situation's default response, then the less disruptive action (list order).
        order = {v.decision: index for index, v in enumerate(eligible)}
        by_rate = sorted(eligible, key=lambda v: (-v.stats.smoothed_success_rate, order[v.decision]))
        best_rate = by_rate[0].stats.smoothed_success_rate
        tied = [v for v in by_rate if best_rate - v.stats.smoothed_success_rate <= RATE_TOLERANCE]
        tied.sort(key=lambda v: (v.decision != default_view.decision, order[v.decision]))
        ranked = tied + [v for v in by_rate if v not in tied]
        chosen = ranked[0]
        runner_up = ranked[1]
        return Selection(
            chosen, SELECTION_BASIS_HISTORY, tuple(v.decision for v in ranked), default_view.decision,
            len(tied) > 1, chosen.stats.smoothed_success_rate - runner_up.stats.smoothed_success_rate,
        )

    @staticmethod
    def priority_for(situation: SituationFacts) -> Priority:
        """Grounded in current risk only, so learned history can neither inflate nor hide risk."""
        if situation.analyst_review_needed:
            return Priority.CRITICAL
        if situation.active_risk_indicated:
            return Priority.HIGH
        return Priority.LOW if situation.weak_low_severity_alert else Priority.MEDIUM


# ---------------------------------------------------------------------------
# Critic and Learning Element
# ---------------------------------------------------------------------------


class Critic:
    """Judges feedback. It validates and classifies; it never modifies experience."""

    @staticmethod
    def evaluate(feedback: LearningFeedback, known_patterns: Collection[str]) -> CriticAssessment:
        if feedback.pattern not in known_patterns:
            raise InvalidFeedbackError(
                f"Unknown learning pattern {feedback.pattern!r}; known patterns: {sorted(known_patterns)}"
            )
        if feedback.outcome is FeedbackOutcome.UNKNOWN:
            return CriticAssessment(
                usable=False, outcome=feedback.outcome,
                reason="UNKNOWN feedback carries no information about success or failure, so it is not counted.",
            )
        return CriticAssessment(
            usable=True, outcome=feedback.outcome,
            reason=f"{feedback.outcome.value} feedback for {feedback.action} is valid and is counted.",
        )


class LearningElement:
    """Applies critic-approved feedback to exactly one experience entry and persists it."""

    def __init__(self, repository: ExperienceRepository) -> None:
        self._repository = repository

    def apply(self, feedback: LearningFeedback, assessment: CriticAssessment) -> LearningUpdate:
        if assessment.usable:
            before, after = self._repository.record_outcome(feedback.pattern, feedback.action, feedback.outcome)
        else:
            before = after = self._repository.get_stats(feedback.pattern, feedback.action)
        changed = after != before
        if changed:
            explanation = (
                f"{feedback.outcome.value} feedback for {feedback.action} under pattern {feedback.pattern}: "
                f"successes {before.success} -> {after.success}, failures {before.failure} -> {after.failure}; "
                f"Beta(1,1)-smoothed estimate {before.smoothed_success_rate:.4f} -> {after.smoothed_success_rate:.4f}. "
                "Only this pattern/action entry was updated."
            )
        else:
            explanation = f"{assessment.reason} Experience was left unchanged."
        return LearningUpdate(
            pattern=feedback.pattern, action=feedback.action, outcome=feedback.outcome,
            before_success=before.success, before_failure=before.failure,
            after_success=after.success, after_failure=after.failure,
            before_raw_rate=before.raw_success_rate, after_raw_rate=after.raw_success_rate,
            before_smoothed_rate=before.smoothed_success_rate, after_smoothed_rate=after.smoothed_success_rate,
            changed=changed, critic_assessment=assessment, learning_explanation=explanation,
        )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _counts(stats: ExperienceStats) -> str:
    successes = f"{stats.success} {'success' if stats.success == 1 else 'successes'}"
    failures = f"{stats.failure} {'failure' if stats.failure == 1 else 'failures'}"
    return f"{successes} and {failures}"


class LearningAgent(BaseAgent):
    """Combines the four components. The only state it owns is the injected experience repository."""

    name = AGENT_NAME

    def __init__(self, repository: ExperienceRepository | None = None) -> None:
        self._repository = repository or ExperienceRepository()
        self._performance_element = PerformanceElement()
        self._critic = Critic()
        self._learning_element = LearningElement(self._repository)
        self._problem_generator = ProblemGenerator()

    # --- deciding (read-only) -----------------------------------------------

    @staticmethod
    def _classify_pattern(scenario: SecurityScenario) -> str:
        return _situation_facts(scenario).pattern

    def decide(self, scenario: SecurityScenario) -> AgentDecision:
        situation = _situation_facts(scenario)
        history = self._repository.get_pattern_history(situation.pattern)
        views = tuple(self._view(situation, candidate, history) for candidate in LEARNING_CANDIDATES)
        if not any(view.eligible for view in views):
            raise NoSafeActionError("No candidate action is safe and adequate for the current scenario.")
        default = self._performance_element.default_decision(situation)
        guidance = self._problem_generator.assess(situation, views, default)
        selection = self._performance_element.select(situation, views, guidance)
        return self._build_decision(scenario, situation, views, guidance, selection)

    def _view(self, situation: SituationFacts, candidate: LearningCandidate, history: dict[str, ExperienceStats]) -> CandidateView:
        eligible, reasons = self._performance_element.check_eligibility(situation, candidate)
        stats = history.get(candidate.decision, ExperienceStats(success=0, failure=0))
        return CandidateView(candidate, eligible, reasons, stats)

    # --- learning (explicit feedback only) ----------------------------------

    def record_feedback(
        self, scenario: SecurityScenario, decision: str, outcome: FeedbackOutcome, note: str | None = None
    ) -> LearningUpdate:
        """Record the observed outcome of a recommended action for this scenario's pattern."""
        feedback = LearningFeedback(
            pattern=self._classify_pattern(scenario), action=decision, outcome=outcome, note=note
        )
        return self.learn(feedback)

    def learn(self, feedback: LearningFeedback) -> LearningUpdate:
        """Critic first, then Learning Element. Feedback never authorizes any security action."""
        known = {p.value for p in LearningPattern} | set(self._repository.patterns())
        assessment = self._critic.evaluate(feedback, known)
        return self._learning_element.apply(feedback, assessment)

    # --- explanation ----------------------------------------------------------

    @classmethod
    def _build_decision(
        cls, scenario: SecurityScenario, situation: SituationFacts, views: Sequence[CandidateView],
        guidance: ProblemGeneratorGuidance, selection: Selection,
    ) -> AgentDecision:
        chosen = selection.view
        decision = chosen.decision
        experience_used = cls._experience_used(situation, views, guidance, selection)
        return AgentDecision(
            agent_name=AGENT_NAME,
            agent_type=AgentType.LEARNING,
            priority=PerformanceElement.priority_for(situation),
            decision=decision,
            reasoning=cls._reasoning(situation, views, guidance, selection),
            evidence=[
                *scenario_evidence(scenario),
                f"Learning pattern: {situation.pattern}",
                f"History for {decision}: {_counts(chosen.stats)}",
            ],
            uncertainty=uncertainty_statement(scenario),
            likely_interpretation=cls._interpretation(situation),
            recommended_actions=list(RECOMMENDED_ACTIONS[TriageDecision(decision)]),
            audit_note=(
                f"Scenario {scenario.scenario_id} | Agent: {AGENT_NAME} | Pattern: {situation.pattern} | "
                f"Decision: {decision} | Basis: {selection.basis} | {NO_ACTION_STATEMENT} "
                f"{HUMAN_AUTHORIZATION_STATEMENT} This decision did not change experience; "
                "learning requires explicit outcome feedback."
            ),
            experience_used=experience_used,
            metadata={
                "decision_mechanism": DECISION_MECHANISM,
                "learning_pattern": situation.pattern,
                "learning_components": {
                    "performance_element": True, "critic": True, "learning_element": True, "problem_generator": True,
                },
                "candidate_experience": experience_used["candidates"],
                "selected_experience": experience_used["selected"],
                "experience_sufficient": guidance.experience_sufficient,
                "problem_generator_used": guidance.used,
                "problem_generator": guidance.model_dump(mode="json"),
                "selection_basis": selection.basis,
                "ranked_decisions": list(selection.ranked_decisions),
                "default_decision": selection.default_decision,
                "experience_changed_default": selection.experience_changed_default,
                "tie_breaker_used": selection.tie_breaker_used,
                "smoothing": SMOOTHING_DESCRIPTION,
                "min_experience_count": MIN_EXPERIENCE_COUNT,
                "decide_modifies_experience": False,
                "limitation_note": LIMITATION_NOTE,
                "scenario_id": scenario.scenario_id,
                "uses_learning": True,
                "uses_utility_scoring": False,
                "uses_goal_satisfaction_selection": False,
                "uses_llm": False,
                **safety_metadata(),
            },
        )

    @staticmethod
    def _experience_used(
        situation: SituationFacts, views: Sequence[CandidateView], guidance: ProblemGeneratorGuidance, selection: Selection
    ) -> dict[str, Any]:
        def describe(view: CandidateView) -> dict[str, Any]:
            return {
                "decision": view.decision,
                "action_id": view.candidate.action_id,
                "eligible": view.eligible,
                "ineligibility_reasons": [] if view.eligible else list(view.reasons),
                "success": view.stats.success,
                "failure": view.stats.failure,
                "observation_count": view.stats.observation_count,
                "raw_success_rate": view.stats.raw_success_rate,
                "smoothed_success_rate": view.stats.smoothed_success_rate,
                "sufficient": view.stats.observation_count >= MIN_EXPERIENCE_COUNT,
            }

        return {
            "pattern": situation.pattern,
            "smoothing": SMOOTHING_DESCRIPTION,
            "min_experience_count": MIN_EXPERIENCE_COUNT,
            "experience_sufficient": guidance.experience_sufficient,
            "selection_basis": selection.basis,
            "candidates": [describe(v) for v in views],
            "selected": describe(selection.view),
        }

    @classmethod
    def _reasoning(
        cls, situation: SituationFacts, views: Sequence[CandidateView], guidance: ProblemGeneratorGuidance,
        selection: Selection,
    ) -> str:
        eligible = [v for v in views if v.eligible]
        parts = [f"Pattern {situation.pattern} matched the current scenario facts ({cls._describe(situation)})."]
        for view in eligible:
            s = view.stats
            parts.append(
                f"{view.decision} has {_counts(s)} on record ({s.observation_count} observations); its raw "
                f"historical success rate is {_pct(s.raw_success_rate)}, while the Beta(1,1)-smoothed "
                f"estimate is {_pct(s.smoothed_success_rate)}."
            )
        for view in views:
            if not view.eligible:
                parts.append(
                    f"{view.decision} (history: {_counts(view.stats)}) was excluded by current-scenario "
                    f"constraints before history was used: {view.reasons[0]}"
                )
        parts.append(guidance.guidance)

        chosen = selection.view
        if selection.basis == SELECTION_BASIS_SINGLE:
            parts.append(f"{chosen.decision} is the only eligible action, so history could not change the outcome.")
        elif selection.basis == SELECTION_BASIS_HISTORY:
            others = [v for v in eligible if v is not chosen]
            lead = ", ".join(f"{v.decision} {_pct(v.stats.smoothed_success_rate)}" for v in others)
            if selection.tie_breaker_used:
                parts.append(
                    f"The best smoothed estimates were tied, so the deterministic tie-breaker (the situation's "
                    f"default response {selection.default_decision}, then the less disruptive action) selected {chosen.decision}."
                )
            else:
                parts.append(
                    f"Historical experience therefore supports {chosen.decision} "
                    f"({_pct(chosen.stats.smoothed_success_rate)}) over the other eligible alternatives ({lead})."
                )
            if selection.experience_changed_default:
                parts.append(
                    f"This differs from the situation's default response ({selection.default_decision}): "
                    "accumulated outcome feedback changed the preference."
                )
        else:
            parts.append(f"{chosen.decision} was selected by the Problem Generator / deterministic fallback, not by history.")
        impact = (
            "Impact is confirmed in the scenario."
            if situation.confirmed_impact
            else "Compromise is not confirmed; the recommendation supports investigation and human review only."
        )
        parts.append(f"{impact} No action was executed. This decision did not change experience; "
                     "learning occurs only when explicit outcome feedback is recorded.")
        return " ".join(parts)

    @staticmethod
    def _describe(situation: SituationFacts) -> str:
        if situation.critical_service_pattern:
            return "critical service with correlated alerts and unusual access"
        if situation.privileged_account_pattern:
            return "privileged account with repeated failed logins from an unfamiliar source"
        if situation.weak_low_severity_alert:
            return "low severity, limited evidence, no confirmed impact"
        return "no more specific pattern"

    @staticmethod
    def _interpretation(situation: SituationFacts) -> str:
        findings = []
        if situation.confirmed_impact:
            findings.append("Impact is confirmed in the scenario; analyst-led response is needed.")
        if situation.privileged_account_pattern:
            findings.append("Possible credential-guessing attempt against a privileged account; unconfirmed.")
        if situation.critical_service_pattern:
            findings.append("Possible serious incident on a critical service, or an undocumented but legitimate change; unconfirmed.")
        if not findings:
            findings.append(
                "Significant alert whose cause is not yet understood."
                if situation.high_severity
                else "Likely benign or one-off event; evidence is too weak to conclude."
            )
        return " ".join(findings)
