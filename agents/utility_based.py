"""Utility-Based Agent: choose the safe action with the highest expected utility.

Decision cycle (one focused method per step):
  1. _build_decision_context       scenario facts -> numeric needs (risk, stake, evidence)
  2. _define_utility_profile       six weighted criteria (profile UP-01)
  3. _generate_candidate_actions   the same four defensive actions every time
  4. _check_action_eligibility     safety/policy constraints; ineligible actions never win
  5. _generate_possible_outcomes   FAVORABLE / PARTIAL / UNFAVORABLE per action, with
                                   _estimate_outcome_probabilities and _score_outcome_criteria
  6. _calculate_outcome_utility    U(outcome) = sum(weight * score)
  7. _calculate_expected_utility   EU(a) = sum(P(outcome | a, scenario) * U(outcome))
  8. _rank_actions / _select_best_action
  9. _build_decision

Every number below is a transparent SIMULATION ASSUMPTION for this academic project,
not a statistical estimate. Nothing is random, learned or read from a file, so the same
scenario and profile always give the same result. Scenario IDs never enter a calculation.
"""

import math
from collections.abc import Sequence

from agents.base import BaseAgent
from agents.common import (
    HUMAN_AUTHORIZATION_STATEMENT,
    MULTIPLE_CORRELATED_ALERTS_THRESHOLD,
    NO_ACTION_STATEMENT,
    REPEATED_FAILED_LOGIN_THRESHOLD,
    safety_metadata,
    scenario_evidence,
    uncertainty_statement,
)
from models.goal_models import ProhibitedCondition
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

AGENT_NAME = "Utility-Based Agent"
DECISION_MECHANISM = "expected_utility_maximization"
EXPECTED_UTILITY_FORMULA = "EU(a)=sum(P(outcome|a,scenario)*U(outcome))"
OUTCOME_UTILITY_FORMULA = "U(outcome)=sum(weight_c*score_c(outcome))"
SIMULATION_NOTE = (
    "Probabilities and scores are transparent simulated assumptions for this academic "
    "demonstration, not real-world statistical estimates."
)
TIE_BREAKER_ORDER = (
    "higher expected OPERATIONAL_CONTINUITY (less disruptive)",
    "higher expected EVIDENCE_ALIGNMENT",
    "higher expected HUMAN_CONTROL",
    "stable candidate order",
)
SCORE_DECIMALS = 6
PROBABILITY_DECIMALS = 4
TIE_DECIMALS = 9  # secondary tie-break values closer than this are treated as equal


class NoEligibleActionError(RuntimeError):
    """Every candidate violated a safety constraint; nothing safe can be recommended."""


# ---------------------------------------------------------------------------
# Utility profile UP-01: the single, centralized place for criteria and weights
# ---------------------------------------------------------------------------
# Weights reflect a defensive-security context: risk reduction matters most; asset
# protection and evidence alignment come next; proportionality discourages overreaction;
# human control and operational continuity keep recommendations safe and low-disruption.

DEFAULT_UTILITY_PROFILE = UtilityProfile(
    profile_id="UP-01",
    name="Defensive security triage utility profile",
    description="Weighted criteria for comparing defensive triage recommendations.",
    criteria=(
        UtilityCriterion(criterion_id="UC-01", name=CriterionName.RISK_REDUCTION, weight=0.25,
                         description="How well the action reduces the risk the situation presents."),
        UtilityCriterion(criterion_id="UC-02", name=CriterionName.ASSET_PROTECTION, weight=0.20,
                         description="How well the action protects the affected account or service."),
        UtilityCriterion(criterion_id="UC-03", name=CriterionName.EVIDENCE_ALIGNMENT, weight=0.20,
                         description="How well the action fits the available evidence (no unsupported escalation)."),
        UtilityCriterion(criterion_id="UC-04", name=CriterionName.PROPORTIONALITY, weight=0.15,
                         description="How well the response size matches what the situation calls for."),
        UtilityCriterion(criterion_id="UC-05", name=CriterionName.HUMAN_CONTROL, weight=0.10,
                         description="How strongly humans stay in control of consequential decisions."),
        UtilityCriterion(criterion_id="UC-06", name=CriterionName.OPERATIONAL_CONTINUITY, weight=0.10,
                         description="How little the action disrupts normal operations and analyst effort."),
    ),
)

# ---------------------------------------------------------------------------
# Candidate actions with their documented capability assumptions (0.0 - 1.0)
# ---------------------------------------------------------------------------
# intensity doubles as disruption/effort: operational continuity is 1 - intensity.

CANDIDATE_ACTIONS: tuple[UtilityAction, ...] = (
    UtilityAction(
        action_id="UB-ACT-MONITOR", decision="MONITOR",
        description="Keep passively watching for recurrence or new related alerts.",
        risk_handling=0.10, asset_protection=0.10, evidence_development=0.05, intensity=0.05, human_control=0.40,
        plan_steps=(
            UtilityStep(text="Continue monitoring for recurrence or new correlated alerts."),
            UtilityStep(text="Reassess if stronger evidence appears."),
        ),
    ),
    UtilityAction(
        action_id="UB-ACT-COLLECT", decision="COLLECT_MORE_EVIDENCE",
        description="Gather additional evidence before deciding on a stronger response.",
        risk_handling=0.30, asset_protection=0.30, evidence_development=0.85, intensity=0.25, human_control=0.60,
        plan_steps=(
            UtilityStep(text="Validate alert completeness."),
            UtilityStep(text="Gather limited supporting telemetry."),
            UtilityStep(text="Monitor for recurrence."),
            UtilityStep(text="Reassess if evidence strengthens."),
        ),
    ),
    UtilityAction(
        action_id="UB-ACT-INVESTIGATE", decision="INVESTIGATE",
        description="Open a human-led security investigation of the alert.",
        risk_handling=0.75, asset_protection=0.65, evidence_development=0.80, intensity=0.50, human_control=0.80,
        requires_human_authorization=True,
        plan_steps=(
            UtilityStep(text="Review detailed authentication events for the targeted account.",
                        only_if="failed_logins_present"),
            UtilityStep(text="Validate whether the source is expected.", only_if="source_not_known"),
            UtilityStep(text="Investigate related account and asset activity."),
            UtilityStep(text="Escalate if stronger evidence or confirmed impact appears."),
        ),
    ),
    UtilityAction(
        action_id="UB-ACT-ESCALATE", decision="ESCALATE_TO_SECURITY_ANALYST",
        description="Escalate to a security analyst or SOC for human review.",
        risk_handling=0.90, asset_protection=0.90, evidence_development=0.70, intensity=0.85, human_control=0.95,
        requires_human_authorization=True,
        plan_steps=(
            UtilityStep(text="Preserve and review correlated alert evidence."),
            UtilityStep(text="Validate unusual access against expected behavior of the affected account or service."),
            UtilityStep(text="Escalate for authorized SOC or security-analyst investigation."),
            UtilityStep(text="Request separately authorized containment review only if later evidence justifies it."),
        ),
    ),
)

# ---------------------------------------------------------------------------
# Assumption 1: reading the scenario as needs (all additive, all documented)
# ---------------------------------------------------------------------------
# risk_need = base + severity + privileged-account pattern + critical-service pattern
#             + confirmed impact + unusual access + any correlation   (clamped to 1.0)

BASE_RISK_NEED = 0.10
SEVERITY_RISK = {Severity.LOW: 0.0, Severity.MEDIUM: 0.10, Severity.HIGH: 0.25, Severity.CRITICAL: 0.40}
PRIVILEGED_PATTERN_RISK = 0.30  # privileged account + unfamiliar source + repeated failed logins
CRITICAL_PATTERN_RISK = 0.20  # critical asset + unusual access + multiple correlated alerts
CONFIRMED_IMPACT_RISK = 0.50
UNUSUAL_ACCESS_RISK = 0.10
CORRELATION_RISK = 0.05

ASSET_STAKE = {
    AssetCriticality.LOW: 0.10, AssetCriticality.MEDIUM: 0.35,
    AssetCriticality.HIGH: 0.60, AssetCriticality.CRITICAL: 0.90,
}
EVIDENCE_SUPPORT = {
    EvidenceStrength.LIMITED: 0.20, EvidenceStrength.CONFLICTING: 0.40,
    EvidenceStrength.MODERATE: 0.60, EvidenceStrength.STRONG: 0.85,
}
CONFIRMED_EVIDENCE_FLOOR = 0.90  # confirmed impact implies strong support for acting

# required_intensity = share of risk need + share of asset stake + critical-pattern bonus
INTENSITY_FROM_RISK = 0.30
INTENSITY_FROM_STAKE = 0.55
INTENSITY_FROM_CRITICAL_PATTERN = 0.15

# ---------------------------------------------------------------------------
# Assumption 2: outcome probabilities
# ---------------------------------------------------------------------------
# "under" = how far the action's intensity falls short of required_intensity;
# "over"  = how far it exceeds it. Both lower P(FAVORABLE) and raise P(UNFAVORABLE);
# over-response is penalized slightly more because it burdens analysts and risks
# unsupported escalation. P(PARTIAL) is whatever probability remains.

FAVORABLE_MAX = 0.80
FAVORABLE_MIN = 0.10
UNDER_RESPONSE_FAVORABLE_PENALTY = 1.1
OVER_RESPONSE_FAVORABLE_PENALTY = 1.0
UNFAVORABLE_BASE = 0.05
UNDER_RESPONSE_UNFAVORABLE_INCREASE = 0.55
OVER_RESPONSE_UNFAVORABLE_INCREASE = 0.40
UNFAVORABLE_MAX = 0.60

# ---------------------------------------------------------------------------
# Assumption 3: how much of an action's capability is realised in each outcome
# ---------------------------------------------------------------------------

EFFECTIVENESS = {
    OutcomeCategory.FAVORABLE: 1.00,
    OutcomeCategory.PARTIAL: 0.55,
    OutcomeCategory.UNFAVORABLE: 0.15,
}
OUTCOME_DESCRIPTIONS = {
    OutcomeCategory.FAVORABLE: "The recommendation achieves its intended effect and is accepted by analysts.",
    OutcomeCategory.PARTIAL: "The recommendation achieves part of its intended effect; some risk or effort remains.",
    OutcomeCategory.UNFAVORABLE: "The recommendation misses what the situation needs (too weak or too heavy).",
}
OVERREACH_EVIDENCE_PENALTY = 0.6  # evidence alignment shrinks when intensity exceeds evidence support
UNDER_RESPONSE_PROPORTIONALITY_PENALTY = 1.0
OVER_RESPONSE_PROPORTIONALITY_PENALTY = 1.2

_PRIORITY_BY_DECISION = {
    "ESCALATE_TO_SECURITY_ANALYST": Priority.CRITICAL,
    "INVESTIGATE": Priority.HIGH,
    "MONITOR": Priority.LOW,
}
LOW_RISK_NEED = 0.5  # evidence collection below this risk need is LOW priority, otherwise MEDIUM


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class UtilityBasedAgent(BaseAgent):
    """Holds only its immutable utility profile; no history, so calls are independent."""

    name = AGENT_NAME

    def __init__(self, profile: UtilityProfile = DEFAULT_UTILITY_PROFILE) -> None:
        self._profile = profile

    def decide(self, scenario: SecurityScenario) -> AgentDecision:
        context = self._build_decision_context(scenario)
        profile = self._define_utility_profile()
        candidates = self._generate_candidate_actions()
        evaluated = tuple(self._evaluate_action(context, profile, action) for action in candidates)
        ranked, ranking = self._rank_actions(evaluated)
        selected = self._select_best_action(ranked, ranking)
        analysis = UtilityDecisionAnalysis(
            profile=profile,
            context=context,
            actions=ranked,
            ranking=ranking,
            selected_action_id=selected.action_id,
            priority=self._priority_for(context, selected),
        )
        return self._build_decision(scenario, candidates, analysis)

    # 1 -----------------------------------------------------------------------
    @staticmethod
    def _build_decision_context(scenario: SecurityScenario) -> DecisionContext:
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
        steps: list[tuple[str, float]] = [("base risk need", BASE_RISK_NEED)]
        steps.append((f"severity {scenario.severity.value}", SEVERITY_RISK[scenario.severity]))
        if privileged:
            steps.append(("privileged account, unfamiliar source, repeated failed logins", PRIVILEGED_PATTERN_RISK))
        if critical:
            steps.append(("critical asset with unusual access and correlated alerts", CRITICAL_PATTERN_RISK))
        if scenario.confirmed_impact:
            steps.append(("confirmed impact", CONFIRMED_IMPACT_RISK))
        if scenario.unusual_access:
            steps.append(("unusual access", UNUSUAL_ACCESS_RISK))
        if scenario.correlated_alert_count >= 1:
            steps.append(("correlated alerts present", CORRELATION_RISK))
        risk_need = _clamp(math.fsum(delta for _, delta in steps))

        stake = ASSET_STAKE[scenario.asset_criticality]
        support = EVIDENCE_SUPPORT[scenario.evidence_strength]
        if scenario.confirmed_impact:
            support = max(support, CONFIRMED_EVIDENCE_FLOOR)
        required = _clamp(
            INTENSITY_FROM_RISK * risk_need
            + INTENSITY_FROM_STAKE * stake
            + INTENSITY_FROM_CRITICAL_PATTERN * critical
        )
        adjustments = tuple(f"risk need +{delta:.2f}: {why}" for why, delta in steps if delta) + (
            f"asset stake {stake:.2f}: asset criticality {scenario.asset_criticality.value}",
            f"evidence support {support:.2f}: evidence strength {scenario.evidence_strength.value}",
        )
        return DecisionContext(
            scenario_id=scenario.scenario_id,
            risk_need=round(risk_need, SCORE_DECIMALS),
            asset_stake=stake,
            evidence_support=support,
            required_intensity=round(required, SCORE_DECIMALS),
            confirmed_impact=scenario.confirmed_impact,
            failed_logins_present=scenario.failed_login_count > 0,
            source_not_known=scenario.source_familiarity != SourceFamiliarity.KNOWN,
            privileged_account_pattern=privileged,
            critical_service_pattern=critical,
            adjustments_applied=adjustments,
        )

    # 2 -----------------------------------------------------------------------
    def _define_utility_profile(self) -> UtilityProfile:
        return self._profile

    # 3 -----------------------------------------------------------------------
    @staticmethod
    def _generate_candidate_actions() -> tuple[UtilityAction, ...]:
        """The same four candidates for every scenario."""
        return CANDIDATE_ACTIONS

    # 4 -----------------------------------------------------------------------
    @staticmethod
    def _check_action_eligibility(context: DecisionContext, action: UtilityAction) -> tuple[bool, tuple[str, ...]]:
        """Safety and policy constraints. They are checked before, and never traded off against, utility."""
        violations = [
            effect
            for effect in action.side_effects
            # Describing impact the scenario itself confirms is not an unsupported claim.
            if not (effect is ProhibitedCondition.UNSUPPORTED_COMPROMISE_CLAIM and context.confirmed_impact)
        ]
        if action.is_consequential and not action.requires_human_authorization:
            violations.append(ProhibitedCondition.CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION)
        if violations:
            return False, tuple(f"Prohibited: would bring about {v.value}." for v in dict.fromkeys(violations))
        return True, ("All safety and policy constraints are satisfied.",)

    # 5 -----------------------------------------------------------------------
    @classmethod
    def _generate_possible_outcomes(cls, context: DecisionContext, action: UtilityAction) -> tuple[ProbabilisticOutcome, ...]:
        probabilities = cls._estimate_outcome_probabilities(context, action)
        return tuple(
            ProbabilisticOutcome(
                outcome_id=f"{action.action_id}-{category.value}",
                action_id=action.action_id,
                probability=probabilities[category],
                outcome_state=cls._score_outcome_criteria(context, action, category),
            )
            for category in OutcomeCategory
        )

    @staticmethod
    def _estimate_outcome_probabilities(context: DecisionContext, action: UtilityAction) -> dict[OutcomeCategory, float]:
        under = max(0.0, context.required_intensity - action.intensity)
        over = max(0.0, action.intensity - context.required_intensity)
        favorable = min(
            FAVORABLE_MAX,
            max(
                FAVORABLE_MIN,
                FAVORABLE_MAX
                - UNDER_RESPONSE_FAVORABLE_PENALTY * under
                - OVER_RESPONSE_FAVORABLE_PENALTY * over,
            ),
        )
        unfavorable = min(
            UNFAVORABLE_MAX,
            UNFAVORABLE_BASE
            + UNDER_RESPONSE_UNFAVORABLE_INCREASE * under
            + OVER_RESPONSE_UNFAVORABLE_INCREASE * over,
        )
        favorable = round(favorable, PROBABILITY_DECIMALS)
        unfavorable = round(unfavorable, PROBABILITY_DECIMALS)
        return {
            OutcomeCategory.FAVORABLE: favorable,
            OutcomeCategory.PARTIAL: round(1.0 - favorable - unfavorable, PROBABILITY_DECIMALS),
            OutcomeCategory.UNFAVORABLE: unfavorable,
        }

    @staticmethod
    def _score_outcome_criteria(context: DecisionContext, action: UtilityAction, category: OutcomeCategory) -> OutcomeState:
        """Criterion scores: 0.0 = worst, 1.0 = best. 'Need' terms make a score depend on the scenario."""
        realised = EFFECTIVENESS[category]
        under = max(0.0, context.required_intensity - action.intensity)
        over = max(0.0, action.intensity - context.required_intensity)
        evidence_gap = 1.0 - context.evidence_support
        overreach = max(0.0, action.intensity - context.evidence_support)

        # Residual risk/exposure shrinks as the realised capability meets the need.
        risk_reduction = 1.0 - context.risk_need * (1.0 - action.risk_handling * realised)
        asset_protection = 1.0 - context.asset_stake * (1.0 - action.asset_protection * realised)
        # Close the evidence gap, but lose alignment if acting harder than the evidence supports.
        evidence_alignment = (1.0 - evidence_gap * (1.0 - action.evidence_development * realised)) * (
            1.0 - OVERREACH_EVIDENCE_PENALTY * overreach
        )
        proportionality = 1.0 - (
            UNDER_RESPONSE_PROPORTIONALITY_PENALTY * under + OVER_RESPONSE_PROPORTIONALITY_PENALTY * over
        )
        return OutcomeState(
            outcome_id=f"{action.action_id}-{category.value}",
            name=category,
            description=f"{action.decision}: {OUTCOME_DESCRIPTIONS[category]}",
            risk_reduction=round(_clamp(risk_reduction), SCORE_DECIMALS),
            asset_protection=round(_clamp(asset_protection), SCORE_DECIMALS),
            evidence_alignment=round(_clamp(evidence_alignment), SCORE_DECIMALS),
            proportionality=round(_clamp(proportionality), SCORE_DECIMALS),
            human_control=action.human_control,
            operational_continuity=round(1.0 - action.intensity, SCORE_DECIMALS),
            limitations=("Simulated assumption; not a statistical estimate.",),
        )

    # 6 -----------------------------------------------------------------------
    @staticmethod
    def _calculate_outcome_utility(profile: UtilityProfile, state: OutcomeState) -> OutcomeUtilityBreakdown:
        parts = tuple(
            CriterionContribution(
                criterion=criterion.name,
                raw_score=state.score(criterion.name),
                weight=criterion.weight,
                weighted_contribution=state.score(criterion.name) * criterion.weight,
            )
            for criterion in profile.criteria
        )
        # Rounding removes floating-point noise so the utility stays inside 0.0 - 1.0.
        utility = round(math.fsum(p.weighted_contribution for p in parts), 12)
        return OutcomeUtilityBreakdown(outcome_id=state.outcome_id, contributions=parts, outcome_utility=utility)

    # 7 -----------------------------------------------------------------------
    @staticmethod
    def _calculate_expected_utility(
        action: UtilityAction,
        eligibility: tuple[bool, tuple[str, ...]],
        outcomes: tuple[ProbabilisticOutcome, ...],
        utilities: tuple[OutcomeUtilityBreakdown, ...],
    ) -> ActionExpectedUtility:
        expected = round(
            math.fsum(o.probability * u.outcome_utility for o, u in zip(outcomes, utilities, strict=True)), 12
        )
        return ActionExpectedUtility(
            action_id=action.action_id,
            decision=action.decision,
            eligible=eligibility[0],
            eligibility_reasons=eligibility[1],
            outcomes=outcomes,
            outcome_utilities=utilities,
            expected_utility=expected,
        )

    @classmethod
    def _evaluate_action(cls, context: DecisionContext, profile: UtilityProfile, action: UtilityAction) -> ActionExpectedUtility:
        outcomes = cls._generate_possible_outcomes(context, action)
        utilities = tuple(cls._calculate_outcome_utility(profile, o.outcome_state) for o in outcomes)
        return cls._calculate_expected_utility(action, cls._check_action_eligibility(context, action), outcomes, utilities)

    # 8 -----------------------------------------------------------------------
    @staticmethod
    def _rank_actions(
        evaluated: Sequence[ActionExpectedUtility],
    ) -> tuple[tuple[ActionExpectedUtility, ...], UtilityRanking]:
        """Rank eligible actions by expected utility; ties within TOLERANCE use TIE_BREAKER_ORDER."""
        eligible = [(index, a) for index, a in enumerate(evaluated) if a.eligible]
        # Stable sort on (-EU, candidate index) keeps the order independent of dict/set iteration.
        by_utility = sorted(eligible, key=lambda item: (-item[1].expected_utility, item[0]))

        groups: list[list[tuple[int, ActionExpectedUtility]]] = []
        for item in by_utility:
            if groups and groups[-1][0][1].expected_utility - item[1].expected_utility <= TOLERANCE:
                groups[-1].append(item)
            else:
                groups.append([item])

        def tie_key(item: tuple[int, ActionExpectedUtility]) -> tuple[float, float, float, int]:
            index, action = item
            return (
                -round(action.expected_score(CriterionName.OPERATIONAL_CONTINUITY), TIE_DECIMALS),
                -round(action.expected_score(CriterionName.EVIDENCE_ALIGNMENT), TIE_DECIMALS),
                -round(action.expected_score(CriterionName.HUMAN_CONTROL), TIE_DECIMALS),
                index,
            )

        ordered_groups = [sorted(group, key=tie_key) for group in groups]
        ordered = [item for group in ordered_groups for item in group]
        rank_of = {action.action_id: rank for rank, (_, action) in enumerate(ordered, start=1)}
        ranked = tuple(
            action.model_copy(update={"rank": rank_of[action.action_id]}) if action.eligible else action
            for action in evaluated
        )
        ranking = UtilityRanking(
            ranked_action_ids=tuple(action.action_id for _, action in ordered),
            tie_breaker_used=bool(ordered_groups) and len(ordered_groups[0]) > 1,
            tie_groups=tuple(tuple(a.action_id for _, a in g) for g in ordered_groups if len(g) > 1),
            tie_breaker_order=TIE_BREAKER_ORDER,
        )
        return ranked, ranking

    @staticmethod
    def _select_best_action(ranked: Sequence[ActionExpectedUtility], ranking: UtilityRanking) -> ActionExpectedUtility:
        if not ranking.ranked_action_ids:
            raise NoEligibleActionError("Every candidate action violated a safety or policy constraint.")
        best_id = ranking.ranked_action_ids[0]
        return next(a for a in ranked if a.action_id == best_id)

    @staticmethod
    def _priority_for(context: DecisionContext, selected: ActionExpectedUtility) -> Priority:
        if selected.decision == "COLLECT_MORE_EVIDENCE":
            return Priority.LOW if context.risk_need < LOW_RISK_NEED else Priority.MEDIUM
        return _PRIORITY_BY_DECISION[selected.decision]

    # 9 -----------------------------------------------------------------------
    @classmethod
    def _build_decision(
        cls, scenario: SecurityScenario, candidates: tuple[UtilityAction, ...], analysis: UtilityDecisionAnalysis
    ) -> AgentDecision:
        selected = analysis.selected
        context, profile = analysis.context, analysis.profile
        runner_up = cls._runner_up(analysis)
        trade_off = cls._key_trade_off(profile, selected, runner_up)
        action = next(c for c in candidates if c.action_id == selected.action_id)

        return AgentDecision(
            agent_name=AGENT_NAME,
            agent_type=AgentType.UTILITY_BASED,
            priority=analysis.priority,
            decision=selected.decision,
            reasoning=cls._reasoning(analysis, trade_off),
            evidence=scenario_evidence(scenario),
            uncertainty=uncertainty_statement(scenario),
            likely_interpretation=cls._interpretation(context),
            recommended_actions=[
                step.text
                for step in action.plan_steps
                if step.only_if is None or getattr(context, step.only_if)
            ],
            audit_note=(
                f"Scenario {scenario.scenario_id} | Agent: {AGENT_NAME} | Profile: {profile.profile_id} | "
                f"Decision: {selected.decision} | Expected utility: {selected.expected_utility:.3f} | "
                f"{NO_ACTION_STATEMENT} {HUMAN_AUTHORIZATION_STATEMENT}"
            ),
            utility_score=selected.expected_utility,
            metadata={
                "decision_mechanism": DECISION_MECHANISM,
                "utility_profile": {
                    "profile_id": profile.profile_id,
                    "name": profile.name,
                    "criteria": [
                        {"criterion_id": c.criterion_id, "name": c.name.value, "weight": c.weight}
                        for c in profile.criteria
                    ],
                    "weight_total": profile.weight_total,
                },
                "decision_context": context.model_dump(mode="json"),
                "candidate_actions_considered": [a.decision for a in analysis.actions],
                "expected_utility_formula": EXPECTED_UTILITY_FORMULA,
                "outcome_utility_formula": OUTCOME_UTILITY_FORMULA,
                "action_analyses": [cls._action_metadata(profile, a) for a in analysis.actions],
                "action_ranking": [
                    {"rank": a.rank, "decision": a.decision, "expected_utility": a.expected_utility}
                    for a in sorted((a for a in analysis.actions if a.eligible), key=lambda a: a.rank or 0)
                ],
                "ineligible_actions": [
                    {"decision": a.decision, "reasons": list(a.eligibility_reasons)}
                    for a in analysis.actions
                    if not a.eligible
                ],
                "selected_action": selected.decision,
                "selected_action_id": selected.action_id,
                "selected_expected_utility": selected.expected_utility,
                "tie_breaker_used": analysis.ranking.tie_breaker_used,
                "tie_breaker_order": list(analysis.ranking.tie_breaker_order),
                "tolerance": TOLERANCE,
                "key_trade_off": trade_off,
                "scenario_id": scenario.scenario_id,
                "assumptions_note": SIMULATION_NOTE,
                "uses_goal_satisfaction_selection": False,
                "uses_internal_history": False,
                "uses_learning": False,
                "uses_llm": False,
                **safety_metadata(),
            },
        )

    @staticmethod
    def _runner_up(analysis: UtilityDecisionAnalysis) -> ActionExpectedUtility | None:
        ids = analysis.ranking.ranked_action_ids
        return next((a for a in analysis.actions if len(ids) > 1 and a.action_id == ids[1]), None)

    @staticmethod
    def _weighted_expected_scores(profile: UtilityProfile, action: ActionExpectedUtility) -> dict[CriterionName, float]:
        """Each criterion's share of the action's expected utility."""
        return {c.name: c.weight * action.expected_score(c.name) for c in profile.criteria}

    @classmethod
    def _key_trade_off(
        cls, profile: UtilityProfile, selected: ActionExpectedUtility, runner_up: ActionExpectedUtility | None
    ) -> dict[str, object]:
        if runner_up is None:
            return {"selected": selected.decision, "runner_up": None, "advantages": [], "concessions": []}
        chosen = cls._weighted_expected_scores(profile, selected)
        other = cls._weighted_expected_scores(profile, runner_up)
        margins = {name: round(chosen[name] - other[name], SCORE_DECIMALS) for name in chosen}
        ordered = sorted(margins.items(), key=lambda item: (-item[1], item[0].value))
        return {
            "selected": selected.decision,
            "runner_up": runner_up.decision,
            "advantages": [{"criterion": n.value, "margin": m} for n, m in ordered if m > 0][:3],
            "concessions": [{"criterion": n.value, "margin": m} for n, m in reversed(ordered) if m < 0][:3],
        }

    @staticmethod
    def _action_metadata(profile: UtilityProfile, action: ActionExpectedUtility) -> dict[str, object]:
        outcomes = []
        for outcome, utility in zip(action.outcomes, action.outcome_utilities, strict=True):
            outcomes.append(
                {
                    "name": outcome.outcome_state.name.value,
                    "outcome_id": outcome.outcome_id,
                    "probability": outcome.probability,
                    "outcome_utility": utility.outcome_utility,
                    "contribution_to_expected_utility": outcome.probability * utility.outcome_utility,
                    "criterion_breakdown": [
                        {
                            "criterion": part.criterion.value,
                            "raw_score": part.raw_score,
                            "weight": part.weight,
                            "weighted_contribution": part.weighted_contribution,
                        }
                        for part in utility.contributions
                    ],
                }
            )
        return {
            "decision": action.decision,
            "action_id": action.action_id,
            "eligible": action.eligible,
            "eligibility_reasons": list(action.eligibility_reasons),
            "outcomes": outcomes,
            "expected_utility": action.expected_utility,
            "rank": action.rank,
        }

    @staticmethod
    def _reasoning(analysis: UtilityDecisionAnalysis, trade_off: dict[str, object]) -> str:
        profile, context, selected = analysis.profile, analysis.context, analysis.selected
        compared = ", ".join(a.decision for a in analysis.actions)
        utilities = "; ".join(
            f"{a.decision} {a.expected_utility:.3f}" if a.eligible else f"{a.decision} ineligible"
            for a in analysis.actions
        )
        parts = [
            f"The Utility-Based Agent compared four defensive actions ({compared}) using utility "
            f"profile {profile.profile_id}.",
            "Each action was evaluated across FAVORABLE, PARTIAL and UNFAVORABLE outcomes. "
            "Outcome probabilities are deterministic assumptions derived from scenario facts "
            f"(risk need {context.risk_need:.2f}, asset stake {context.asset_stake:.2f}, evidence "
            f"support {context.evidence_support:.2f}, required response intensity "
            f"{context.required_intensity:.2f}): an action far below or above the required intensity "
            "is less likely to have a favorable outcome.",
            "Outcome utility is the weighted sum of six criteria (" + ", ".join(
                f"{c.name.value} {c.weight:.2f}" for c in profile.criteria
            ) + "); expected utility is the sum of outcome probability times outcome utility.",
            f"Expected utilities: {utilities}.",
            f"{selected.decision} ranked first with expected utility {selected.expected_utility:.3f}.",
        ]
        advantages: list[dict[str, object]] = trade_off["advantages"]  # type: ignore[assignment]
        concessions: list[dict[str, object]] = trade_off["concessions"]  # type: ignore[assignment]
        if trade_off["runner_up"] and advantages:
            lead = ", ".join(str(item["criterion"]) for item in advantages)
            sentence = f"Against runner-up {trade_off['runner_up']}, it gained most on {lead}"
            if concessions:
                give = ", ".join(str(item["criterion"]) for item in concessions)
                sentence += f", while giving up value on {give}"
            parts.append(sentence + ".")
        impact = (
            "Impact is confirmed in the scenario."
            if context.confirmed_impact
            else "Compromise remains unconfirmed; the recommendation supports human investigation and review only."
        )
        parts.append(f"{impact} No action was executed. {SIMULATION_NOTE}")
        return " ".join(parts)

    @staticmethod
    def _interpretation(context: DecisionContext) -> str:
        findings = []
        if context.confirmed_impact:
            findings.append("Impact is confirmed in the scenario; analyst-led response is needed.")
        if context.privileged_account_pattern:
            findings.append("Possible credential-guessing attempt against a privileged account; unconfirmed.")
        if context.critical_service_pattern:
            findings.append(
                "Possible serious incident on a critical service, or an undocumented but legitimate change; unconfirmed."
            )
        if not findings:
            findings.append(
                "Significant alert whose cause is not yet understood."
                if context.risk_need >= 0.3
                else "Likely benign or one-off event; evidence is too weak to conclude."
            )
        return " ".join(findings)
