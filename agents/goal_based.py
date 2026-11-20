"""Goal-Based Agent: current state + explicit goal + candidate actions -> selected plan.

Decision cycle (one focused method per step):
  1. _build_current_state          interpret the scenario as a compact state
  2. _define_goal                  the explicit goal GB-G01
  3. _generate_candidate_actions   the same four defensive actions every time
  4. _predict_outcome              what each action is expected to achieve here
  5. _evaluate_goal_satisfaction   test each outcome against the goal's conditions
  6. _select_plan                  least disruptive candidate that satisfies the goal
  7. _build_decision               explainable AgentDecision

The agent asks "which action achieves the goal?" using qualitative checks
(adequacy, proportionality, safety). It never computes scores or utilities, keeps no
history, reads no files and calls no LLM, so identical input gives identical output.
"""

from agents.base import BaseAgent
from agents.common import (
    HUMAN_AUTHORIZATION_STATEMENT,
    MULTIPLE_CORRELATED_ALERTS_THRESHOLD,
    NO_ACTION_STATEMENT,
    REPEATED_FAILED_LOGIN_THRESHOLD,
    TriageDecision,
    safety_metadata,
    scenario_evidence,
    uncertainty_statement,
)
from models.goal_models import (
    ActionPlan,
    CandidateAction,
    CurrentState,
    GoalCondition,
    GoalDeliberation,
    GoalEvaluation,
    PlanStep,
    PredictedEffect,
    PredictedOutcome,
    ProhibitedCondition,
    RiskAttention,
    SecurityGoal,
    StateFact,
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

AGENT_NAME = "Goal-Based Agent"
DECISION_MECHANISM = "goal_satisfaction_and_plan_selection"
SELECTION_POLICY = "least_disruptive_goal_satisfying_action"


class GoalNotAchievableError(RuntimeError):
    """No candidate action satisfies the goal; never silently picks an unsafe fallback."""


# --- The explicit goal -------------------------------------------------------

SECURITY_GOAL = SecurityGoal(
    goal_id="GB-G01",
    name="Safe and proportionate security triage",
    description=(
        "Protect important accounts and services by recommending proportionate, "
        "evidence-based investigation or escalation while preserving uncertainty, avoiding "
        "unsupported conclusions, and requiring human authorization for consequential actions."
    ),
    required_conditions=tuple(GoalCondition),
    prohibited_conditions=tuple(ProhibitedCondition),
)


# --- Candidate actions, ordered from least to most disruptive ----------------
# The order is the selection policy's tie-breaker; it is an ordering, not a score.

CANDIDATE_ACTIONS: tuple[CandidateAction, ...] = (
    CandidateAction(
        action_id="GB-ACT-MONITOR",
        decision=TriageDecision.MONITOR.value,
        description="Keep passively watching for recurrence or new related alerts.",
        predicted_effects=(PredictedEffect.PASSIVE_WATCH,),
        plan_steps=(
            PlanStep(text="Continue monitoring for recurrence or new correlated alerts."),
            PlanStep(text="Reassess if stronger evidence appears."),
        ),
        limitations=("Does not reduce the evidence gap or investigate the risk.",),
    ),
    CandidateAction(
        action_id="GB-ACT-COLLECT",
        decision=TriageDecision.COLLECT_MORE_EVIDENCE.value,
        description="Gather additional evidence before deciding on a stronger response.",
        predicted_effects=(PredictedEffect.REDUCES_EVIDENCE_GAP,),
        plan_steps=(
            PlanStep(text="Validate alert completeness."),
            PlanStep(text="Gather limited supporting telemetry."),
            PlanStep(text="Monitor for recurrence or correlation."),
            PlanStep(text="Reassess if stronger evidence appears."),
        ),
        limitations=("Gathers evidence but does not by itself investigate an active risk.",),
    ),
    CandidateAction(
        action_id="GB-ACT-INVESTIGATE",
        decision=TriageDecision.INVESTIGATE.value,
        description="Open a human-led security investigation of the alert.",
        predicted_effects=(
            PredictedEffect.REDUCES_EVIDENCE_GAP,
            PredictedEffect.ACTIVE_INVESTIGATION,
        ),
        plan_steps=(
            PlanStep(text="Open a security investigation for the alert."),
            PlanStep(
                text="Review detailed authentication activity for the targeted account.",
                only_if=StateFact.FAILED_LOGINS_PRESENT,
            ),
            PlanStep(
                text="Validate whether the source is expected.",
                only_if=StateFact.SOURCE_NOT_KNOWN,
            ),
            PlanStep(text="Check correlated activity involving the affected account or asset."),
            PlanStep(
                text="Escalate to a security analyst if investigation finds stronger evidence "
                "or confirmed impact."
            ),
        ),
        requires_human_authorization=True,
        limitations=("Does not by itself bring analyst review to a critical service.",),
    ),
    CandidateAction(
        action_id="GB-ACT-ESCALATE",
        decision=TriageDecision.ESCALATE_TO_SECURITY_ANALYST.value,
        description="Escalate to a security analyst or SOC for human review.",
        preconditions=(StateFact.ACTIVE_RISK_INDICATED,),
        predicted_effects=(
            PredictedEffect.REDUCES_EVIDENCE_GAP,
            PredictedEffect.ACTIVE_INVESTIGATION,
            PredictedEffect.ANALYST_REVIEW,
        ),
        plan_steps=(
            PlanStep(text="Preserve and review correlated alert evidence."),
            PlanStep(text="Validate unusual access against expected behavior of the affected account or service."),
            PlanStep(text="Escalate to a security analyst or SOC for authorized human investigation."),
            PlanStep(text="Request separately authorized containment review only if later evidence justifies it."),
        ),
        requires_human_authorization=True,
        limitations=("Uses analyst time; justified only when an active risk is indicated.",),
    ),
)

_PRIORITY_BY_DECISION = {
    TriageDecision.ESCALATE_TO_SECURITY_ANALYST.value: Priority.CRITICAL,
    TriageDecision.INVESTIGATE.value: Priority.HIGH,
    TriageDecision.MONITOR.value: Priority.LOW,
}

_FACT_JUSTIFICATION = {
    StateFact.ACTIVE_RISK_INDICATED: (
        "no active risk indicator (privileged-account, critical-service, high-severity or "
        "confirmed-impact) justifies this level of response on the current evidence"
    ),
}


class GoalBasedAgent(BaseAgent):
    """Stateless: every call deliberates from scratch using only the given scenario."""

    name = AGENT_NAME

    def decide(self, scenario: SecurityScenario) -> AgentDecision:
        state = self._build_current_state(scenario)
        goal = self._define_goal()
        candidates = self._generate_candidate_actions()
        outcomes = tuple(self._predict_outcome(state, action) for action in candidates)
        evaluations = tuple(
            self._evaluate_goal_satisfaction(goal, state, action, outcome)
            for action, outcome in zip(candidates, outcomes, strict=True)
        )
        plan = self._select_plan(goal, state, candidates, outcomes, evaluations)
        deliberation = GoalDeliberation(
            goal=goal,
            state=state,
            candidates=candidates,
            outcomes=outcomes,
            evaluations=evaluations,
            plan=plan,
            priority=self._priority_for(state, plan),
        )
        return self._build_decision(scenario, deliberation)

    # 1 -----------------------------------------------------------------------
    @staticmethod
    def _build_current_state(scenario: SecurityScenario) -> CurrentState:
        privileged_risk = (
            scenario.account_privilege == AccountPrivilege.PRIVILEGED
            and scenario.source_familiarity == SourceFamiliarity.UNFAMILIAR
            and scenario.failed_login_count >= REPEATED_FAILED_LOGIN_THRESHOLD
        )
        critical_risk = (
            scenario.asset_criticality == AssetCriticality.CRITICAL
            and scenario.unusual_access
            and scenario.correlated_alert_count >= MULTIPLE_CORRELATED_ALERTS_THRESHOLD
        )
        high_severity = scenario.severity in (Severity.HIGH, Severity.CRITICAL)
        confirmed = scenario.confirmed_impact
        return CurrentState(
            scenario_id=scenario.scenario_id,
            target_account=scenario.target_account,
            asset_name=scenario.asset_name,
            privileged_account_at_risk=privileged_risk,
            critical_asset_at_risk=critical_risk,
            high_severity=high_severity,
            weak_evidence=scenario.evidence_strength == EvidenceStrength.LIMITED,
            impact_confirmed=confirmed,
            evidence_gap_exists=not confirmed or scenario.evidence_strength != EvidenceStrength.STRONG,
            failed_logins_present=scenario.failed_login_count > 0,
            source_not_known=scenario.source_familiarity != SourceFamiliarity.KNOWN,
            active_risk_indicated=privileged_risk or critical_risk or high_severity or confirmed,
            analyst_review_needed=critical_risk or confirmed,
        )

    # 2 -----------------------------------------------------------------------
    @staticmethod
    def _define_goal() -> SecurityGoal:
        return SECURITY_GOAL

    # 3 -----------------------------------------------------------------------
    @staticmethod
    def _generate_candidate_actions() -> tuple[CandidateAction, ...]:
        """Same four candidates for every scenario, least disruptive first."""
        return CANDIDATE_ACTIONS

    # 4 -----------------------------------------------------------------------
    @classmethod
    def _predict_outcome(cls, state: CurrentState, action: CandidateAction) -> PredictedOutcome:
        effects = set(action.predicted_effects)
        attention = cls._attention_level(effects)
        analyst_review = PredictedEffect.ANALYST_REVIEW in effects
        investigates = attention in (RiskAttention.ACTIVE_INVESTIGATION, RiskAttention.ANALYST_REVIEW)
        prohibited = cls._prohibited_effects(state, action)
        claim_avoided = ProhibitedCondition.UNSUPPORTED_COMPROMISE_CLAIM not in prohibited
        evidence_gap_addressed = not state.evidence_gap_exists or PredictedEffect.REDUCES_EVIDENCE_GAP in effects
        return PredictedOutcome(
            action_id=action.action_id,
            decision=action.decision,
            predicted_state=(
                f"If {action.decision} is recommended, the risk receives {attention.value} attention; "
                f"the evidence gap is {'addressed' if evidence_gap_addressed else 'left open'}; "
                f"human review is {'requested' if analyst_review else 'not requested'}; "
                f"compromise remains {'confirmed per the scenario' if state.impact_confirmed else 'unconfirmed'}."
            ),
            evidence_gap_addressed=evidence_gap_addressed,
            risk_attention_level=attention,
            active_risk_addressed=not state.active_risk_indicated or investigates,
            critical_asset_protected=not state.analyst_review_needed or analyst_review,
            preserves_uncertainty=claim_avoided,
            requires_human_review=analyst_review,
            unsafe_automation_avoided=not any(
                p is not ProhibitedCondition.UNSUPPORTED_COMPROMISE_CLAIM for p in prohibited
            ),
            unsupported_claim_avoided=claim_avoided,
            limitations=action.limitations,
        )

    @staticmethod
    def _attention_level(effects: set[PredictedEffect]) -> RiskAttention:
        if PredictedEffect.ANALYST_REVIEW in effects:
            return RiskAttention.ANALYST_REVIEW
        if PredictedEffect.ACTIVE_INVESTIGATION in effects:
            return RiskAttention.ACTIVE_INVESTIGATION
        if PredictedEffect.REDUCES_EVIDENCE_GAP in effects:
            return RiskAttention.EVIDENCE_REVIEW
        return RiskAttention.WATCHING

    @staticmethod
    def _prohibited_effects(state: CurrentState, action: CandidateAction) -> tuple[ProhibitedCondition, ...]:
        """Prohibited conditions the action would bring about in this state."""
        effects = [
            effect
            for effect in action.side_effects
            # Describing impact the scenario itself confirms is not an unsupported claim.
            if not (effect is ProhibitedCondition.UNSUPPORTED_COMPROMISE_CLAIM and state.impact_confirmed)
        ]
        if action.is_consequential and not action.requires_human_authorization:
            effects.append(ProhibitedCondition.CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION)
        return tuple(dict.fromkeys(effects))

    # 5 -----------------------------------------------------------------------
    @classmethod
    def _evaluate_goal_satisfaction(
        cls, goal: SecurityGoal, state: CurrentState, action: CandidateAction, outcome: PredictedOutcome
    ) -> GoalEvaluation:
        satisfied: list[GoalCondition] = []
        unsatisfied: list[GoalCondition] = []
        reasons: list[str] = []
        for condition in goal.required_conditions:
            if cls._condition_holds(condition, state, action, outcome):
                satisfied.append(condition)
            else:
                unsatisfied.append(condition)
                reasons.append(cls._rejection_reason(condition, state, action))

        violated = tuple(p for p in cls._prohibited_effects(state, action) if p in goal.prohibited_conditions)
        reasons.extend(f"Prohibited by goal {goal.goal_id}: would bring about {p.value}." for p in violated)

        return GoalEvaluation(
            action_id=action.action_id,
            goal_id=goal.goal_id,
            satisfies_goal=not unsatisfied and not violated,
            satisfied_conditions=tuple(satisfied),
            unsatisfied_conditions=tuple(unsatisfied),
            violated_prohibited_conditions=violated,
            rejection_reasons=tuple(reasons),
        )

    @staticmethod
    def _condition_holds(
        condition: GoalCondition, state: CurrentState, action: CandidateAction, outcome: PredictedOutcome
    ) -> bool:
        match condition:
            case GoalCondition.EVIDENCE_GAP_ADDRESSED:
                return outcome.evidence_gap_addressed
            case GoalCondition.ACTIVE_RISK_ADDRESSED:
                return outcome.active_risk_addressed
            case GoalCondition.CRITICAL_ASSET_PROTECTED:
                return outcome.critical_asset_protected
            case GoalCondition.PROPORTIONATE_RESPONSE:
                return all(state.holds(fact) for fact in action.preconditions)
            case GoalCondition.UNCERTAINTY_PRESERVED:
                return outcome.preserves_uncertainty and outcome.unsupported_claim_avoided
            case GoalCondition.HUMAN_CONTROL_PRESERVED:
                return outcome.unsafe_automation_avoided
        raise ValueError(f"No check implemented for goal condition {condition}")

    @classmethod
    def _rejection_reason(cls, condition: GoalCondition, state: CurrentState, action: CandidateAction) -> str:
        decision = action.decision
        match condition:
            case GoalCondition.EVIDENCE_GAP_ADDRESSED:
                return f"Inadequate: {decision} does not reduce the current evidence gap."
            case GoalCondition.ACTIVE_RISK_ADDRESSED:
                return (
                    f"Inadequate for {cls._describe_risks(state)}: {decision} does not "
                    "actively investigate the risk."
                )
            case GoalCondition.CRITICAL_ASSET_PROTECTED:
                subject = (
                    "a critical service with correlated unusual activity"
                    if state.critical_asset_at_risk
                    else "confirmed impact"
                )
                return f"Inadequate for {subject}: analyst review is needed and {decision} does not provide it."
            case GoalCondition.PROPORTIONATE_RESPONSE:
                missing = [f for f in action.preconditions if not state.holds(f)]
                why = "; ".join(_FACT_JUSTIFICATION[f] for f in missing)
                return f"Disproportionate: {why}."
            case GoalCondition.UNCERTAINTY_PRESERVED:
                return f"Overstates impact: {decision} would assert compromise the scenario does not confirm."
            case GoalCondition.HUMAN_CONTROL_PRESERVED:
                return f"Unsafe: {decision} would take consequential action without human authorization."
        raise ValueError(f"No rejection reason defined for goal condition {condition}")

    @staticmethod
    def _describe_risks(state: CurrentState) -> str:
        risks = []
        if state.privileged_account_at_risk:
            risks.append("repeated failed logins from an unfamiliar source against a privileged account")
        if state.critical_asset_at_risk:
            risks.append("unusual correlated activity on a critical service")
        if state.impact_confirmed:
            risks.append("confirmed impact")
        if state.high_severity:
            risks.append("a high-severity alert")
        return "; ".join(risks) if risks else "the current alert"

    # 6 -----------------------------------------------------------------------
    @classmethod
    def _select_plan(
        cls,
        goal: SecurityGoal,
        state: CurrentState,
        candidates: tuple[CandidateAction, ...],
        outcomes: tuple[PredictedOutcome, ...],
        evaluations: tuple[GoalEvaluation, ...],
    ) -> ActionPlan:
        acceptable = [
            (action, outcome, evaluation)
            for action, outcome, evaluation in zip(candidates, outcomes, evaluations, strict=True)
            if evaluation.satisfies_goal
        ]
        if not acceptable:
            raise GoalNotAchievableError(
                f"No candidate action satisfies goal {goal.goal_id} for scenario {state.scenario_id}"
            )
        # Candidates are ordered least disruptive first, so the first acceptable one wins.
        action, outcome, evaluation = acceptable[0]
        return ActionPlan(
            plan_id=f"GB-PLAN-{state.scenario_id}-{action.decision}",
            goal_id=goal.goal_id,
            selected_action_id=action.action_id,
            primary_decision=action.decision,
            steps=cls._plan_steps(state, action),
            selection_reason=(
                f"{action.decision} is the least disruptive candidate that satisfies goal "
                f"{goal.goal_id}: it meets every required condition and brings about no "
                "prohibited condition."
            ),
            predicted_outcome=outcome,
            goal_evaluation=evaluation,
        )

    @staticmethod
    def _plan_steps(state: CurrentState, action: CandidateAction) -> tuple[str, ...]:
        return tuple(
            step.text for step in action.plan_steps if step.only_if is None or state.holds(step.only_if)
        )

    @staticmethod
    def _priority_for(state: CurrentState, plan: ActionPlan) -> Priority:
        if plan.primary_decision == TriageDecision.COLLECT_MORE_EVIDENCE.value:
            return Priority.LOW if state.weak_evidence else Priority.MEDIUM
        return _PRIORITY_BY_DECISION[plan.primary_decision]

    # 7 -----------------------------------------------------------------------
    @classmethod
    def _build_decision(cls, scenario: SecurityScenario, deliberation: GoalDeliberation) -> AgentDecision:
        plan, goal, state = deliberation.plan, deliberation.goal, deliberation.state
        rejected = cls._rejected_actions(deliberation)
        not_needed = cls._acceptable_not_selected(deliberation)
        decision = plan.primary_decision

        return AgentDecision(
            agent_name=AGENT_NAME,
            agent_type=AgentType.GOAL_BASED,
            priority=deliberation.priority,
            decision=decision,
            reasoning=cls._reasoning(deliberation, not_needed),
            evidence=scenario_evidence(scenario),
            uncertainty=uncertainty_statement(scenario),
            likely_interpretation=cls._interpretation(state),
            recommended_actions=list(plan.steps),
            audit_note=(
                f"Scenario {scenario.scenario_id} | Agent: {AGENT_NAME} | Goal: {goal.goal_id} | "
                f"Plan: {plan.plan_id} | Decision: {decision} | "
                f"{NO_ACTION_STATEMENT} {HUMAN_AUTHORIZATION_STATEMENT}"
            ),
            metadata={
                "decision_mechanism": DECISION_MECHANISM,
                "selection_policy": SELECTION_POLICY,
                "goal": {
                    "goal_id": goal.goal_id,
                    "name": goal.name,
                    "required_conditions": [c.value for c in goal.required_conditions],
                    "prohibited_conditions": [c.value for c in goal.prohibited_conditions],
                },
                "current_state": state.model_dump(mode="json"),
                "candidate_actions_considered": [c.decision for c in deliberation.candidates],
                "predicted_outcomes": {o.decision: o.model_dump(mode="json") for o in deliberation.outcomes},
                "rejected_actions": rejected,
                "acceptable_actions_not_selected": not_needed,
                "selected_action_id": plan.selected_action_id,
                "predicted_outcome": plan.predicted_outcome.model_dump(mode="json"),
                "goal_evaluation": plan.goal_evaluation.model_dump(mode="json"),
                "selected_plan": {
                    "plan_id": plan.plan_id,
                    "steps": list(plan.steps),
                    "selection_reason": plan.selection_reason,
                    "goal_satisfied": plan.goal_evaluation.satisfies_goal,
                },
                "scenario_id": scenario.scenario_id,
                "uses_internal_history": False,
                "uses_utility_scoring": False,
                "uses_learning": False,
                "uses_llm": False,
                **safety_metadata(),
            },
        )

    @staticmethod
    def _rejected_actions(deliberation: GoalDeliberation) -> list[dict[str, object]]:
        decisions = {c.action_id: c.decision for c in deliberation.candidates}
        return [
            {
                "decision": decisions[e.action_id],
                "reasons": list(e.rejection_reasons),
                "unsatisfied_conditions": [c.value for c in e.unsatisfied_conditions],
            }
            for e in deliberation.evaluations
            if not e.satisfies_goal
        ]

    @staticmethod
    def _acceptable_not_selected(deliberation: GoalDeliberation) -> list[dict[str, str]]:
        decisions = {c.action_id: c.decision for c in deliberation.candidates}
        selected = deliberation.plan.selected_action_id
        return [
            {
                "decision": decisions[e.action_id],
                "reason": "Also satisfies the goal but is more intensive than necessary; "
                "it remains available if later evidence warrants it.",
            }
            for e in deliberation.evaluations
            if e.satisfies_goal and e.action_id != selected
        ]

    @staticmethod
    def _reasoning(
        deliberation: GoalDeliberation,
        not_needed: list[dict[str, str]],
    ) -> str:
        goal, state, plan = deliberation.goal, deliberation.state, deliberation.plan
        considered = ", ".join(c.decision for c in deliberation.candidates)
        parts = [
            f"The agent evaluated {considered} against goal {goal.goal_id} ({goal.name}).",
            f"Current state: {GoalBasedAgent._describe_state(state)}.",
        ]
        # Only the leading reason per rejected action here; the full list is in metadata.
        for candidate, evaluation in zip(deliberation.candidates, deliberation.evaluations, strict=True):
            if not evaluation.satisfies_goal:
                parts.append(f"Rejected {candidate.decision}: {evaluation.rejection_reasons[0]}")
        if not_needed:
            names = ", ".join(n["decision"] for n in not_needed)
            parts.append(f"{names} would also satisfy the goal but is more intensive than necessary.")
        parts.append(plan.selection_reason)
        impact = (
            "Impact is confirmed in the scenario."
            if state.impact_confirmed
            else "Compromise is not confirmed; the recommendation supports investigation and human review only."
        )
        parts.append(f"{impact} No action was executed.")
        return " ".join(parts)

    @staticmethod
    def _describe_state(state: CurrentState) -> str:
        if state.active_risk_indicated:
            return GoalBasedAgent._describe_risks(state)
        return "no active risk indicator; evidence is " + ("limited" if state.weak_evidence else "not conclusive")

    @staticmethod
    def _interpretation(state: CurrentState) -> str:
        findings = []
        if state.impact_confirmed:
            findings.append("Impact is confirmed in the scenario; analyst-led response is needed.")
        if state.privileged_account_at_risk:
            findings.append("Possible credential-guessing attempt against a privileged account; unconfirmed.")
        if state.critical_asset_at_risk:
            findings.append("Possible serious incident on a critical service, or an undocumented but legitimate change; unconfirmed.")
        if not findings:
            findings.append(
                "Significant alert whose cause is not yet understood."
                if state.high_severity
                else "Likely benign or one-off event; evidence is too weak to conclude."
            )
        return " ".join(findings)
