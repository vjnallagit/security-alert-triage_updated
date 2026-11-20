import builtins
from pathlib import Path
from typing import get_args

import pytest
from agents import GoalBasedAgent, ModelBasedReflexAgent, SimpleReflexAgent
from agents.base import BaseAgent
from agents.common import TriageDecision
from agents.goal_based import GoalNotAchievableError
from models.goal_models import (
    ActionPlan,
    CandidateAction,
    CurrentState,
    DecisionName,
    GoalCondition,
    GoalDeliberation,
    GoalEvaluation,
    PlanStep,
    PredictedEffect,
    PredictedOutcome,
    ProhibitedCondition,
    RiskAttention,
    SecurityGoal,
)
from models.schemas import AgentDecision, Priority, SecurityScenario
from pydantic import ValidationError
from services.data_loader import DATA_DIR
from services.scenario_service import get_all_scenarios, get_scenario


# ---------------------------------------------------------------------------
# Builders for valid model instances that individual tests then corrupt
# ---------------------------------------------------------------------------


def make_goal(**overrides) -> SecurityGoal:
    data = {
        "goal_id": "GB-G01",
        "name": "Safe and proportionate security triage",
        "description": "d",
        "required_conditions": tuple(GoalCondition),
        "prohibited_conditions": tuple(ProhibitedCondition),
    }
    return SecurityGoal(**{**data, **overrides})


def make_action(**overrides) -> CandidateAction:
    data = {
        "action_id": "ACT-TEST",
        "decision": "INVESTIGATE",
        "description": "d",
        "predicted_effects": (PredictedEffect.ACTIVE_INVESTIGATION,),
        "plan_steps": (PlanStep(text="Review activity."),),
    }
    return CandidateAction(**{**data, **overrides})


def make_outcome(**overrides) -> PredictedOutcome:
    data = {
        "action_id": "ACT-TEST",
        "decision": "INVESTIGATE",
        "predicted_state": "s",
        "evidence_gap_addressed": True,
        "risk_attention_level": RiskAttention.ACTIVE_INVESTIGATION,
        "active_risk_addressed": True,
        "critical_asset_protected": True,
        "preserves_uncertainty": True,
        "requires_human_review": False,
        "unsafe_automation_avoided": True,
        "unsupported_claim_avoided": True,
    }
    return PredictedOutcome(**{**data, **overrides})


def make_evaluation(**overrides) -> GoalEvaluation:
    data = {"action_id": "ACT-TEST", "goal_id": "GB-G01", "satisfies_goal": True}
    return GoalEvaluation(**{**data, **overrides})


def make_plan(**overrides) -> ActionPlan:
    data = {
        "plan_id": "PLAN-TEST",
        "goal_id": "GB-G01",
        "selected_action_id": "ACT-TEST",
        "primary_decision": "INVESTIGATE",
        "steps": ("Step one.", "Step two."),
        "selection_reason": "r",
        "predicted_outcome": make_outcome(),
        "goal_evaluation": make_evaluation(),
    }
    return ActionPlan(**{**data, **overrides})


@pytest.fixture
def goal_state() -> CurrentState:
    return CurrentState(
        scenario_id="X", target_account="acct", asset_name="asset",
        privileged_account_at_risk=False, critical_asset_at_risk=False, high_severity=False,
        weak_evidence=True, impact_confirmed=False, evidence_gap_exists=True,
        failed_logins_present=False, source_not_known=False,
        active_risk_indicated=False, analyst_review_needed=False,
    )


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


def test_valid_models_construct():
    assert make_goal() and make_action() and make_outcome() and make_evaluation() and make_plan()


@pytest.mark.parametrize("field", ["goal_id", "name"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_goal_id_and_name_are_rejected(field, value):
    with pytest.raises(ValidationError):
        make_goal(**{field: value})


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_candidate_action_id_is_rejected(value):
    with pytest.raises(ValidationError):
        make_action(action_id=value)


def test_candidate_decision_must_be_in_defensive_vocabulary():
    with pytest.raises(ValidationError):
        make_action(decision="BLOCK_ACCOUNT")


def test_decision_name_matches_triage_decision_vocabulary():
    assert set(get_args(DecisionName)) == {d.value for d in TriageDecision}


def test_plan_with_no_steps_is_rejected():
    with pytest.raises(ValidationError):
        make_plan(steps=())


@pytest.mark.parametrize("steps", [("only one",), tuple(f"s{i}" for i in range(6))])
def test_plan_step_count_must_be_two_to_five(steps):
    with pytest.raises(ValidationError):
        make_plan(steps=steps)


def test_goal_cannot_drop_a_prohibited_condition():
    weakened = tuple(c for c in ProhibitedCondition if c is not ProhibitedCondition.OFFENSIVE_ACTION)
    with pytest.raises(ValidationError, match="OFFENSIVE_ACTION"):
        make_goal(prohibited_conditions=weakened)


def test_goal_prohibits_every_unsafe_action_named_in_requirements():
    names = {c.value for c in make_goal().prohibited_conditions}
    assert {
        "AUTOMATIC_USER_BLOCKING",
        "AUTOMATIC_ACCOUNT_DISABLING",
        "AUTOMATIC_MACHINE_QUARANTINE",
        "AUTOMATIC_FIREWALL_MODIFICATION",
        "OFFENSIVE_ACTION",
        "UNSUPPORTED_COMPROMISE_CLAIM",
        "CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION",
    } <= names


def test_goal_requires_conditions_without_duplicates():
    with pytest.raises(ValidationError):
        make_goal(required_conditions=())
    with pytest.raises(ValidationError):
        make_goal(required_conditions=(GoalCondition.EVIDENCE_GAP_ADDRESSED,) * 2)


def test_unsafe_action_cannot_be_an_approved_plan():
    with pytest.raises(ValidationError):
        make_evaluation(
            satisfies_goal=True,
            violated_prohibited_conditions=(ProhibitedCondition.AUTOMATIC_USER_BLOCKING,),
        )
    rejected = make_evaluation(
        satisfies_goal=False,
        violated_prohibited_conditions=(ProhibitedCondition.AUTOMATIC_USER_BLOCKING,),
        rejection_reasons=("Prohibited",),
    )
    with pytest.raises(ValidationError, match="satisfy the goal"):
        make_plan(goal_evaluation=rejected)


def test_evaluation_with_failed_condition_cannot_satisfy_goal():
    with pytest.raises(ValidationError):
        make_evaluation(unsatisfied_conditions=(GoalCondition.ACTIVE_RISK_ADDRESSED,))


def test_rejected_evaluation_needs_a_reason_and_accepted_must_not_have_one():
    with pytest.raises(ValidationError):
        make_evaluation(satisfies_goal=False, unsatisfied_conditions=(GoalCondition.ACTIVE_RISK_ADDRESSED,))
    with pytest.raises(ValidationError):
        make_evaluation(rejection_reasons=("odd",))


def test_selected_plans_must_satisfy_the_goal():
    rejected = make_evaluation(
        satisfies_goal=False,
        unsatisfied_conditions=(GoalCondition.ACTIVE_RISK_ADDRESSED,),
        rejection_reasons=("Inadequate",),
    )
    with pytest.raises(ValidationError, match="satisfy the goal"):
        make_plan(goal_evaluation=rejected)


def test_plan_parts_must_reference_the_same_action_and_goal():
    with pytest.raises(ValidationError, match="same action"):
        make_plan(selected_action_id="OTHER")
    with pytest.raises(ValidationError, match="different goal"):
        make_plan(goal_evaluation=make_evaluation(goal_id="OTHER-GOAL"))
    with pytest.raises(ValidationError, match="primary decision"):
        make_plan(primary_decision="MONITOR")


def test_selected_outcome_must_reference_a_considered_candidate(goal_state):
    action = make_action()
    kwargs = dict(
        goal=make_goal(),
        state=goal_state,
        candidates=(action,),
        outcomes=(make_outcome(),),
        evaluations=(make_evaluation(),),
        priority=Priority.HIGH,
    )
    assert GoalDeliberation(plan=make_plan(), **kwargs)
    other = make_plan(
        selected_action_id="OTHER",
        predicted_outcome=make_outcome(action_id="OTHER"),
        goal_evaluation=make_evaluation(action_id="OTHER"),
    )
    with pytest.raises(ValidationError, match="not one of the candidates"):
        GoalDeliberation(plan=other, **kwargs)


def test_models_reject_unexpected_fields_and_are_immutable():
    with pytest.raises(ValidationError):
        make_action(utility_score=0.9)
    with pytest.raises(ValidationError):
        make_goal(weight=1)
    with pytest.raises(ValidationError):
        make_goal().goal_id = "changed"


def test_goal_models_define_no_numeric_fields():
    """Structural guard: no model field is a number (no scores, weights or probabilities)."""
    import models.goal_models as module

    for model in (
        module.CurrentState, module.SecurityGoal, module.CandidateAction, module.PredictedOutcome,
        module.GoalEvaluation, module.ActionPlan, module.GoalDeliberation,
    ):
        for name, field in model.model_fields.items():
            assert field.annotation not in (int, float), f"{model.__name__}.{name} is numeric"


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

ALL_IDS = ["A", "B", "C"]
EXPECTED = {
    "A": (Priority.HIGH, "INVESTIGATE"),
    "B": (Priority.LOW, "COLLECT_MORE_EVIDENCE"),
    "C": (Priority.CRITICAL, "ESCALATE_TO_SECURITY_ANALYST"),
}
FOUR_DECISIONS = [d.value for d in TriageDecision]


def variant(base_id: str = "A", **changes) -> SecurityScenario:
    """A validated copy of a Phase 1 scenario with some facts changed."""
    return SecurityScenario.model_validate({**get_scenario(base_id).model_dump(mode="json"), **changes})


def rejected_by_decision(result: AgentDecision) -> dict[str, list[str]]:
    return {r["decision"]: r["reasons"] for r in result.metadata["rejected_actions"]}


def test_gb01_inherits_directly_from_base_agent():
    assert GoalBasedAgent.__bases__ == (BaseAgent,)
    assert isinstance(GoalBasedAgent(), BaseAgent)


def test_gb02_is_not_a_reflex_agent():
    assert not issubclass(GoalBasedAgent, (SimpleReflexAgent, ModelBasedReflexAgent))


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb03_to_gb05_expected_priority_and_decision(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert (result.priority, result.decision) == EXPECTED[scenario_id]


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb06_result_validates_as_agent_decision(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert AgentDecision.model_validate(result.model_dump()) == result
    assert result.agent_type.value == "GOAL_BASED"
    assert result.agent_name == "Goal-Based Agent"


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb07_result_has_all_explanation_fields(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert result.reasoning and result.evidence and result.uncertainty
    assert result.likely_interpretation and result.recommended_actions and result.audit_note
    assert result.tool_calls == [] and result.utility_score is None and result.experience_used is None


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb08_gb09_metadata_names_mechanism_and_goal(scenario_id):
    metadata = GoalBasedAgent().decide(get_scenario(scenario_id)).metadata
    assert metadata["decision_mechanism"] == "goal_satisfaction_and_plan_selection"
    assert metadata["goal"]["goal_id"] == "GB-G01"
    assert metadata["goal"]["name"] == "Safe and proportionate security triage"


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb10_all_four_candidates_are_considered(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert result.metadata["candidate_actions_considered"] == FOUR_DECISIONS
    assert set(result.metadata["predicted_outcomes"]) == set(FOUR_DECISIONS)
    for decision in FOUR_DECISIONS:
        assert decision in result.reasoning


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb11_at_least_one_rejection_with_meaningful_reason(scenario_id):
    rejected = rejected_by_decision(GoalBasedAgent().decide(get_scenario(scenario_id)))
    assert rejected
    for reasons in rejected.values():
        assert reasons and all(len(reason) > 30 for reason in reasons)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb12_gb13_selected_plan_has_two_to_five_ordered_steps(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    plan = result.metadata["selected_plan"]
    assert 2 <= len(plan["steps"]) <= 5
    assert plan["steps"] == result.recommended_actions
    assert plan["plan_id"].startswith("GB-PLAN-")


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb14_goal_evaluation_states_selected_plan_satisfies_goal(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert result.metadata["selected_plan"]["goal_satisfied"] is True
    evaluation = result.metadata["goal_evaluation"]
    assert evaluation["satisfies_goal"] is True
    assert evaluation["unsatisfied_conditions"] == []
    assert evaluation["violated_prohibited_conditions"] == []
    assert set(evaluation["satisfied_conditions"]) == {c.value for c in GoalCondition}


def test_gb15_unsafe_actions_are_prohibited_by_the_goal_model():
    result = GoalBasedAgent().decide(get_scenario("A"))
    prohibited = set(result.metadata["goal"]["prohibited_conditions"])
    assert {
        "AUTOMATIC_USER_BLOCKING", "AUTOMATIC_ACCOUNT_DISABLING", "AUTOMATIC_MACHINE_QUARANTINE",
        "AUTOMATIC_FIREWALL_MODIFICATION", "OFFENSIVE_ACTION", "UNSUPPORTED_COMPROMISE_CLAIM",
        "CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION",
    } <= prohibited


# --- Scenario A ---------------------------------------------------------------


def test_gb_a01_scenario_a_rejects_monitor_as_inadequate():
    result = GoalBasedAgent().decide(get_scenario("A"))
    reasons = rejected_by_decision(result)["MONITOR"]
    assert any("inadequate" in r.lower() and "privileged" in r.lower() for r in reasons)
    assert "COLLECT_MORE_EVIDENCE" in rejected_by_decision(result)


def test_gb_a02_scenario_a_selects_investigate_not_escalation():
    result = GoalBasedAgent().decide(get_scenario("A"))
    assert result.decision == "INVESTIGATE"
    assert result.metadata["selected_action_id"] == "GB-ACT-INVESTIGATE"
    # Escalation would also achieve the goal, but is not the least disruptive adequate response.
    assert [x["decision"] for x in result.metadata["acceptable_actions_not_selected"]] == [
        "ESCALATE_TO_SECURITY_ANALYST"
    ]


def test_gb_a03_scenario_a_does_not_claim_confirmed_compromise():
    result = GoalBasedAgent().decide(get_scenario("A"))
    assert "NOT confirmed" in result.uncertainty
    assert "Confirmed impact: False" in result.evidence
    assert "Compromise is not confirmed" in result.reasoning


def test_gb_a04_scenario_a_plan_validates_source_and_reviews_authentication():
    steps = " ".join(GoalBasedAgent().decide(get_scenario("A")).recommended_actions).lower()
    assert "authentication" in steps and "source is expected" in steps
    assert "escalate to a security analyst if" in steps  # escalation stays available


# --- Scenario B ---------------------------------------------------------------


def test_gb_b01_scenario_b_rejects_escalation_as_disproportionate():
    result = GoalBasedAgent().decide(get_scenario("B"))
    reasons = rejected_by_decision(result)["ESCALATE_TO_SECURITY_ANALYST"]
    assert any(r.startswith("Disproportionate") for r in reasons)
    assert "PROPORTIONATE_RESPONSE" in next(
        r for r in result.metadata["rejected_actions"] if r["decision"] == "ESCALATE_TO_SECURITY_ANALYST"
    )["unsatisfied_conditions"]


def test_gb_b01_scenario_b_monitor_alone_leaves_evidence_gap():
    reasons = rejected_by_decision(GoalBasedAgent().decide(get_scenario("B")))["MONITOR"]
    assert any("evidence gap" in r for r in reasons)


def test_gb_b02_scenario_b_selects_collect_more_evidence():
    result = GoalBasedAgent().decide(get_scenario("B"))
    assert (result.priority, result.decision) == (Priority.LOW, "COLLECT_MORE_EVIDENCE")
    # INVESTIGATE also satisfies the goal but is more intensive than currently justified.
    assert [x["decision"] for x in result.metadata["acceptable_actions_not_selected"]] == ["INVESTIGATE"]


def test_gb_b03_scenario_b_plan_includes_monitoring_and_reassessment():
    steps = " ".join(GoalBasedAgent().decide(get_scenario("B")).recommended_actions).lower()
    assert "monitor" in steps and "reassess" in steps


# --- Scenario C ---------------------------------------------------------------


def test_gb_c01_scenario_c_rejects_monitor_as_inadequate():
    reasons = rejected_by_decision(GoalBasedAgent().decide(get_scenario("C")))["MONITOR"]
    assert any("inadequate" in r.lower() and "critical service" in r.lower() for r in reasons)


def test_gb_c01_scenario_c_rejects_lower_actions_without_analyst_review():
    rejected = rejected_by_decision(GoalBasedAgent().decide(get_scenario("C")))
    assert {"MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE"} <= set(rejected)
    assert "ESCALATE_TO_SECURITY_ANALYST" not in rejected
    assert any("analyst review" in r for r in rejected["INVESTIGATE"])


def test_gb_c02_scenario_c_selects_escalation():
    result = GoalBasedAgent().decide(get_scenario("C"))
    assert (result.priority, result.decision) == EXPECTED["C"]
    assert result.metadata["predicted_outcome"]["requires_human_review"] is True


def test_gb_c03_scenario_c_states_compromise_is_not_confirmed():
    result = GoalBasedAgent().decide(get_scenario("C"))
    assert "NOT confirmed" in result.uncertainty
    assert "Compromise is not confirmed" in result.reasoning
    assert "Confirmed impact: False" in result.evidence


def test_gb_c04_scenario_c_escalation_is_human_analyst_review():
    result = GoalBasedAgent().decide(get_scenario("C"))
    assert "human review" in result.reasoning
    assert any("security analyst or SOC" in s and "human" in s for s in result.recommended_actions)


def test_gb_c05_scenario_c_does_not_claim_containment_executed():
    result = GoalBasedAgent().decide(get_scenario("C"))
    assert result.metadata["executed_actions"] == []
    assert "No action was executed" in result.reasoning
    assert any("only if later evidence justifies" in s for s in result.recommended_actions)


# --- Architectural differentiation ----------------------------------------------


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb_d01_metadata_has_goal_candidates_outcome_and_plan(scenario_id):
    metadata = GoalBasedAgent().decide(get_scenario(scenario_id)).metadata
    for key in ("goal", "candidate_actions_considered", "predicted_outcome", "selected_plan"):
        assert metadata[key]
    assert metadata["predicted_outcome"]["risk_attention_level"] in {a.value for a in RiskAttention}


@pytest.mark.parametrize("agent_class", [SimpleReflexAgent, ModelBasedReflexAgent])
@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb_d02_d03_reflex_agents_do_not_claim_goal_planning(agent_class, scenario_id):
    metadata = agent_class().decide(get_scenario(scenario_id)).metadata
    assert metadata["decision_mechanism"] != "goal_satisfaction_and_plan_selection"
    assert not {"goal", "selected_plan", "candidate_actions_considered", "predicted_outcome"} & set(metadata)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb_d04_to_d06_boundaries_are_declared_in_metadata(scenario_id):
    metadata = GoalBasedAgent().decide(get_scenario(scenario_id)).metadata
    assert metadata["uses_utility_scoring"] is False
    assert metadata["uses_learning"] is False
    assert metadata["uses_llm"] is False
    assert metadata["uses_internal_history"] is False


def test_gb_d07_agent_holds_no_mutable_history():
    agent = GoalBasedAgent()
    for scenario in get_all_scenarios() * 2:
        agent.decide(scenario)
    assert vars(agent) == {}


def test_gb_d08_repeated_calls_are_equivalent():
    agent = GoalBasedAgent()
    for scenario in get_all_scenarios():
        first, second = agent.decide(scenario), agent.decide(scenario)
        assert first == second
        assert first.metadata["selected_plan"] == second.metadata["selected_plan"]


def test_gb_d09_separate_instances_are_equivalent():
    for scenario in get_all_scenarios():
        assert GoalBasedAgent().decide(scenario) == GoalBasedAgent().decide(scenario)


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_gb_d10_scenario_is_not_mutated(scenario_id):
    scenario = get_scenario(scenario_id)
    snapshot = scenario.model_copy(deep=True)
    GoalBasedAgent().decide(scenario)
    assert scenario == snapshot


def test_decisions_do_not_depend_on_scenario_identity():
    """Guards against a hard-coded A/B/C lookup: only the facts matter."""
    assert GoalBasedAgent().decide(variant("A", scenario_id="ZZZ")).decision == "INVESTIGATE"
    assert GoalBasedAgent().decide(variant("B", scenario_id="A")).decision == "COLLECT_MORE_EVIDENCE"
    assert GoalBasedAgent().decide(variant("C", scenario_id="B")).decision == "ESCALATE_TO_SECURITY_ANALYST"


def test_changing_facts_changes_the_selected_plan():
    calm = variant("A", severity="LOW", failed_login_count=1, source_familiarity="KNOWN",
                   evidence_strength="LIMITED", correlated_alert_count=0, unusual_access=False)
    assert GoalBasedAgent().decide(calm).decision == "COLLECT_MORE_EVIDENCE"

    one_alert = variant("C", correlated_alert_count=1)  # critical service, but no longer correlated
    result = GoalBasedAgent().decide(one_alert)
    assert (result.priority, result.decision) == (Priority.HIGH, "INVESTIGATE")


def test_confirmed_impact_requires_analyst_review_and_is_reported_as_confirmed():
    scenario = variant("B", severity="HIGH", confirmed_impact=True, evidence_strength="STRONG")
    result = GoalBasedAgent().decide(scenario)
    assert (result.priority, result.decision) == (Priority.CRITICAL, "ESCALATE_TO_SECURITY_ANALYST")
    assert "Impact is confirmed in the scenario" in result.reasoning
    assert "NOT confirmed" not in result.uncertainty


def test_least_disruptive_acceptable_candidate_is_selected():
    result = GoalBasedAgent().decide(get_scenario("A"))
    accepted = [d for d in FOUR_DECISIONS if d not in rejected_by_decision(result)]
    assert accepted == ["INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"]
    assert result.decision == accepted[0]


def test_no_acceptable_candidate_raises_instead_of_guessing(monkeypatch):
    import agents.goal_based as module

    monkeypatch.setattr(module, "CANDIDATE_ACTIONS", module.CANDIDATE_ACTIONS[:1])  # MONITOR only
    with pytest.raises(GoalNotAchievableError):
        GoalBasedAgent().decide(get_scenario("A"))


def test_goal_agent_never_reads_or_writes_files(monkeypatch):
    scenarios = get_all_scenarios()  # load before file access is blocked
    experience = DATA_DIR / "experience.json"
    before = (experience.read_bytes(), experience.stat().st_mtime_ns)

    def blocked(*args, **kwargs):
        raise AssertionError("agent attempted file access")

    monkeypatch.setattr(builtins, "open", blocked)
    for name in ("read_text", "read_bytes", "write_text", "write_bytes", "open"):
        monkeypatch.setattr(Path, name, blocked)
    for scenario in scenarios:
        GoalBasedAgent().decide(scenario)
    monkeypatch.undo()
    assert (experience.read_bytes(), experience.stat().st_mtime_ns) == before


def test_agent_module_does_not_depend_on_data_services():
    import agents.goal_based as module

    assert not any(name.startswith("services") for name in vars(module))
    assert not hasattr(module, "get_experience")


# --- Evaluation of unsafe candidates (never in the catalogue, but must be rejected) ---


def _evaluate(action: CandidateAction, scenario: SecurityScenario | None = None) -> GoalEvaluation:
    agent = GoalBasedAgent()
    state = agent._build_current_state(scenario or get_scenario("A"))
    outcome = agent._predict_outcome(state, action)
    return agent._evaluate_goal_satisfaction(agent._define_goal(), state, action, outcome)


@pytest.mark.parametrize(
    "prohibited",
    [c for c in ProhibitedCondition if c is not ProhibitedCondition.CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION],
)
def test_candidate_with_a_prohibited_side_effect_is_rejected(prohibited):
    evaluation = _evaluate(make_action(side_effects=(prohibited,)))
    assert evaluation.satisfies_goal is False
    assert prohibited in evaluation.violated_prohibited_conditions
    assert any("Prohibited" in r for r in evaluation.rejection_reasons)


def test_consequential_action_without_human_authorization_is_rejected():
    evaluation = _evaluate(make_action(is_consequential=True, requires_human_authorization=False))
    assert not evaluation.satisfies_goal
    assert GoalCondition.HUMAN_CONTROL_PRESERVED in evaluation.unsatisfied_conditions
    assert ProhibitedCondition.CONSEQUENTIAL_ACTION_WITHOUT_HUMAN_AUTHORIZATION in evaluation.violated_prohibited_conditions


def test_consequential_action_with_human_authorization_is_acceptable():
    action = make_action(
        is_consequential=True,
        requires_human_authorization=True,
        predicted_effects=(PredictedEffect.REDUCES_EVIDENCE_GAP, PredictedEffect.ACTIVE_INVESTIGATION),
    )
    assert _evaluate(action).satisfies_goal is True


def test_unsupported_compromise_claim_rejected_unless_scenario_confirms_impact():
    claim = make_action(side_effects=(ProhibitedCondition.UNSUPPORTED_COMPROMISE_CLAIM,))
    unconfirmed = _evaluate(claim)
    assert GoalCondition.UNCERTAINTY_PRESERVED in unconfirmed.unsatisfied_conditions
    confirmed = _evaluate(claim, variant("A", confirmed_impact=True))
    assert GoalCondition.UNCERTAINTY_PRESERVED not in confirmed.unsatisfied_conditions


# --- No utility scoring ---------------------------------------------------------


def _walk(value, path=""):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_no_utility_scoring_anywhere_in_the_output(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    assert result.utility_score is None
    forbidden_keys = ("score", "utility", "weight", "reward", "probab", "rank")
    declared_flags = {"uses_utility_scoring", "uses_learning", "uses_llm", "uses_internal_history"}
    assert all(result.metadata[flag] is False for flag in declared_flags)
    for path, value in _walk({k: v for k, v in result.metadata.items() if k not in declared_flags}):
        assert not any(word in path.lower() for word in forbidden_keys), path
        assert not isinstance(value, float) and (isinstance(value, bool) or not isinstance(value, int)), path


@pytest.mark.parametrize("scenario_id", ALL_IDS)
def test_selection_is_justified_by_goal_conditions(scenario_id):
    result = GoalBasedAgent().decide(get_scenario(scenario_id))
    evaluation = result.metadata["goal_evaluation"]
    assert evaluation["satisfied_conditions"] and not evaluation["unsatisfied_conditions"]
    assert "satisfies goal GB-G01" in result.metadata["selected_plan"]["selection_reason"]
    assert result.metadata["selection_policy"] == "least_disruptive_goal_satisfying_action"
