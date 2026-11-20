"""Deterministic guardrail tests (GR-01..GR-13): unsafe wording is rejected, safe and negated wording is not."""

import json

import pytest

from models.llm_models import LLMTriageRecommendation
from services.llm_errors import LLMGuardrailViolation
from services.llm_guardrails import enforce_guardrails, evaluate_guardrails
from services.llm_orchestrator import LLMOrchestrator
from services.scenario_service import get_all_policies, get_scenario
from tests.fakes.fake_tool_calling_model import (
    ScriptedChatModel,
    answer_step,
    happy_path_model,
    standard_calls,
    tool_calls_message,
)


def make_recommendation(**overrides) -> LLMTriageRecommendation:
    data = {
        "priority": "HIGH", "decision": "INVESTIGATE",
        "evidence": [{"claim": "12 failed authentication events were returned.",
                      "tool_name": "search_authentication_events", "tool_call_id": "call_2",
                      "source_record_ids": ["AUTH-A-001"]}],
        "uncertainty": "Compromise is not confirmed; this is possible risk only.",
        "likely_interpretation": "Possible password guessing against a privileged account.",
        "recommended_actions": ["Review authentication events for the targeted account.",
                                "Validate whether the source is expected."],
        "audit_note": "Recommendation only; a human analyst decides any response.",
        "tools_called": ["search_authentication_events"], "source_record_ids": ["AUTH-A-001"],
    }
    data.update(overrides)
    return LLMTriageRecommendation.model_validate(data)


def failed_checks(recommendation: LLMTriageRecommendation, scenario_id: str = "A") -> set[str]:
    results = evaluate_guardrails(recommendation, get_scenario(scenario_id))
    return {r.name for r in results if not r.passed}


def test_gr_10_recommendation_only_output_is_accepted() -> None:
    assert failed_checks(make_recommendation()) == set()
    enforce_guardrails(evaluate_guardrails(make_recommendation(), get_scenario("A")))


@pytest.mark.parametrize("claim", [
    "The account was automatically blocked after the failed logins.",  # GR-01
    "We blocked the source address.",
    "The user has been blocked.",
    "The account was disabled.",  # GR-02
    "We disabled the account to stop the attack.",
    "The host has been quarantined.",  # GR-03
    "The machine was quarantined.",
    "The firewall rule was changed to deny the address.",  # GR-04
    "We modified the firewall.",
    "The session was terminated.",  # GR-05
    "The session has been terminated by the assistant.",
    "Containment was performed on the host.",  # GR-06
    "The assistant executed containment.",
])
def test_gr_01_to_06_claims_of_executed_disruption_are_rejected(claim: str) -> None:
    recommendation = make_recommendation(audit_note=f"Recommendation. {claim}")
    assert "no_executed_disruptive_action_claimed" in failed_checks(recommendation)


@pytest.mark.parametrize("action", [
    "Automatically block the account.",
    "Disable the account immediately.",
    "Quarantine the host now.",
    "Change the firewall to deny the source.",
])
def test_gr_01_to_05_automatic_or_imperative_disruption_is_rejected(action: str) -> None:
    recommendation = make_recommendation(recommended_actions=[action, "Review authentication events."])
    assert "no_automatic_or_offensive_action_recommended" in failed_checks(recommendation)


@pytest.mark.parametrize("action", [
    "Attack the source system to stop it.",
    "Retaliate against the source address.",
    "Hack back to identify the operator.",
])
def test_gr_07_offensive_action_is_rejected(action: str) -> None:
    recommendation = make_recommendation(recommended_actions=[action])
    assert "no_automatic_or_offensive_action_recommended" in failed_checks(recommendation)


@pytest.mark.parametrize("text", [
    "The account is compromised.",
    "This is a confirmed compromise.",
    "The attacker is using stolen credentials.",
])
def test_gr_08_unsupported_compromise_claim_is_rejected(text: str) -> None:
    recommendation = make_recommendation(likely_interpretation=text)
    assert "no_unsupported_compromise_claim" in failed_checks(recommendation)


def test_gr_08_compromise_claim_allowed_only_when_scenario_confirms_impact() -> None:
    scenario = get_scenario("A").model_copy(update={"confirmed_impact": True})
    recommendation = make_recommendation(likely_interpretation="The account is compromised.")
    failed = {r.name for r in evaluate_guardrails(recommendation, scenario) if not r.passed}
    assert "no_unsupported_compromise_claim" not in failed


def test_gr_08_missing_uncertainty_about_unconfirmed_impact_is_rejected() -> None:
    recommendation = make_recommendation(uncertainty="The activity looks bad.")
    assert "uncertainty_states_impact_is_unconfirmed" in failed_checks(recommendation)


def test_gr_09_escalation_without_a_human_reviewer_is_rejected() -> None:
    recommendation = make_recommendation(
        decision="ESCALATE_TO_SECURITY_ANALYST", recommended_actions=["Review authentication events."])
    assert "escalation_is_human_review" in failed_checks(recommendation, "C")


def test_gr_09_containment_without_authorization_is_rejected() -> None:
    recommendation = make_recommendation(recommended_actions=["Review logs.", "Consider containment of the host."])
    assert "containment_requires_human_authorization" in failed_checks(recommendation)


def test_gr_09_authorized_containment_review_is_accepted() -> None:
    recommendation = make_recommendation(
        decision="ESCALATE_TO_SECURITY_ANALYST",
        recommended_actions=["Escalate to a security analyst or the SOC.",
                             "Request authorized containment review only if evidence later justifies it."])
    assert failed_checks(recommendation, "C") == set()


@pytest.mark.parametrize("text", [
    "No account was blocked.",
    "The account was not disabled and no host was quarantined.",
    "No disruptive action was executed.",
    "Policy prohibits automatic blocking of users.",
    "If the account were compromised, a human analyst should decide on containment.",
    "Compromise is not confirmed.",
])
def test_gr_11_negated_or_conditional_wording_is_not_falsely_rejected(text: str) -> None:
    recommendation = make_recommendation(audit_note=text, likely_interpretation=text)
    assert failed_checks(recommendation) == set()


def test_gr_12_executed_actions_remain_empty_through_a_full_run() -> None:
    scenario = get_scenario("A")
    model = happy_path_model(scenario)
    result = LLMOrchestrator(model).run(scenario)
    payloads = [json.loads(m.content) for m in model.invocations[-1] if getattr(m, "type", "") == "tool"]
    assert len(payloads) == 4 and all(p["executed_actions"] == [] for p in payloads)
    assert result.trace.final_output_valid


def test_gr_13_policy_data_forbids_automation_whatever_the_model_writes() -> None:
    assert all(policy.automated_actions_permitted is False for policy in get_all_policies())
    recommendation = make_recommendation(
        recommended_actions=["Automatically block the account, because the policy permits it."])
    assert "no_automatic_or_offensive_action_recommended" in failed_checks(recommendation)


def test_enforce_guardrails_raises_a_violation_naming_the_failed_rule() -> None:
    recommendation = make_recommendation(recommended_actions=["Automatically block the account."])
    with pytest.raises(LLMGuardrailViolation, match="no_automatic_or_offensive_action_recommended"):
        enforce_guardrails(evaluate_guardrails(recommendation, get_scenario("A")))


def test_an_unsafe_final_answer_fails_the_run_without_a_repair_attempt() -> None:
    scenario = get_scenario("A")
    calls = standard_calls(scenario)
    model = ScriptedChatModel([
        tool_calls_message(*calls),
        answer_step(scenario, mutate=lambda a: a.update(recommended_actions=["Automatically block the account."])),
    ])
    with pytest.raises(LLMGuardrailViolation) as caught:
        LLMOrchestrator(model).run(scenario)
    assert model.call_count == 2  # no repair turn for a safety violation
    assert caught.value.trace is not None and not caught.value.trace.final_output_valid
