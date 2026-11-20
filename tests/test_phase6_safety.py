import json
import re

import pytest
from agents import GoalBasedAgent, LearningAgent, SimpleReflexAgent, UtilityBasedAgent
from models.schemas import AgentDecision
from models.tool_models import TOOL_CATALOGUE, ToolResultBase, ToolStatus
from services.scenario_service import get_all_scenarios
from services.security_tool_service import SecurityToolService
from tools import (
    get_alert_details,
    get_asset_criticality,
    get_response_policy,
    search_authentication_events,
)

# Phrases claiming a disruptive action happened, a remediation occurred, or a session was ended.
DISRUPTIVE_CLAIMS = re.compile(
    r"\b(blocked|disabled|terminated|quarantined|isolated|deleted|attacked|retaliated|remediated|contained)\b"
    r"|firewall (was|has been|rule changed)|changed the firewall|session (was|has been) terminated"
    r"|containment (was|has been) (performed|executed|applied)",
    re.IGNORECASE,
)
# Claims that compromise is established. Negated or conditional forms must not match.
COMPROMISE_CLAIMS = re.compile(
    r"(?<!not )(?<!un)confirmed (compromise|breach)|compromise (is|was|has been) confirmed"
    r"|(account|system|asset|user) (is|was|has been) compromised",
    re.IGNORECASE,
)
SECRET_WORDS = re.compile(r"password|passwd|secret|token|api[_-]?key|credential|private[_-]?key", re.IGNORECASE)
DECISION_KEYS = {"agent_name", "agent_type", "priority", "decision", "reasoning", "likely_interpretation",
                 "audit_note", "utility_score", "recommended_actions"}


def results_for_every_scenario(service: SecurityToolService | None = None) -> dict[str, list[ToolResultBase]]:
    """For each tool, the results for scenarios A, B and C (using only each scenario's own references)."""
    out: dict[str, list[ToolResultBase]] = {d.name: [] for d in TOOL_CATALOGUE}
    kwargs = {"service": service} if service else {}
    for s in get_all_scenarios():
        out["get_alert_details"].append(get_alert_details(s.alert_id, **kwargs))
        out["search_authentication_events"].append(
            search_authentication_events(scenario_id=s.scenario_id, **kwargs))
        out["get_asset_criticality"].append(get_asset_criticality(s.asset_id, **kwargs))
        out["get_response_policy"].append(get_response_policy(policy_id=s.policy_id, **kwargs))
    return out


def all_text(result: ToolResultBase) -> str:
    return json.dumps(result.model_dump(mode="json"))


def keys_recursive(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_recursive(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_recursive(v)}
    return set()


RESULTS = results_for_every_scenario()
TOOL_NAMES = [d.name for d in TOOL_CATALOGUE]


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_every_tool_is_read_only_and_synthetic_in_catalogue_and_results(tool_name):
    definition = next(d for d in TOOL_CATALOGUE if d.name == tool_name)
    assert definition.read_only is True and definition.synthetic is True
    for result in RESULTS[tool_name]:
        assert result.status is ToolStatus.SUCCESS
        assert result.is_read_only is True and result.is_synthetic is True and result.executed_actions == ()


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_no_result_claims_blocking_disabling_quarantine_firewall_session_or_containment_or_remediation(tool_name):
    for result in RESULTS[tool_name]:
        assert not DISRUPTIVE_CLAIMS.search(all_text(result)), all_text(result)


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_no_result_returns_credentials_or_secrets(tool_name):
    for result in RESULTS[tool_name]:
        assert not SECRET_WORDS.search(all_text(result)), all_text(result)


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_unconfirmed_scenarios_remain_unconfirmed(tool_name):
    for scenario, result in zip(get_all_scenarios(), RESULTS[tool_name], strict=True):
        assert scenario.confirmed_impact is False
        assert not COMPROMISE_CLAIMS.search(all_text(result)), f"{scenario.scenario_id}: {all_text(result)}"
        if tool_name == "get_alert_details":
            assert result.evidence.confirmed_impact is None


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_tools_make_no_final_decision_and_return_no_agent_decision(tool_name):
    for result in RESULTS[tool_name]:
        assert not isinstance(result, AgentDecision)
        assert not DECISION_KEYS & keys_recursive(result.model_dump(mode="json")), tool_name
        assert not DECISION_KEYS & set(result.to_llm_evidence()["evidence"] or {})


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_every_result_discloses_limitations_and_names_its_source_records(tool_name):
    for result in RESULTS[tool_name]:
        assert result.limitations and "synthetic" in " ".join(result.limitations).lower()
        assert result.source_type and result.source_record_ids  # never an unsourced answer
        assert all(source_id.strip() for source_id in result.source_record_ids)


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_failed_lookups_stay_safe_no_paths_no_stack_traces(tool_name):
    service = SecurityToolService()
    bad = {
        "get_alert_details": lambda: service.get_alert_details("../../etc/passwd"),
        "search_authentication_events": lambda: service.search_authentication_events(scenario_id="C:\\Windows\\system32"),
        "get_asset_criticality": lambda: service.get_asset_criticality("..\\..\\secret.json"),
        "get_response_policy": lambda: service.get_response_policy(policy_id="/Users/someone/policy"),
    }[tool_name]()
    assert bad.status is ToolStatus.NOT_FOUND and bad.evidence is None
    # The caller's own input is echoed in `query`; our error-facing fields must not expose paths or traces.
    user_facing = json.dumps([bad.error.model_dump(), bad.summary, list(bad.warnings)])
    assert "Traceback" not in user_facing and "C:\\\\Windows" not in user_facing and "/Users/" not in user_facing
    assert "etc/passwd" not in user_facing and "secret.json" not in user_facing


def test_policy_restrictions_are_retained_in_every_returned_policy():
    for result in RESULTS["get_response_policy"]:
        evidence = result.evidence
        assert evidence.automatic_disruptive_action_permitted is False
        assert evidence.human_authorization_required is True
        assert {"block_user", "disable_account", "quarantine_host", "modify_firewall_rule",
                "offensive_response"} <= set(evidence.restricted_actions)
        assert "Consequential security actions require human authorization" in evidence.restriction_statement


def test_scenario_c_policy_requires_human_analyst_escalation_and_authorized_containment_review_only():
    c_policy = RESULTS["get_response_policy"][2].evidence
    assert "Escalate to a security analyst" in " ".join(c_policy.recommended_investigation_actions)
    containment = [a for a in c_policy.recommended_investigation_actions if "containment" in a.lower()]
    assert containment and all("authorized" in a and "human approver" in a for a in containment)
    assert "SOC" in c_policy.escalation_criteria


def test_scenario_b_policy_remains_proportionate():
    b_policy = RESULTS["get_response_policy"][1].evidence
    assert "Do not escalate on limited evidence alone" in b_policy.escalation_criteria
    assert "Monitor activity" in " ".join(b_policy.recommended_investigation_actions)


def test_tool_layer_never_touches_agent_classes_or_decision_logic():
    import services.security_tool_service as service_module
    import tools.security_tools as tools_module

    for module in (service_module, tools_module):
        for name in ("SimpleReflexAgent", "GoalBasedAgent", "UtilityBasedAgent", "LearningAgent", "BaseAgent",
                     "AgentDecision", "ExperienceRepository"):
            assert not hasattr(module, name), (module.__name__, name)
    forbidden_classes = (SimpleReflexAgent, GoalBasedAgent, UtilityBasedAgent, LearningAgent)
    for module in (service_module, tools_module):
        assert not any(value is cls for value in vars(module).values() for cls in forbidden_classes)


def test_a_tool_error_never_claims_remediation_or_exposes_internals(tmp_path):
    (tmp_path / "alert_details.json").write_text("{ bad", encoding="utf-8")
    service = SecurityToolService(alert_details_path=tmp_path / "alert_details.json")
    result = service.get_alert_details("ALERT-A-001")
    assert result.status is ToolStatus.ERROR
    assert str(tmp_path) not in all_text(result) and not DISRUPTIVE_CLAIMS.search(all_text(result))
    assert result.executed_actions == () and result.is_read_only is True


def test_compromise_and_disruption_patterns_detect_claims_but_not_negations():
    assert COMPROMISE_CLAIMS.search("The account was compromised.")
    assert COMPROMISE_CLAIMS.search("This is a confirmed compromise.")
    assert not COMPROMISE_CLAIMS.search("Compromise is not confirmed.")
    assert not COMPROMISE_CLAIMS.search("if compromise is later confirmed")
    assert DISRUPTIVE_CLAIMS.search("The account was blocked.") and DISRUPTIVE_CLAIMS.search("Alert remediated.")
    assert not DISRUPTIVE_CLAIMS.search("block_user is a restricted action")
