import re

import pytest
from models.schemas import (
    AccountPrivilege,
    AssetCriticality,
    EvidenceStrength,
    ScenarioCategory,
    Severity,
    SourceFamiliarity,
)
from services.scenario_service import (
    RecordNotFoundError,
    ScenarioNotFoundError,
    get_alert_detail,
    get_all_policies,
    get_all_scenarios,
    get_asset,
    get_authentication_events,
    get_experience,
    get_policy,
    get_scenario,
)

REQUIRED_PROHIBITED_ACTIONS = {
    "block_user",
    "disable_account",
    "modify_firewall_rule",
    "terminate_process",
    "quarantine_host",
    "offensive_response",
}
# Words that would indicate a recommendation to perform a disruptive action.
DISRUPTIVE_WORDS = re.compile(
    r"\b(block|disable|terminate|kill|quarantine|isolate|attack|retaliate|hack back|firewall)\b",
    re.IGNORECASE,
)


# Test 1
def test_exactly_three_mandatory_scenarios():
    assert [s.scenario_id for s in get_all_scenarios()] == ["A", "B", "C"]


# Test 2
def test_every_scenario_validates_as_security_scenario():
    from models.schemas import SecurityScenario

    assert all(isinstance(s, SecurityScenario) for s in get_all_scenarios())


# Test 3
@pytest.mark.parametrize(
    "scenario_id, category",
    [
        ("A", ScenarioCategory.CRITICAL_FAILURE),
        ("B", ScenarioCategory.NORMAL),
        ("C", ScenarioCategory.EDGE_AMBIGUOUS),
    ],
)
def test_scenario_classification(scenario_id, category):
    assert get_scenario(scenario_id).scenario_category == category


# Test 4
def test_scenario_a_characteristics():
    a = get_scenario("A")
    assert a.account_privilege == AccountPrivilege.PRIVILEGED
    assert a.source_familiarity == SourceFamiliarity.UNFAMILIAR
    assert a.failed_login_count > 1
    assert a.confirmed_impact is False
    assert a.source == "203.0.113.25"
    assert a.correlated_alert_count >= 1


# Test 5
def test_scenario_b_characteristics():
    b = get_scenario("B")
    assert b.severity == Severity.LOW
    assert b.evidence_strength == EvidenceStrength.LIMITED
    assert b.confirmed_impact is False
    assert b.correlated_alert_count == 0


# Test 6
def test_scenario_c_characteristics():
    c = get_scenario("C")
    assert c.asset_criticality == AssetCriticality.CRITICAL
    assert c.correlated_alert_count > 1
    assert c.unusual_access is True
    assert c.confirmed_impact is False
    assert c.evidence_strength in {EvidenceStrength.MODERATE, EvidenceStrength.CONFLICTING}


# Test 9
def test_unknown_scenario_raises_understandable_error():
    with pytest.raises(ScenarioNotFoundError, match="DOES_NOT_EXIST"):
        get_scenario("DOES_NOT_EXIST")


# Test 10
@pytest.mark.parametrize("scenario", get_all_scenarios(), ids=lambda s: s.scenario_id)
def test_scenario_references_resolve(scenario):
    alert = get_alert_detail(scenario.alert_id)
    asset = get_asset(scenario.asset_id)
    policy = get_policy(scenario.policy_id)
    assert get_authentication_events(scenario.alert_id), "no authentication events"

    # Supporting records must agree with the scenario they describe.
    assert alert.alert_type == scenario.alert_type
    assert alert.severity == scenario.severity
    assert alert.target == scenario.target_account
    assert alert.source == scenario.source
    assert asset.asset_name == scenario.asset_name
    assert asset.criticality == scenario.asset_criticality
    assert policy.policy_id == scenario.policy_id


@pytest.mark.parametrize("scenario", get_all_scenarios(), ids=lambda s: s.scenario_id)
def test_failed_logins_match_authentication_events(scenario):
    events = get_authentication_events(scenario.alert_id)
    failures = [e for e in events if not e.success]
    assert len(failures) == scenario.failed_login_count
    assert all(e.asset_id == scenario.asset_id for e in events)
    assert all(e.account == scenario.target_account for e in events)


def test_scenario_a_has_no_successful_login_from_unfamiliar_source():
    events = get_authentication_events("ALERT-A-001")
    assert not any(e.success for e in events)


def test_missing_supporting_records_raise_clear_error():
    with pytest.raises(RecordNotFoundError):
        get_alert_detail("NOPE")
    with pytest.raises(RecordNotFoundError):
        get_asset("NOPE")
    with pytest.raises(RecordNotFoundError):
        get_policy("NOPE")


# Test 11
@pytest.mark.parametrize("policy", get_all_policies(), ids=lambda p: p.policy_id)
def test_policies_do_not_permit_automatic_disruptive_actions(policy):
    assert policy.automated_actions_permitted is False
    assert policy.human_authorization_required is True
    assert REQUIRED_PROHIBITED_ACTIONS <= set(policy.prohibited_automated_actions)
    assert "human authorization" in policy.restriction
    for action in policy.recommended_actions:
        assert not DISRUPTIVE_WORDS.search(action), f"disruptive recommendation: {action}"


# Test 12
def test_experience_data_is_valid_and_useful():
    experience = get_experience()
    assert experience.simulated is True
    assert experience.records

    for actions in experience.records.values():
        for outcome in actions.values():
            assert outcome.success >= 0
            assert outcome.failure >= 0

    investigate = experience.records["repeated_failed_login"]["investigate"]
    assert investigate.success_rate == pytest.approx(0.8)


def test_experience_covers_every_scenario_alert_type():
    experience = get_experience()
    for scenario in get_all_scenarios():
        assert scenario.alert_type in experience.records
