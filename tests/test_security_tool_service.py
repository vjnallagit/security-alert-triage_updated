import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from models.tool_models import ToolStatus
from services.data_loader import DATA_DIR
from services.security_tool_service import SecurityToolService, ToolDataError

DATA_FILES = ("alert_details.json", "authentication_events.json", "assets.json", "policies.json", "scenarios.json")
RAW_EVENTS = json.loads((DATA_DIR / "authentication_events.json").read_text(encoding="utf-8"))
RAW_ALERTS = {r["alert_id"]: r for r in json.loads((DATA_DIR / "alert_details.json").read_text(encoding="utf-8"))}
RAW_ASSETS = {r["asset_id"]: r for r in json.loads((DATA_DIR / "assets.json").read_text(encoding="utf-8"))}
RAW_POLICIES = {r["policy_id"]: r for r in json.loads((DATA_DIR / "policies.json").read_text(encoding="utf-8"))}
RAW_SCENARIOS = {r["scenario_id"]: r for r in json.loads((DATA_DIR / "scenarios.json").read_text(encoding="utf-8"))}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


@pytest.fixture
def service() -> SecurityToolService:
    return SecurityToolService()


@pytest.fixture
def data_copy(tmp_path) -> Path:
    """A writable copy of the five synthetic source files that tests may corrupt."""
    target = tmp_path / "data"
    target.mkdir()
    for name in DATA_FILES:
        shutil.copy(DATA_DIR / name, target / name)
    return target


def service_for(directory: Path, **kwargs) -> SecurityToolService:
    return SecurityToolService(
        alert_details_path=directory / "alert_details.json",
        authentication_events_path=directory / "authentication_events.json",
        assets_path=directory / "assets.json",
        policies_path=directory / "policies.json",
        scenarios_path=directory / "scenarios.json",
        **kwargs,
    )


def rewrite(directory: Path, name: str, mutate) -> None:
    path = directory / name
    data = json.loads(path.read_text(encoding="utf-8"))
    mutated = mutate(data)
    path.write_text(json.dumps(data if mutated is None else mutated, indent=2), encoding="utf-8")


def source_hashes(directory: Path) -> dict[str, str]:
    return {name: digest(directory / name) for name in DATA_FILES}


# ===========================================================================
# get_alert_details (AD-01 .. AD-13)
# ===========================================================================


def test_ad01_ad02_ad11_known_alert_returns_success_with_its_own_id(service):
    result = service.get_alert_details("ALERT-A-001")
    assert result.status is ToolStatus.SUCCESS and result.record_found is True and result.error is None
    assert result.evidence.alert_id == "ALERT-A-001"
    assert result.source_record_ids == ("ALERT-A-001",)
    assert result.query == {"alert_id": "ALERT-A-001", "include_raw_evidence": True}


@pytest.mark.parametrize("alert_id", sorted(RAW_ALERTS))
def test_ad03_returned_details_match_the_source_file(service, alert_id):
    raw, evidence = RAW_ALERTS[alert_id], service.get_alert_details(alert_id).evidence
    assert evidence.timestamp == parse(raw["timestamp"])
    assert (evidence.alert_type, evidence.severity.value, evidence.summary, evidence.target, evidence.source) == (
        raw["alert_type"], raw["severity"], raw["summary"], raw["target"], raw["source"])
    assert evidence.evidence_items == tuple(raw["evidence"])


def test_ad04_scenario_a_evidence_carries_the_repeated_login_facts_in_the_source(service):
    evidence = service.get_alert_details("ALERT-A-001").evidence
    assert evidence.alert_type == "repeated_failed_login" and evidence.severity.value == "HIGH"
    assert evidence.source == "203.0.113.25" and evidence.target == "admin.synthetic01"
    text = " ".join(evidence.evidence_items)
    assert "12 failed authentication attempts" in text and "No successful login" in text


def test_ad05_scenario_b_returns_low_severity(service):
    evidence = service.get_alert_details("ALERT-B-001").evidence
    assert evidence.severity.value == "LOW" and "No related alerts" in " ".join(evidence.evidence_items)


def test_ad06_scenario_c_does_not_claim_confirmed_compromise(service):
    result = service.get_alert_details("ALERT-C-001")
    assert result.evidence.confirmed_impact is None  # the record does not state impact; the tool does not infer it
    text = (result.summary + " " + " ".join(result.evidence.evidence_items)).lower()
    assert "compromise" not in text and "No data loss or service disruption observed" in " ".join(result.evidence.evidence_items)


def test_ad07_unknown_alert_returns_not_found(service):
    result = service.get_alert_details("ALERT-DOES-NOT-EXIST")
    assert result.status is ToolStatus.NOT_FOUND and result.record_found is False
    assert result.evidence is None and result.source_record_ids == ()
    assert result.error.error_type == "NotFound" and "no synthetic alert" in result.error.message.lower()


@pytest.mark.parametrize("alert_id", ["", "   "])
def test_ad08_blank_alert_id_is_reported_as_invalid_input_consistently(service, alert_id):
    result = service.get_alert_details(alert_id)
    assert result.status is ToolStatus.INVALID_INPUT and result.evidence is None
    assert "alert_id" in result.error.message and result.record_found is False


def test_non_string_alert_id_is_invalid_input_not_coerced(service):
    assert service.get_alert_details(12345).status is ToolStatus.INVALID_INPUT  # type: ignore[arg-type]


def test_ad09_duplicate_alert_ids_return_ambiguous_and_none_is_chosen(data_copy):
    rewrite(data_copy, "alert_details.json", lambda d: d.append({**d[0], "summary": "A conflicting duplicate record."}))
    result = service_for(data_copy).get_alert_details("ALERT-A-001")
    assert result.status is ToolStatus.AMBIGUOUS and result.evidence is None and result.source_record_ids == ()
    assert result.error.error_type == "Ambiguous" and "2" in result.warnings[0]


def test_ad10_no_unrelated_alert_record_is_returned(service):
    result = service.get_alert_details("ALERT-B-001")
    dumped = json.dumps(result.model_dump(mode="json"))
    assert "ALERT-A-001" not in dumped and "ALERT-C-001" not in dumped and "admin.synthetic01" not in dumped


def test_include_raw_evidence_false_omits_the_evidence_items_with_a_warning(service):
    result = service.get_alert_details("ALERT-A-001", include_raw_evidence=False)
    assert result.evidence.evidence_items == () and result.warnings and result.evidence.alert_id == "ALERT-A-001"


def test_ad12_result_is_an_evidence_envelope_not_a_decision(service):
    dumped = service.get_alert_details("ALERT-A-001").model_dump(mode="json")
    assert not {"decision", "priority", "agent_name", "agent_type", "reasoning", "recommended_actions"} & set(dumped)
    assert not {"decision", "priority"} & set(dumped["evidence"])


def test_ad13_input_fixtures_are_never_modified(data_copy):
    before = source_hashes(data_copy)
    service = service_for(data_copy)
    for alert_id in (*RAW_ALERTS, "NOPE", " "):
        service.get_alert_details(alert_id)
    assert source_hashes(data_copy) == before


# ===========================================================================
# search_authentication_events (AE-01 .. AE-24)
# ===========================================================================


def raw_matches(**criteria) -> list[dict]:
    return [e for e in RAW_EVENTS if all(e[k] == v for k, v in criteria.items())]


def test_ae01_search_by_scenario_a_account_returns_the_expected_events(service):
    result = service.search_authentication_events(account="admin.synthetic01")
    assert result.status is ToolStatus.SUCCESS
    assert {e.event_id for e in result.evidence.events} == {e["event_id"] for e in raw_matches(account="admin.synthetic01")}
    assert result.evidence.matched_count == 12


def test_ae02_search_by_source_filters_correctly(service):
    result = service.search_authentication_events(source="203.0.113.25")
    assert {e.source for e in result.evidence.events} == {"203.0.113.25"}
    assert result.evidence.matched_count == len(raw_matches(source="203.0.113.25"))
    assert service.search_authentication_events(source="198.51.100.17").evidence.matched_count == 2


def test_ae03_search_by_asset_id_filters_correctly(service):
    result = service.search_authentication_events(asset_id="ASSET-PAY-01")
    assert result.evidence.matched_count == len(raw_matches(asset_id="ASSET-PAY-01")) == 3
    assert {e.asset_id for e in result.evidence.events} == {"ASSET-PAY-01"}


def test_ae04_search_by_scenario_id_resolves_through_the_scenarios_alert_id(service):
    for scenario_id, scenario in RAW_SCENARIOS.items():
        result = service.search_authentication_events(scenario_id=scenario_id)
        assert {e.alert_id for e in result.evidence.events} == {scenario["alert_id"]}
        assert result.evidence.matched_count == len(raw_matches(alert_id=scenario["alert_id"]))
    unknown = service.search_authentication_events(scenario_id="Z")
    assert unknown.status is ToolStatus.NOT_FOUND and unknown.error.error_type == "NotFound"
    assert service.search_authentication_events(alert_id="ALERT-B-001", scenario_id="A").evidence.matched_count == 0


def test_ae05_success_false_returns_failed_events_only(service):
    result = service.search_authentication_events(account="admin.synthetic01", success=False)
    assert result.evidence.matched_count == 12 and all(not e.success for e in result.evidence.events)
    assert service.search_authentication_events(asset_id="ASSET-PAY-01", success=False).evidence.matched_count == 1


def test_ae06_success_true_returns_successful_events_only(service):
    result = service.search_authentication_events(asset_id="ASSET-PAY-01", success=True)
    assert result.evidence.matched_count == 2 and all(e.success for e in result.evidence.events)
    assert service.search_authentication_events(account="admin.synthetic01", success=True).evidence.matched_count == 0


def test_ae07_combined_filters_apply_logical_and(service):
    result = service.search_authentication_events(account="svc-payments-api", source="192.0.2.80", success=True)
    expected = [e for e in RAW_EVENTS if e["account"] == "svc-payments-api" and e["source"] == "192.0.2.80" and e["success"]]
    assert {e.event_id for e in result.evidence.events} == {e["event_id"] for e in expected}
    assert service.search_authentication_events(account="admin.synthetic01", source="192.0.2.80").evidence.matched_count == 0


def test_ae08_results_are_newest_first_with_event_id_breaking_ties(service, data_copy):
    events = service.search_authentication_events(account="admin.synthetic01").evidence.events
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps, reverse=True) and events[0].event_id == "AUTH-A-012"

    def make_ties(data):
        base = data[0]
        return [{**base, "event_id": "AUTH-T-2", "timestamp": "2026-03-10T05:00:00+00:00", "account": "tie"},
                {**base, "event_id": "AUTH-T-1", "timestamp": "2026-03-10T05:00:00+00:00", "account": "tie"},
                {**base, "event_id": "AUTH-T-3", "timestamp": "2026-03-10T06:00:00+00:00", "account": "tie"}]

    rewrite(data_copy, "authentication_events.json", make_ties)
    tied = service_for(data_copy).search_authentication_events(account="tie").evidence.events
    assert [e.event_id for e in tied] == ["AUTH-T-3", "AUTH-T-1", "AUTH-T-2"]


def test_ae09_ae10_matched_and_returned_counts_are_correct(service):
    result = service.search_authentication_events(account="admin.synthetic01")
    assert result.evidence.matched_count == result.evidence.returned_count == len(result.evidence.events) == 12
    limited = service.search_authentication_events(account="admin.synthetic01", limit=5)
    assert (limited.evidence.matched_count, limited.evidence.returned_count) == (12, 5)


@pytest.mark.parametrize("criteria", [{"account": "admin.synthetic01"}, {"alert_id": "ALERT-B-001"},
                                      {"asset_id": "ASSET-PAY-01"}, {"success": False}, {"success": True}])
def test_ae11_ae12_failed_and_successful_counts_match_the_raw_data(service, criteria):
    matched = [e for e in RAW_EVENTS if all(e[k] == v for k, v in criteria.items())]
    evidence = service.search_authentication_events(**criteria).evidence
    assert evidence.failed_login_count == sum(1 for e in matched if not e["success"])
    assert evidence.successful_login_count == sum(1 for e in matched if e["success"])
    assert evidence.failed_login_count + evidence.successful_login_count == evidence.matched_count == len(matched)


def test_ae13_unique_sources_are_deduplicated_and_sorted(service):
    evidence = service.search_authentication_events(success=True).evidence
    assert evidence.unique_sources == tuple(sorted({e["source"] for e in RAW_EVENTS if e["success"]}))
    assert service.search_authentication_events(account="admin.synthetic01").evidence.unique_sources == ("203.0.113.25",)


def test_ae14_no_matches_is_a_valid_empty_observation(service):
    result = service.search_authentication_events(account="nobody.synthetic")
    assert result.status is ToolStatus.SUCCESS and result.error is None
    assert result.evidence.matched_count == 0 and result.evidence.events == () and result.evidence.time_range is None
    assert result.source_record_ids == () and "No matching synthetic authentication events" in result.summary


def test_ae15_ae16_limit_truncates_but_preserves_matched_count(service):
    truncated = service.search_authentication_events(account="admin.synthetic01", limit=5)
    assert truncated.evidence.truncated is True and truncated.warnings and "5 of 12" in truncated.warnings[0]
    assert truncated.evidence.failed_login_count == 12  # counts cover every match, not just the returned page
    assert [e.event_id for e in truncated.evidence.events] == [
        "AUTH-A-012", "AUTH-A-011", "AUTH-A-010", "AUTH-A-009", "AUTH-A-008",
    ]  # the newest five, in order
    full = service.search_authentication_events(account="admin.synthetic01", limit=12)
    assert full.evidence.truncated is False and full.warnings == ()
    assert service.search_authentication_events(account="admin.synthetic01", limit=1).evidence.truncated is True


def test_ae17_timezone_aware_time_filtering_works(service):
    start, end = parse("2026-03-10T02:10:00+00:00"), parse("2026-03-10T02:12:00+00:00")
    expected = [e["event_id"] for e in RAW_EVENTS if e["account"] == "admin.synthetic01" and start <= parse(e["timestamp"]) <= end]
    result = service.search_authentication_events(account="admin.synthetic01", start_time=start, end_time=end)
    assert sorted(e.event_id for e in result.evidence.events) == sorted(expected) and len(expected) == 5
    # The same instant in another timezone selects the same events.
    other_zone = timezone_shift(start), timezone_shift(end)
    again = service.search_authentication_events(account="admin.synthetic01", start_time=other_zone[0], end_time=other_zone[1])
    assert again.evidence.matched_count == 5
    assert result.evidence.time_range.earliest == start and result.evidence.time_range.latest == end


def timezone_shift(moment: datetime) -> datetime:
    from datetime import timezone

    return moment.astimezone(timezone(timedelta(hours=5, minutes=30)))


def test_ae18_start_after_end_is_rejected(service):
    result = service.search_authentication_events(account="a", start_time=parse("2026-03-10T05:00:00+00:00"),
                                                  end_time=parse("2026-03-10T04:00:00+00:00"))
    assert result.status is ToolStatus.INVALID_INPUT and "must not be after" in result.error.message


def test_naive_datetimes_are_rejected_not_assumed_utc(service):
    result = service.search_authentication_events(account="a", start_time=datetime(2026, 3, 10, 2, 0))
    assert result.status is ToolStatus.INVALID_INPUT


def test_ae19_no_filters_is_rejected(service):
    for result in (service.search_authentication_events(), service.search_authentication_events(limit=10)):
        assert result.status is ToolStatus.INVALID_INPUT and "at least one search criterion" in result.error.message


@pytest.mark.parametrize("limit", [0, 101, -3])
def test_limit_out_of_range_is_invalid_input(service, limit):
    assert service.search_authentication_events(account="a", limit=limit).status is ToolStatus.INVALID_INPUT


def test_ae20_tool_does_not_claim_compromise_or_attribute_attackers(service):
    for scenario_id in RAW_SCENARIOS:
        result = service.search_authentication_events(scenario_id=scenario_id)
        claims = (result.summary + " " + " ".join(result.warnings)).lower()
        assert not any(word in claims for word in ("compromised", "attacker", "breach", "malicious", "brute"))
    # The limitations disclose, rather than assert, what the tool does not do.
    disclaimer = " ".join(service.search_authentication_events(account="x").limitations)
    assert "do not establish compromise" in disclaimer and "no IP reputation" in disclaimer.lower().replace("no ip", "no IP")


def test_ae21_no_external_ip_reputation_lookup_is_performed(monkeypatch, service):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    result = service.search_authentication_events(source="203.0.113.25")
    assert result.status is ToolStatus.SUCCESS and "No IP reputation" in " ".join(result.limitations)


def test_ae22_input_event_data_remains_unchanged(data_copy):
    before = source_hashes(data_copy)
    service = service_for(data_copy)
    for criteria in ({"account": "admin.synthetic01"}, {"success": False}, {"scenario_id": "C"}, {"source": "x"}, {}):
        service.search_authentication_events(**criteria)
    assert source_hashes(data_copy) == before


def test_ae23_every_returned_source_id_resolves_to_a_real_event_record(service):
    real_ids = {e["event_id"] for e in RAW_EVENTS}
    for criteria in ({"account": "admin.synthetic01"}, {"scenario_id": "B"}, {"asset_id": "ASSET-PAY-01"}, {"success": True}):
        result = service.search_authentication_events(**criteria)
        assert set(result.source_record_ids) <= real_ids
        assert result.source_record_ids == tuple(e.event_id for e in result.evidence.events)
        assert len(set(result.source_record_ids)) == len(result.source_record_ids)


def test_ae24_the_summary_can_be_derived_from_the_returned_evidence(service):
    result = service.search_authentication_events(scenario_id="B")
    evidence = result.evidence
    assert evidence.failed_login_count == sum(1 for e in evidence.events if not e.success)
    assert evidence.successful_login_count == sum(1 for e in evidence.events if e.success)
    assert f"{evidence.matched_count} synthetic authentication event(s) matched" in result.summary
    assert f"{evidence.failed_login_count} failed" in result.summary


# ===========================================================================
# get_asset_criticality (AC-01 .. AC-10)
# ===========================================================================


def test_ac01_known_asset_returns_success(service):
    result = service.get_asset_criticality("ASSET-IDP-01")
    assert result.status is ToolStatus.SUCCESS and result.record_found is True


def test_ac02_scenario_c_asset_is_critical(service):
    result = service.get_asset_criticality(RAW_SCENARIOS["C"]["asset_id"])
    assert result.evidence.criticality.value == "CRITICAL" and "criticality CRITICAL" in result.summary


@pytest.mark.parametrize("asset_id", sorted(RAW_ASSETS))
def test_ac03_ac07_returned_facts_match_the_source_and_carry_its_id(service, asset_id):
    raw, result = RAW_ASSETS[asset_id], service.get_asset_criticality(asset_id)
    assert (result.evidence.asset_name, result.evidence.asset_type, result.evidence.criticality.value,
            result.evidence.description) == (raw["asset_name"], raw["asset_type"], raw["criticality"], raw["description"])
    assert result.source_record_ids == (asset_id,) and result.evidence.business_impact_summary is None


def test_ac04_unknown_asset_returns_not_found(service):
    result = service.get_asset_criticality("ASSET-NOPE")
    assert result.status is ToolStatus.NOT_FOUND and result.evidence is None and result.error.error_type == "NotFound"


@pytest.mark.parametrize("asset_id", ["", "  "])
def test_ac05_blank_asset_id_is_invalid_input(service, asset_id):
    assert service.get_asset_criticality(asset_id).status is ToolStatus.INVALID_INPUT


def test_ac06_duplicate_asset_ids_return_ambiguous(data_copy):
    rewrite(data_copy, "assets.json", lambda d: d.append({**d[2], "criticality": "LOW"}))
    result = service_for(data_copy).get_asset_criticality("ASSET-PAY-01")
    assert result.status is ToolStatus.AMBIGUOUS and result.evidence is None


def test_ac08_ac09_criticality_is_not_turned_into_an_incident_status_or_decision(service):
    result = service.get_asset_criticality("ASSET-PAY-01")
    text = json.dumps(result.model_dump(mode="json")).lower()
    assert not any(word in text for word in ("confirmed", "compromised", "block", "escalate now", "must be"))
    assert not {"decision", "priority", "incident_status"} & set(result.evidence.model_dump())
    assert "not translated into an incident status" in " ".join(result.limitations)


def test_ac10_asset_source_data_is_never_modified(data_copy):
    before = source_hashes(data_copy)
    service = service_for(data_copy)
    for asset_id in (*RAW_ASSETS, "NOPE", ""):
        service.get_asset_criticality(asset_id)
    assert source_hashes(data_copy) == before


# ===========================================================================
# get_response_policy (RP-01 .. RP-15)
# ===========================================================================


@pytest.mark.parametrize("policy_id", sorted(RAW_POLICIES))
def test_rp01_rp02_rp15_exact_policy_lookup_matches_the_source(service, policy_id):
    raw, result = RAW_POLICIES[policy_id], service.get_response_policy(policy_id=policy_id)
    assert result.status is ToolStatus.SUCCESS and result.source_record_ids == (policy_id,)
    evidence = result.evidence
    assert evidence.policy_name == raw["title"] and evidence.description == raw["description"]
    assert evidence.recommended_investigation_actions == tuple(raw["recommended_actions"])
    assert evidence.escalation_criteria == raw["escalation_guidance"]
    assert evidence.restricted_actions == tuple(raw["prohibited_automated_actions"])
    assert evidence.restriction_statement == raw["restriction"]
    assert policy_id in RAW_POLICIES  # the returned source ID resolves to a real record


@pytest.mark.parametrize("policy_id", sorted(RAW_POLICIES))
def test_rp03_rp04_rp13_rp14_every_policy_forbids_automation_and_requires_human_authorization(service, policy_id):
    evidence = service.get_response_policy(policy_id=policy_id).evidence
    assert evidence.automatic_disruptive_action_permitted is False and evidence.human_authorization_required is True
    assert {"block_user", "disable_account", "modify_firewall_rule", "terminate_process", "quarantine_host",
            "offensive_response"} <= set(evidence.restricted_actions)
    assert "human authorization" in evidence.restriction_statement
    assert "not permitted" in evidence.restriction_statement


def test_rp05_policy_contains_defensive_investigation_guidance(service):
    actions = " ".join(service.get_response_policy(policy_id="POL-PRIV-LOGIN-001").evidence.recommended_investigation_actions)
    assert "Review authentication activity" in actions and "Validate whether the source address is expected" in actions
    assert "Escalate to a security analyst" in actions


def test_rp06_policy_contains_restricted_actions(service):
    assert len(service.get_response_policy(policy_id="POL-LOW-SEVERITY-001").evidence.restricted_actions) >= 6


def test_rp07_unknown_policy_id_returns_not_found(service):
    result = service.get_response_policy(policy_id="POL-NOPE")
    assert result.status is ToolStatus.NOT_FOUND and result.evidence is None and result.record_found is False


@pytest.mark.parametrize("selectors", [{}, {"policy_id": "  "}, {"alert_type": ""}, {"policy_id": " ", "alert_type": " "}])
def test_rp08_blank_only_selector_input_is_rejected(service, selectors):
    assert service.get_response_policy(**selectors).status is ToolStatus.INVALID_INPUT


def test_invalid_severity_value_is_rejected_not_coerced(service):
    assert service.get_response_policy(severity="SEVERE").status is ToolStatus.INVALID_INPUT


@pytest.mark.parametrize("selectors, expected", [
    ({"alert_type": "repeated_failed_login"}, "POL-PRIV-LOGIN-001"),
    ({"alert_type": "minor_authentication_anomaly"}, "POL-LOW-SEVERITY-001"),
    ({"severity": "LOW"}, "POL-LOW-SEVERITY-001"),
    ({"asset_criticality": "CRITICAL"}, "POL-CORRELATED-CRITICAL-001"),
    ({"severity": "HIGH", "asset_criticality": "HIGH"}, "POL-PRIV-LOGIN-001"),
    ({"alert_type": "correlated_service_anomaly", "severity": "HIGH", "asset_criticality": "CRITICAL"}, "POL-CORRELATED-CRITICAL-001"),
])
def test_rp09_unambiguous_context_lookup_returns_the_correct_policy(service, selectors, expected):
    result = service.get_response_policy(**selectors)
    assert result.status is ToolStatus.SUCCESS and result.source_record_ids == (expected,)
    assert result.evidence.applies_to  # the context it was matched on is reported back


def test_rp10_rp11_ambiguous_context_returns_ambiguous_and_never_the_first_match(service):
    result = service.get_response_policy(severity="HIGH")  # A's and C's policies both apply to HIGH alerts
    assert result.status is ToolStatus.AMBIGUOUS and result.evidence is None and result.source_record_ids == ()
    assert result.candidate_policy_ids == ("POL-CORRELATED-CRITICAL-001", "POL-PRIV-LOGIN-001")
    assert result.error.error_type == "Ambiguous" and "2" in result.warnings[0]


def test_rp11_duplicate_policy_ids_are_ambiguous_too(data_copy):
    rewrite(data_copy, "policies.json", lambda d: d.append({**d[0], "title": "Conflicting duplicate"}))
    result = service_for(data_copy).get_response_policy(policy_id="POL-PRIV-LOGIN-001")
    assert result.status is ToolStatus.AMBIGUOUS and result.evidence is None


def test_context_with_no_matching_policy_returns_not_found(service):
    result = service.get_response_policy(alert_type="never_seen_alert_type")
    assert result.status is ToolStatus.NOT_FOUND and result.candidate_policy_ids == ()


def test_policy_id_takes_precedence_over_context_selectors_with_a_warning(service):
    result = service.get_response_policy(policy_id="POL-PRIV-LOGIN-001", severity="LOW")
    assert result.source_record_ids == ("POL-PRIV-LOGIN-001",) and "ignored" in result.warnings[0]


def test_rp12_policy_tool_executes_no_recommended_action(service):
    result = service.get_response_policy(policy_id="POL-CORRELATED-CRITICAL-001")
    assert result.executed_actions == () and result.is_read_only is True
    assert "no recommended action is executed" in " ".join(result.limitations)


def test_policy_applies_to_is_derived_from_scenarios_and_disclosed(service):
    result = service.get_response_policy(policy_id="POL-CORRELATED-CRITICAL-001")
    context = result.evidence.applies_to[0]
    assert (context.alert_type, context.severity.value, context.asset_criticality.value) == (
        "correlated_service_anomaly", "HIGH", "CRITICAL")
    assert "derived from the synthetic scenarios" in " ".join(result.limitations)


# ===========================================================================
# Cross-source consistency (CS-01 .. CS-10)
# ===========================================================================


@pytest.mark.parametrize("scenario_id", sorted(RAW_SCENARIOS))
def test_cs01_cs02_cs03_cs08_every_scenario_reference_resolves_through_the_tools(service, scenario_id):
    scenario = RAW_SCENARIOS[scenario_id]
    assert service.get_alert_details(scenario["alert_id"]).status is ToolStatus.SUCCESS
    assert service.get_asset_criticality(scenario["asset_id"]).status is ToolStatus.SUCCESS
    assert service.get_response_policy(policy_id=scenario["policy_id"]).status is ToolStatus.SUCCESS


@pytest.mark.parametrize("scenario_id", sorted(RAW_SCENARIOS))
def test_cs04_cs09_tool_evidence_does_not_contradict_the_scenario(service, scenario_id):
    scenario = RAW_SCENARIOS[scenario_id]
    alert = service.get_alert_details(scenario["alert_id"]).evidence
    events = service.search_authentication_events(alert_id=scenario["alert_id"]).evidence
    asset = service.get_asset_criticality(scenario["asset_id"]).evidence
    assert events.failed_login_count == scenario["failed_login_count"]
    assert (alert.alert_type, alert.severity.value, alert.target, alert.source) == (
        scenario["alert_type"], scenario["severity"], scenario["target_account"], scenario["source"])
    assert asset.criticality.value == scenario["asset_criticality"] and asset.asset_name == scenario["asset_name"]
    assert scenario["confirmed_impact"] is False and alert.confirmed_impact is None


def test_cs04_scenario_a_authentication_evidence_supports_the_failed_login_count(service):
    events = service.search_authentication_events(account="admin.synthetic01", success=False, source="203.0.113.25").evidence
    assert events.matched_count == RAW_SCENARIOS["A"]["failed_login_count"] == 12
    assert service.search_authentication_events(scenario_id="A").evidence.successful_login_count == 0


def test_cs05_scenario_b_evidence_remains_limited_and_unconfirmed(service):
    alert, events = service.get_alert_details("ALERT-B-001").evidence, service.search_authentication_events(scenario_id="B").evidence
    assert alert.severity.value == "LOW" and events.matched_count == 2 and events.failed_login_count == 1
    assert events.successful_login_count == 1  # a failed login followed by a success: inconclusive
    assert alert.confirmed_impact is None


def test_cs06_scenario_c_asset_is_critical(service):
    assert service.get_asset_criticality(RAW_SCENARIOS["C"]["asset_id"]).evidence.criticality.value == "CRITICAL"


def test_cs07_scenario_c_correlated_and_unusual_access_is_represented_across_records(service):
    alert = service.get_alert_details("ALERT-C-001").evidence
    events = service.search_authentication_events(scenario_id="C").evidence
    assert "correlated alerts" in alert.summary and "off-hours access" in alert.summary
    assert any("outside its normal operating window" in item for item in alert.evidence_items)
    assert {e.event_type for e in events.events} >= {"off_hours_service_access", "login_failure"}
    assert RAW_SCENARIOS["C"]["correlated_alert_count"] == 3 and RAW_SCENARIOS["C"]["unusual_access"] is True


def test_committed_data_has_no_consistency_issues(service):
    report = service.check_consistency()
    assert report.consistent is True and report.issues == ()


def test_cs10_deliberately_introduced_inconsistencies_are_detected(data_copy):
    service = service_for(data_copy)
    assert service.check_consistency().consistent
    rewrite(data_copy, "authentication_events.json", lambda d: d.append({**d[0], "event_id": "AUTH-X-1", "asset_id": "ASSET-GHOST"}))
    rewrite(data_copy, "scenarios.json", lambda d: d[0].update({"failed_login_count": 3}))
    rewrite(data_copy, "assets.json", lambda d: d[2].update({"criticality": "LOW"}))
    rewrite(data_copy, "policies.json", lambda d: d.append({**d[0]}))
    issues = " | ".join(service.check_consistency().issues)
    assert "references missing asset ASSET-GHOST" in issues
    assert "failed_login_count differs" in issues
    assert "asset criticality differs" in issues
    assert "Duplicate policy ID: POL-PRIV-LOGIN-001" in issues


def test_consistency_check_flags_dangling_references(data_copy):
    rewrite(data_copy, "scenarios.json", lambda d: d[1].update({"policy_id": "POL-GHOST", "alert_id": "ALERT-GHOST"}))
    issues = " | ".join(service_for(data_copy).check_consistency().issues)
    assert "missing alert ALERT-GHOST" in issues and "missing policy POL-GHOST" in issues


# ===========================================================================
# Error handling (ERROR status, safe messages, developer logging)
# ===========================================================================


@pytest.mark.parametrize("breaker, cause", [
    (lambda d: (d / "alert_details.json").write_text("{ not json", encoding="utf-8"), "MalformedDataError"),
    (lambda d: (d / "alert_details.json").unlink(), "DataFileNotFoundError"),
    (lambda d: (d / "alert_details.json").write_text("{}", encoding="utf-8"), "UnexpectedStructureError"),
    (lambda d: rewrite(d, "alert_details.json", lambda data: data[0].update({"severity": "SEVERE"})), "ValidationError"),
])
def test_unexpected_data_failures_return_a_safe_error_and_log_the_cause(data_copy, caplog, breaker, cause):
    breaker(data_copy)
    with caplog.at_level(logging.ERROR, logger="security_tools"):
        result = service_for(data_copy).get_alert_details("ALERT-A-001")
    assert result.status is ToolStatus.ERROR and result.evidence is None and result.record_found is False
    assert result.error.error_type == cause
    dumped = json.dumps(result.model_dump(mode="json"))
    assert str(data_copy) not in dumped and "Traceback" not in dumped and "tmp" not in result.error.message.lower()
    assert result.error.message == "Source data for alert details could not be read or validated."
    # Developers still get the full chained cause in the log.
    record = next(r for r in caplog.records if r.name == "security_tools")
    assert record.exc_info and type(record.exc_info[1]).__name__ == "ToolDataError"
    assert type(record.exc_info[1].__cause__).__name__ == cause


def test_malformed_events_file_is_an_error_never_a_silent_empty_success(data_copy):
    (data_copy / "authentication_events.json").write_text("[{ broken", encoding="utf-8")
    result = service_for(data_copy).search_authentication_events(account="admin.synthetic01")
    assert result.status is ToolStatus.ERROR and result.evidence is None


def test_invalid_stored_record_is_an_error_for_every_tool(data_copy):
    rewrite(data_copy, "assets.json", lambda d: d[0].update({"criticality": "EXTREME"}))
    rewrite(data_copy, "policies.json", lambda d: d[0].update({"automated_actions_permitted": True}))
    service = service_for(data_copy)
    assert service.get_asset_criticality("ASSET-IDP-01").status is ToolStatus.ERROR
    assert service.get_response_policy(policy_id="POL-PRIV-LOGIN-001").status is ToolStatus.ERROR


def test_check_consistency_raises_a_clear_exception_for_unreadable_data(data_copy):
    (data_copy / "assets.json").write_text("nope", encoding="utf-8")
    with pytest.raises(ToolDataError) as info:
        service_for(data_copy).check_consistency()
    assert info.value.__cause__ is not None


def test_a_failure_in_one_source_does_not_affect_tools_that_do_not_use_it(data_copy):
    (data_copy / "assets.json").write_text("nope", encoding="utf-8")
    service = service_for(data_copy)
    assert service.get_asset_criticality("ASSET-IDP-01").status is ToolStatus.ERROR
    assert service.get_alert_details("ALERT-A-001").status is ToolStatus.SUCCESS


# ===========================================================================
# Invocation audit (IA-01 .. IA-10)
# ===========================================================================


def run_four_tools(service: SecurityToolService) -> None:
    service.get_alert_details("ALERT-A-001")
    service.search_authentication_events(account="admin.synthetic01", success=False)
    service.get_asset_criticality("ASSET-IDP-01")
    service.get_response_policy(policy_id="POL-PRIV-LOGIN-001")


def test_ia01_ia03_every_call_creates_one_record_in_invocation_order(service):
    run_four_tools(service)
    log = service.get_invocation_log()
    assert [r.tool_name for r in log] == ["get_alert_details", "search_authentication_events",
                                          "get_asset_criticality", "get_response_policy"]
    assert [r.started_at for r in log] == sorted(r.started_at for r in log)


def test_ia01_failed_and_invalid_calls_are_recorded_too(service):
    service.get_alert_details("NOPE")
    service.get_alert_details(" ")
    service.get_response_policy(severity="HIGH")
    assert [r.status for r in service.get_invocation_log()] == [ToolStatus.NOT_FOUND, ToolStatus.INVALID_INPUT, ToolStatus.AMBIGUOUS]
    assert [r.error_type for r in service.get_invocation_log()] == ["NotFound", "InvalidInput", "Ambiguous"]


def test_ia02_call_ids_are_unique_and_match_the_results(service):
    results = [service.get_alert_details("ALERT-A-001") for _ in range(5)]
    ids = [r.call_id for r in results]
    assert len(set(ids)) == 5 and [r.call_id for r in service.get_invocation_log()] == ids


def test_ia04_ia05_record_status_evidence_count_and_sources_match_the_result(service):
    results = [service.get_alert_details("ALERT-A-001"), service.search_authentication_events(scenario_id="A"),
               service.get_asset_criticality("NOPE"), service.get_response_policy(policy_id="POL-PRIV-LOGIN-001")]
    for result, record in zip(results, service.get_invocation_log(), strict=True):
        assert record.status is result.status and record.call_id == result.call_id
        assert record.evidence_count == result.evidence_count and record.source_record_ids == result.source_record_ids
        assert record.warnings == result.warnings and record.request == result.query
    assert [r.evidence_count for r in service.get_invocation_log()] == [1, 12, 0, 1]


def test_ia_records_are_timezone_aware_and_durations_are_non_negative(service):
    run_four_tools(service)
    for record in service.get_invocation_log():
        assert record.started_at.tzinfo is not None and record.completed_at >= record.started_at
        assert record.duration_ms >= 0


def test_ia06_the_log_snapshot_cannot_mutate_the_internal_log(service):
    run_four_tools(service)
    snapshot = service.get_invocation_log()
    snapshot[0].request["alert_id"] = "TAMPERED"
    assert service.get_invocation_log()[0].request["alert_id"] == "ALERT-A-001"
    assert isinstance(snapshot, tuple)
    with pytest.raises(TypeError):
        snapshot[0] = snapshot[1]  # type: ignore[index]


def test_ia07_separate_services_do_not_share_logs():
    first, second = SecurityToolService(), SecurityToolService()
    first.get_alert_details("ALERT-A-001")
    assert len(first.get_invocation_log()) == 1 and second.get_invocation_log() == ()


def test_ia08_reset_clears_only_the_selected_services_log():
    first, second = SecurityToolService(), SecurityToolService()
    run_four_tools(first)
    run_four_tools(second)
    first.reset_invocation_log()
    assert first.get_invocation_log() == () and len(second.get_invocation_log()) == 4


def test_ia09_the_log_is_bounded():
    service = SecurityToolService(max_log_entries=3)
    results = [service.get_alert_details("ALERT-A-001") for _ in range(5)]
    assert [r.call_id for r in service.get_invocation_log()] == [r.call_id for r in results[-3:]]
    default = SecurityToolService()
    for _ in range(105):
        default.get_asset_criticality("ASSET-IDP-01")
    assert len(default.get_invocation_log()) == 100
    for bad in (0, 101, -1):
        with pytest.raises(ValueError, match="max_log_entries"):
            SecurityToolService(max_log_entries=bad)


def test_ia10_audit_records_contain_no_secrets_paths_or_stack_traces(data_copy):
    service = service_for(data_copy)
    run_four_tools(service)
    (data_copy / "assets.json").write_text("nope", encoding="utf-8")
    service.get_asset_criticality("ASSET-IDP-01")  # an ERROR call is audited too
    text = json.dumps([r.model_dump(mode="json") for r in service.get_invocation_log()]).lower()
    for forbidden in ("password", "secret", "token", "api_key", "credential", "traceback", "c:\\\\", str(data_copy).lower().replace("\\", "\\\\")):
        assert forbidden not in text


def test_injected_clock_and_id_factory_make_audit_records_deterministic(data_copy):
    moments = iter(datetime(2026, 1, 1, 0, 0, i, tzinfo=UTC) for i in range(10))
    ids = iter(f"CALL-{n:04d}" for n in range(10))
    service = service_for(data_copy, clock=lambda: next(moments), id_factory=lambda: next(ids))
    first, second = service.get_alert_details("ALERT-A-001"), service.get_asset_criticality("ASSET-IDP-01")
    assert (first.call_id, second.call_id) == ("CALL-0000", "CALL-0001")
    assert service.get_invocation_log()[0].started_at == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert first.timestamp >= datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
