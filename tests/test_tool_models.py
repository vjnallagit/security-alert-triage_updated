import json
from datetime import UTC, datetime, timedelta

import pytest
from models.schemas import AuthenticationEvent
from models.tool_models import (
    TOOL_CATALOGUE,
    AlertEvidence,
    AssetEvidence,
    AuthenticationSearchEvidence,
    GetAlertDetailsInput,
    GetAlertDetailsResult,
    GetAssetCriticalityInput,
    GetAssetCriticalityResult,
    GetResponsePolicyInput,
    GetResponsePolicyResult,
    PolicyEvidence,
    SearchAuthenticationEventsInput,
    SearchAuthenticationEventsResult,
    ToolError,
    ToolInvocationRecord,
    ToolStatus,
    get_tool_catalogue,
)
from pydantic import ValidationError

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
LIMITATIONS = ("Synthetic data only.",)


def alert_evidence(**overrides) -> AlertEvidence:
    data = dict(alert_id="ALERT-X", timestamp=NOW, alert_type="t", severity="LOW", summary="s",
                target="acct", source="203.0.113.1", evidence_items=("e",))
    return AlertEvidence(**{**data, **overrides})


def policy_evidence(**overrides) -> PolicyEvidence:
    data = dict(policy_id="POL-X", policy_name="n", description="d", recommended_investigation_actions=("Review.",),
                escalation_criteria="c", restricted_actions=("block_user",), restriction_statement="r",
                human_authorization_required=True, automatic_disruptive_action_permitted=False)
    return PolicyEvidence(**{**data, **overrides})


def alert_result(**overrides) -> GetAlertDetailsResult:
    data = dict(call_id="CALL-1", status=ToolStatus.SUCCESS, query={"alert_id": "ALERT-X"}, evidence=alert_evidence(),
                summary="Alert ALERT-X.", source_type="alert_details", source_record_ids=("ALERT-X",),
                limitations=LIMITATIONS, timestamp=NOW)
    return GetAlertDetailsResult(**{**data, **overrides})


def not_found_result(**overrides) -> GetAlertDetailsResult:
    data = dict(call_id="CALL-2", status=ToolStatus.NOT_FOUND, query={"alert_id": "NOPE"}, summary="No alert.",
                source_type="alert_details", limitations=LIMITATIONS, timestamp=NOW,
                error=ToolError(error_type="NotFound", message="No synthetic alert with that ID was found."))
    return GetAlertDetailsResult(**{**data, **overrides})


def invocation(**overrides) -> ToolInvocationRecord:
    data = dict(call_id="CALL-1", tool_name="get_alert_details", started_at=NOW, completed_at=NOW + timedelta(milliseconds=5),
                duration_ms=5.0, request={"alert_id": "ALERT-X"}, status=ToolStatus.SUCCESS,
                source_record_ids=("ALERT-X",), evidence_count=1)
    return ToolInvocationRecord(**{**data, **overrides})


@pytest.mark.parametrize("model, field, kwargs", [
    (GetAlertDetailsInput, "alert_id", {}),
    (GetAssetCriticalityInput, "asset_id", {}),
])
@pytest.mark.parametrize("blank", ["", "   "])
def test_tm01_blank_required_ids_are_rejected(model, field, kwargs, blank):
    with pytest.raises(ValidationError):
        model(**{field: blank, **kwargs})


@pytest.mark.parametrize("blank", ["", "  "])
def test_tm01_blank_optional_selectors_are_rejected_when_supplied(blank):
    with pytest.raises(ValidationError):
        GetResponsePolicyInput(policy_id=blank)
    with pytest.raises(ValidationError):
        SearchAuthenticationEventsInput(account=blank)


@pytest.mark.parametrize("model, kwargs", [
    (GetAlertDetailsInput, {"alert_id": "A"}),
    (GetAssetCriticalityInput, {"asset_id": "A"}),
    (GetResponsePolicyInput, {"policy_id": "P"}),
    (SearchAuthenticationEventsInput, {"account": "a"}),
])
def test_tm02_unexpected_fields_are_rejected(model, kwargs):
    assert model(**kwargs)
    with pytest.raises(ValidationError):
        model(**kwargs, run_remediation=True)


def test_tm03_timestamps_must_be_timezone_aware():
    naive = datetime(2026, 3, 10, 12, 0)
    with pytest.raises(ValidationError):
        SearchAuthenticationEventsInput(account="a", start_time=naive)
    with pytest.raises(ValidationError):
        SearchAuthenticationEventsInput(account="a", end_time=naive)
    with pytest.raises(ValidationError):
        alert_evidence(timestamp=naive)
    with pytest.raises(ValidationError):
        alert_result(timestamp=naive)
    with pytest.raises(ValidationError):
        invocation(started_at=naive)


def test_tm04_status_only_accepts_defined_values():
    assert {s.value for s in ToolStatus} == {"SUCCESS", "NOT_FOUND", "INVALID_INPUT", "AMBIGUOUS", "ERROR"}
    with pytest.raises(ValidationError):
        alert_result(status="PARTIAL")
    with pytest.raises(ValidationError):
        invocation(status="OK")


def test_tm05_successful_result_cannot_contain_a_fatal_error():
    with pytest.raises(ValidationError, match="successful result cannot carry an error"):
        alert_result(error=ToolError(error_type="X", message="boom"))
    with pytest.raises(ValidationError, match="must carry evidence"):
        alert_result(evidence=None)


@pytest.mark.parametrize("status", [ToolStatus.NOT_FOUND, ToolStatus.INVALID_INPUT, ToolStatus.AMBIGUOUS, ToolStatus.ERROR])
def test_tm06_failed_result_needs_a_safe_error_explanation(status):
    assert not_found_result(status=status).error.message
    with pytest.raises(ValidationError, match="must explain the failure"):
        not_found_result(status=status, error=None)
    with pytest.raises(ValidationError, match="must not carry evidence"):
        not_found_result(status=status, evidence=alert_evidence())
    with pytest.raises(ValidationError, match="must not claim source records"):
        not_found_result(status=status, source_record_ids=("ALERT-X",))


@pytest.mark.parametrize("message", [
    "Failed reading C:\\Users\\example\\data\\alerts.json", "Failed at /Users/example/data.json",
    "Traceback (most recent call last):", 'File "x.py", line 3',
])
def test_tm06_error_messages_cannot_contain_paths_or_stack_traces(message):
    with pytest.raises(ValidationError, match="paths or stack traces"):
        ToolError(error_type="E", message=message)


def test_tm07_executed_actions_must_be_empty():
    assert alert_result().executed_actions == ()
    with pytest.raises(ValidationError):
        alert_result(executed_actions=("block_user",))


def test_tm08_is_synthetic_must_remain_true():
    with pytest.raises(ValidationError):
        alert_result(is_synthetic=False)


def test_tm09_is_read_only_must_remain_true():
    with pytest.raises(ValidationError):
        alert_result(is_read_only=False)


def test_tm10_authentication_search_requires_at_least_one_criterion():
    with pytest.raises(ValidationError, match="at least one search criterion"):
        SearchAuthenticationEventsInput()
    with pytest.raises(ValidationError, match="at least one search criterion"):
        SearchAuthenticationEventsInput(limit=5)  # a limit alone is not a criterion
    for criterion in ({"account": "a"}, {"source": "s"}, {"asset_id": "x"}, {"alert_id": "x"}, {"scenario_id": "A"},
                      {"success": False}, {"start_time": NOW}, {"end_time": NOW}):
        assert SearchAuthenticationEventsInput(**criterion)


@pytest.mark.parametrize("limit", [0, -1, 101, 1000])
def test_tm11_limit_rejects_zero_and_values_over_100(limit):
    with pytest.raises(ValidationError):
        SearchAuthenticationEventsInput(account="a", limit=limit)


@pytest.mark.parametrize("limit", [1, 50, 100])
def test_tm11_limit_accepts_the_valid_range(limit):
    assert SearchAuthenticationEventsInput(account="a", limit=limit).limit == limit
    assert SearchAuthenticationEventsInput(account="a").limit == 100


def test_tm12_invalid_time_range_is_rejected():
    with pytest.raises(ValidationError, match="must not be after"):
        SearchAuthenticationEventsInput(account="a", start_time=NOW, end_time=NOW - timedelta(seconds=1))
    assert SearchAuthenticationEventsInput(account="a", start_time=NOW, end_time=NOW)


def test_tm13_policy_lookup_requires_at_least_one_selector():
    with pytest.raises(ValidationError, match="at least one selector"):
        GetResponsePolicyInput()
    for selector in ({"policy_id": "P"}, {"alert_type": "t"}, {"severity": "HIGH"}, {"asset_criticality": "CRITICAL"}):
        assert GetResponsePolicyInput(**selector)
    with pytest.raises(ValidationError):
        GetResponsePolicyInput(severity="SEVERE")


def test_tm14_results_serialize_cleanly_to_json():
    for result in (alert_result(), not_found_result()):
        text = json.dumps(result.model_dump(mode="json"))
        assert json.loads(text)["is_read_only"] is True
    assert json.dumps(alert_result().to_llm_evidence())
    assert alert_result().model_dump(mode="json")["record_found"] is True
    assert not_found_result().model_dump(mode="json")["record_found"] is False


def test_tm15_source_record_ids_contain_no_blank_values():
    for bad in ("", "   "):
        with pytest.raises(ValidationError, match="must not be blank"):
            alert_result(source_record_ids=("ALERT-X", bad))


def test_tm16_invocation_duration_cannot_be_negative():
    with pytest.raises(ValidationError):
        invocation(duration_ms=-0.5)


def test_tm17_invocation_timestamps_are_internally_consistent():
    with pytest.raises(ValidationError, match="must not be before"):
        invocation(completed_at=NOW - timedelta(seconds=1))
    with pytest.raises(ValidationError, match="must match the difference"):
        invocation(duration_ms=500.0)
    assert invocation(duration_ms=5.4)  # within the 1 ms tolerance


def test_tm18_catalogue_contains_exactly_the_four_phase_6_tools():
    assert [t.name for t in TOOL_CATALOGUE] == [
        "get_alert_details", "search_authentication_events", "get_asset_criticality", "get_response_policy",
    ]
    assert get_tool_catalogue() is TOOL_CATALOGUE and len({t.name for t in TOOL_CATALOGUE}) == 4


def test_tm19_every_catalogue_entry_is_read_only_and_synthetic():
    for tool in TOOL_CATALOGUE:
        assert tool.read_only is True and tool.synthetic is True
        assert tool.safety_classification == "READ_ONLY_SYNTHETIC_EVIDENCE_RETRIEVAL"
        with pytest.raises(ValidationError):
            tool.model_copy(update={}).__class__(**{**tool.model_dump(), "read_only": False})


def test_tm20_catalogue_descriptions_explain_use_and_limits():
    for tool in TOOL_CATALOGUE:
        assert len(tool.description) > 60 and tool.allowed_use and tool.prohibited_use
        assert "does not" in tool.description  # states what the tool does NOT do
        assert tool.input_schema()["properties"]
        assert any("final triage decision" in p for p in tool.prohibited_use)
    names = {t.name: t for t in TOOL_CATALOGUE}
    assert "alert_id" in names["get_alert_details"].description
    assert "asset_id" in names["get_asset_criticality"].description
    assert "at least one" in names["search_authentication_events"].description
    assert "never permits automatic disruptive action" in names["get_response_policy"].description


def test_catalogue_binds_each_tool_to_its_input_and_output_models():
    pairs = {t.name: (t.input_model, t.output_model) for t in TOOL_CATALOGUE}
    assert pairs["get_alert_details"] == (GetAlertDetailsInput, GetAlertDetailsResult)
    assert pairs["search_authentication_events"] == (SearchAuthenticationEventsInput, SearchAuthenticationEventsResult)
    assert pairs["get_asset_criticality"] == (GetAssetCriticalityInput, GetAssetCriticalityResult)
    assert pairs["get_response_policy"] == (GetResponsePolicyInput, GetResponsePolicyResult)


def test_policy_evidence_cannot_permit_automatic_disruptive_action():
    with pytest.raises(ValidationError):
        policy_evidence(automatic_disruptive_action_permitted=True)
    with pytest.raises(ValidationError):
        policy_evidence(human_authorization_required=False)
    with pytest.raises(ValidationError):
        policy_evidence(restricted_actions=())


def event(event_id: str, success: bool, source: str = "203.0.113.25", minutes: int = 0) -> AuthenticationEvent:
    return AuthenticationEvent(event_id=event_id, alert_id="ALERT-X", timestamp=NOW + timedelta(minutes=minutes),
                               account="acct", source=source, event_type="login_failure", success=success, asset_id="ASSET-X")


def test_search_evidence_enforces_consistent_counts():
    events = (event("E1", False), event("E2", True))
    ok = dict(matched_count=2, returned_count=2, events=events, failed_login_count=1, successful_login_count=1,
              unique_sources=("203.0.113.25",), truncated=False)
    assert AuthenticationSearchEvidence(**ok)
    for bad in (dict(returned_count=3), dict(truncated=True), dict(failed_login_count=2), dict(matched_count=1)):
        with pytest.raises(ValidationError):
            AuthenticationSearchEvidence(**{**ok, **bad})


def test_search_result_sources_must_be_exactly_the_returned_events():
    events = (event("E1", False),)
    evidence = AuthenticationSearchEvidence(matched_count=1, returned_count=1, events=events, failed_login_count=1,
                                            successful_login_count=0, unique_sources=("203.0.113.25",), truncated=False)
    base = dict(call_id="C", status=ToolStatus.SUCCESS, query={"account": "acct"}, evidence=evidence, summary="s",
                source_type="authentication_events", limitations=LIMITATIONS, timestamp=NOW)
    assert SearchAuthenticationEventsResult(**base, source_record_ids=("E1",))
    with pytest.raises(ValidationError, match="exactly the returned event IDs"):
        SearchAuthenticationEventsResult(**base, source_record_ids=("E1", "E9"))


def test_results_are_immutable_and_to_llm_evidence_is_compact():
    result = alert_result()
    with pytest.raises(ValidationError):
        result.status = ToolStatus.ERROR
    compact = result.to_llm_evidence()
    assert set(compact) == {"tool_name", "status", "summary", "evidence", "source_record_ids", "warnings",
                            "limitations", "error"}
    assert compact["status"] == "SUCCESS" and compact["source_record_ids"] == ["ALERT-X"]


def test_results_round_trip_through_their_own_dump():
    """The derived record_found field is emitted by model_dump but must not break model_validate."""
    for original in (alert_result(), not_found_result()):
        for dump in (original.model_dump(), original.model_dump(mode="json")):
            assert dump["record_found"] is original.record_found
            assert type(original).model_validate(dump) == original
    with pytest.raises(ValidationError):  # other unknown fields are still rejected
        GetAlertDetailsResult.model_validate({**alert_result().model_dump(), "remediated": True})


def test_asset_result_model_round_trips():
    evidence = AssetEvidence(asset_id="ASSET-X", asset_name="n", asset_type="t", criticality="CRITICAL", description="d")
    result = GetAssetCriticalityResult(call_id="C", status=ToolStatus.SUCCESS, query={"asset_id": "ASSET-X"},
                                       evidence=evidence, summary="s", source_type="assets",
                                       source_record_ids=("ASSET-X",), limitations=LIMITATIONS, timestamp=NOW)
    assert result.evidence_count == 1 and result.record_found is True
    policy = GetResponsePolicyResult(call_id="C", status=ToolStatus.SUCCESS, query={"policy_id": "POL-X"},
                                     evidence=policy_evidence(), summary="s", source_type="policies",
                                     source_record_ids=("POL-X",), limitations=LIMITATIONS, timestamp=NOW)
    assert policy.candidate_policy_ids == () and policy.record_found is True
