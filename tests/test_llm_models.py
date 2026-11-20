import json
from datetime import UTC, datetime, timedelta

import pytest
from models.llm_models import (
    AssertedFacts,
    CallStatus,
    CompactLLMTriageDecision,
    GroundedEvidenceItem,
    GuardrailResult,
    LLMExecutionTrace,
    LLMToolCallRecord,
    LLMTriageRecommendation,
    RuntimeMode,
    ToolObservation,
)
from pydantic import ValidationError

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)


def item(**overrides) -> GroundedEvidenceItem:
    data = dict(claim="Alert ALERT-A-001 is HIGH severity.", tool_name="get_alert_details", tool_call_id="call_1",
                source_record_ids=("ALERT-A-001",))
    return GroundedEvidenceItem(**{**data, **overrides})


def recommendation(**overrides) -> LLMTriageRecommendation:
    data = dict(priority="HIGH", decision="INVESTIGATE", evidence=(item(),), uncertainty="Compromise is not confirmed.",
                likely_interpretation="Possible credential guessing; unconfirmed.", recommended_actions=("Review events.",),
                audit_note="Recommendation only.", tools_called=("get_alert_details",), source_record_ids=("ALERT-A-001",))
    return LLMTriageRecommendation(**{**data, **overrides})


def trace(**overrides) -> LLMExecutionTrace:
    data = dict(run_id="run-1", scenario_id="A", runtime_mode=RuntimeMode.MOCK, model_name="fake", started_at=NOW,
                completed_at=NOW + timedelta(seconds=1), iterations=3, final_output_valid=True)
    return LLMExecutionTrace(**{**data, **overrides})


@pytest.mark.parametrize("priority", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
def test_lm01_priority_accepts_approved_values(priority):
    assert recommendation(priority=priority).priority.value == priority


@pytest.mark.parametrize("bad", ["URGENT", "high", "", "SEVERE", 3])
def test_lm01_priority_rejects_other_values(bad):
    with pytest.raises(ValidationError):
        recommendation(priority=bad)


@pytest.mark.parametrize("decision", ["MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"])
def test_lm02_decision_accepts_approved_values(decision):
    assert recommendation(decision=decision).decision == decision


@pytest.mark.parametrize("bad", ["BLOCK_ACCOUNT", "investigate", "ESCALATE", "", "QUARANTINE_HOST"])
def test_lm02_decision_rejects_other_values(bad):
    with pytest.raises(ValidationError):
        recommendation(decision=bad)


def test_lm03_grounded_evidence_requires_a_valid_tool_name():
    with pytest.raises(ValidationError):
        item(tool_name="")
    with pytest.raises(ValidationError):
        item(tool_name="get_weather")
    with pytest.raises(ValidationError):
        GroundedEvidenceItem(claim="c", tool_call_id="x", source_record_ids=("A",))  # type: ignore[call-arg]


@pytest.mark.parametrize("call_id", ["", "   "])
def test_lm04_grounded_evidence_requires_a_tool_call_id(call_id):
    with pytest.raises(ValidationError):
        item(tool_call_id=call_id)


def test_lm05_substantive_evidence_requires_source_ids_but_absence_observations_do_not():
    with pytest.raises(ValidationError, match="at least one source record ID"):
        item(source_record_ids=())
    absence = item(claim="No matching events were found.", tool_name="search_authentication_events",
                   source_record_ids=(), is_absence_observation=True)
    assert absence.source_record_ids == () and absence.is_absence_observation is True


def test_lm06_recommended_actions_cannot_be_empty_or_blank():
    with pytest.raises(ValidationError):
        recommendation(recommended_actions=())
    with pytest.raises(ValidationError):
        recommendation(recommended_actions=("Review events.", "  "))


def test_lm07_tool_list_cannot_contain_blank_names():
    with pytest.raises(ValidationError):
        recommendation(tools_called=())
    with pytest.raises(ValidationError):
        recommendation(tools_called=("get_alert_details", ""))


def test_lm08_source_ids_cannot_contain_blanks():
    with pytest.raises(ValidationError):
        recommendation(source_record_ids=("ALERT-A-001", " "))
    with pytest.raises(ValidationError):
        item(source_record_ids=("ALERT-A-001", ""))
    with pytest.raises(ValidationError):
        recommendation(source_record_ids=())


def test_lm09_execution_trace_timestamps_are_timezone_aware_and_ordered():
    with pytest.raises(ValidationError):
        trace(started_at=datetime(2026, 3, 10, 12, 0))
    with pytest.raises(ValidationError):
        trace(completed_at=datetime(2026, 3, 10, 12, 0, 1))
    with pytest.raises(ValidationError, match="must not be before"):
        trace(completed_at=NOW - timedelta(seconds=1))
    assert trace().started_at.tzinfo is not None


def test_lm10_iteration_count_cannot_be_negative():
    with pytest.raises(ValidationError):
        trace(iterations=-1)
    with pytest.raises(ValidationError):
        LLMToolCallRecord(tool_call_id="c", tool_name="t", arguments={}, iteration=-1, status=CallStatus.SUCCESS)
    assert trace(iterations=0).iterations == 0


@pytest.mark.parametrize("build", [lambda: item(extra_field=1), lambda: recommendation(utility_score=0.9),
                                   lambda: trace(chain_of_thought="private"), lambda: AssertedFacts(risk=1),
                                   lambda: GuardrailResult(name="n", passed=True, detail="d", extra=1)])
def test_lm11_unexpected_fields_are_rejected(build):
    with pytest.raises(ValidationError):
        build()


def test_lm12_all_models_serialize_cleanly_to_json():
    models = [item(), recommendation(), trace(tool_calls=(LLMToolCallRecord(
        tool_call_id="c", tool_name="get_alert_details", arguments={"alert_id": "A"}, iteration=0,
        status=CallStatus.SUCCESS, source_record_ids=("A",), evidence_count=1),)),
        ToolObservation(tool_call_id="c", tool_name="t", status=CallStatus.SUCCESS, evidence={"a": 1}),
        GuardrailResult(name="n", passed=True)]
    for model in models:
        text = json.dumps(model.model_dump(mode="json"))
        assert type(model).model_validate_json(text) == model


def test_lm13_models_are_immutable():
    rec = recommendation()
    with pytest.raises(ValidationError):
        rec.priority = "LOW"
    with pytest.raises(ValidationError):
        item().claim = "changed"
    assert rec.model_copy(deep=True) == rec


def test_the_recommendation_schema_requires_the_documented_fields():
    required = set(LLMTriageRecommendation.model_json_schema()["required"])
    assert required == {"priority", "decision", "evidence", "uncertainty", "likely_interpretation",
                        "recommended_actions", "audit_note", "tools_called", "source_record_ids"}


def test_trace_messages_cannot_carry_paths_or_stack_traces():
    for message in ("Failed at C:\\Users\\x\\data.json", "Traceback (most recent call last):", 'File "a.py", line 1'):
        with pytest.raises(ValidationError, match="paths or stack traces"):
            trace(errors=(message,))
    assert trace(errors=("The model call timed out.",), warnings=("one correction was needed",)).errors


def test_a_tool_observation_is_reliable_only_when_successful():
    for status in CallStatus:
        observation = ToolObservation(tool_call_id="c", tool_name="t", status=status, is_relevant=True)
        assert observation.is_reliable is (status is CallStatus.SUCCESS)
        assert observation.counts_as_evidence is (status is CallStatus.SUCCESS)
    assert ToolObservation(tool_call_id="c", tool_name="t", status=CallStatus.SUCCESS).counts_as_evidence is False


def test_asserted_facts_report_only_what_is_stated():
    assert AssertedFacts().stated() == {}
    stated = AssertedFacts(failed_login_count=12, asset_criticality="CRITICAL").stated()
    assert stated == {"failed_login_count": 12, "asset_criticality": "CRITICAL"}
    with pytest.raises(ValidationError):
        AssertedFacts(failed_login_count=-1)
    with pytest.raises(ValidationError):
        AssertedFacts(asset_criticality="EXTREME")

def test_compact_llm_decision_accepts_valid_judgment() -> None:
    decision = CompactLLMTriageDecision(
        priority="HIGH",
        decision="INVESTIGATE",
        uncertainty="No compromise or impact is confirmed.",
        likely_interpretation=(
            "The evidence is consistent with possible "
            "brute-force authentication activity."
        ),
        recommended_actions=(
            "Review related authentication activity.",
            "Check for correlated alerts.",
        ),
    )

    assert decision.priority == "HIGH"
    assert decision.decision == "INVESTIGATE"
    assert len(decision.recommended_actions) == 2


def test_compact_llm_decision_requires_uncertainty() -> None:
    with pytest.raises(ValidationError):
        CompactLLMTriageDecision(
            priority="HIGH",
            decision="INVESTIGATE",
            uncertainty="",
            likely_interpretation="Possible brute-force activity.",
            recommended_actions=(
                "Review authentication activity.",
            ),
        )


def test_compact_llm_decision_rejects_blank_actions() -> None:
    with pytest.raises(ValidationError):
        CompactLLMTriageDecision(
            priority="HIGH",
            decision="INVESTIGATE",
            uncertainty="No compromise is confirmed.",
            likely_interpretation="Possible brute-force activity.",
            recommended_actions=("   ",),
        )


def test_compact_llm_decision_contains_only_judgment_fields() -> None:
    fields = set(
        CompactLLMTriageDecision.model_fields
    )

    assert fields == {
        "priority",
        "decision",
        "uncertainty",
        "likely_interpretation",
        "recommended_actions",
    }

    assert "evidence" not in fields
    assert "tool_call_id" not in fields
    assert "source_record_ids" not in fields
    assert "tools_called" not in fields
    assert "audit_note" not in fields
