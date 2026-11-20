"""Typed inputs, outputs, audit records and catalogue for the simulated security evidence tools.

The tools RETRIEVE evidence from local synthetic data. They never decide, never act and never
return an AgentDecision. Every result carries the IDs of the source records it came from so a
later agent can trace its recommendation back to the evidence.

Matching policy for all lookups: exact and case-sensitive.
"""

import re
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, Field, computed_field, field_validator, model_validator

from models.goal_models import FrozenModel
from models.schemas import AssetCriticality, AuthenticationEvent, Severity

MAX_SEARCH_LIMIT = 100
MAX_INVOCATION_LOG_ENTRIES = 100
DURATION_TOLERANCE_MS = 1.0

# User-facing text must never contain local paths or stack traces.
_UNSAFE_MESSAGE = re.compile(r"[A-Za-z]:\\|/Users/|/home/|Traceback|File \"")

ToolName = Literal[
    "get_alert_details",
    "search_authentication_events",
    "get_asset_criticality",
    "get_response_policy",
]


class ToolStatus(StrEnum):
    SUCCESS = "SUCCESS"  # includes "no matching events": an empty search is an observation, not a failure
    NOT_FOUND = "NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    AMBIGUOUS = "AMBIGUOUS"  # several records matched; the tool never picks one silently
    ERROR = "ERROR"  # unexpected data/system failure, reported safely


# --------------------------------------------------------------------------
# Tool inputs (also the argument schemas for later LangChain wrapping)
# --------------------------------------------------------------------------


class GetAlertDetailsInput(FrozenModel):
    """Retrieve the synthetic alert record for one alert ID."""

    alert_id: str = Field(min_length=1, description="Exact ID of the alert, for example ALERT-A-001.")
    include_raw_evidence: bool = Field(
        default=True, description="Include the alert's evidence items. False returns the record without them."
    )


class SearchAuthenticationEventsInput(FrozenModel):
    """Search synthetic authentication events. All supplied criteria are combined with AND."""

    account: str | None = Field(default=None, min_length=1, description="Exact account name.")
    source: str | None = Field(default=None, min_length=1, description="Exact source address.")
    asset_id: str | None = Field(default=None, min_length=1, description="Exact asset ID.")
    alert_id: str | None = Field(default=None, min_length=1, description="Exact ID of the alert the events belong to.")
    scenario_id: str | None = Field(
        default=None, min_length=1,
        description="Scenario ID, resolved to that scenario's alert ID (stored events carry only alert_id).",
    )
    success: bool | None = Field(default=None, description="True: successful events only. False: failed events only.")
    start_time: AwareDatetime | None = Field(default=None, description="Earliest event time (inclusive, timezone-aware).")
    end_time: AwareDatetime | None = Field(default=None, description="Latest event time (inclusive, timezone-aware).")
    limit: int = Field(default=MAX_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT, description="Maximum events returned (1-100).")

    @model_validator(mode="after")
    def _needs_a_criterion_and_a_valid_window(self) -> Self:
        criteria = (self.account, self.source, self.asset_id, self.alert_id, self.scenario_id,
                    self.success, self.start_time, self.end_time)
        if all(value is None for value in criteria):
            raise ValueError("at least one search criterion is required")
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must not be after end_time")
        return self


class GetAssetCriticalityInput(FrozenModel):
    """Retrieve the synthetic criticality record for one asset ID."""

    asset_id: str = Field(min_length=1, description="Exact ID of the asset, for example ASSET-PAY-01.")


class GetResponsePolicyInput(FrozenModel):
    """Retrieve a synthetic defensive response policy by ID, or by one unambiguous alert context."""

    policy_id: str | None = Field(default=None, min_length=1, description="Exact policy ID (takes precedence).")
    alert_type: str | None = Field(default=None, min_length=1, description="Alert type the policy applies to.")
    severity: Severity | None = Field(default=None, description="Alert severity the policy applies to.")
    asset_criticality: AssetCriticality | None = Field(default=None, description="Asset criticality the policy applies to.")

    @model_validator(mode="after")
    def _needs_a_selector(self) -> Self:
        if all(v is None for v in (self.policy_id, self.alert_type, self.severity, self.asset_criticality)):
            raise ValueError("at least one selector is required")
        return self


# --------------------------------------------------------------------------
# Evidence payloads
# --------------------------------------------------------------------------


class AlertEvidence(FrozenModel):
    alert_id: str
    timestamp: AwareDatetime
    alert_type: str
    severity: Severity
    summary: str
    target: str
    source: str
    evidence_items: tuple[str, ...] = ()
    # None means "the source record does not state it"; the tool never infers impact.
    confirmed_impact: bool | None = None


class TimeRange(FrozenModel):
    earliest: AwareDatetime
    latest: AwareDatetime


class AuthenticationSearchEvidence(FrozenModel):
    matched_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    events: tuple[AuthenticationEvent, ...] = ()  # newest first
    failed_login_count: int = Field(ge=0)  # matched events whose success flag is false
    successful_login_count: int = Field(ge=0)  # matched events whose success flag is true
    unique_sources: tuple[str, ...] = ()  # sorted, deduplicated
    time_range: TimeRange | None = None
    truncated: bool

    @model_validator(mode="after")
    def _counts_must_be_consistent(self) -> Self:
        if self.returned_count != len(self.events) or self.returned_count > self.matched_count:
            raise ValueError("returned_count must equal the number of events and not exceed matched_count")
        if self.truncated != (self.matched_count > self.returned_count):
            raise ValueError("truncated must be true exactly when matched_count exceeds returned_count")
        if self.failed_login_count + self.successful_login_count != self.matched_count:
            raise ValueError("failed and successful counts must add up to matched_count")
        return self


class AssetEvidence(FrozenModel):
    asset_id: str
    asset_name: str
    asset_type: str
    criticality: AssetCriticality
    description: str
    business_impact_summary: str | None = None  # not present in the synthetic data


class PolicyContext(FrozenModel):
    """One alert context a policy applies to (derived from the scenarios that reference it)."""

    alert_type: str
    severity: Severity
    asset_criticality: AssetCriticality


class PolicyEvidence(FrozenModel):
    policy_id: str
    policy_name: str
    description: str
    applies_to: tuple[PolicyContext, ...] = ()
    recommended_investigation_actions: tuple[str, ...] = Field(min_length=1)
    escalation_criteria: str
    restricted_actions: tuple[str, ...] = Field(min_length=1)
    restriction_statement: str
    human_authorization_required: Literal[True]
    automatic_disruptive_action_permitted: Literal[False]


# --------------------------------------------------------------------------
# Common result envelope
# --------------------------------------------------------------------------


class ToolError(FrozenModel):
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def _message_must_be_safe(cls, value: str) -> str:
        if _UNSAFE_MESSAGE.search(value):
            raise ValueError("error messages must not contain local paths or stack traces")
        return value


class ToolResultBase(FrozenModel):
    tool_name: ToolName
    call_id: str = Field(min_length=1)
    status: ToolStatus
    query: dict[str, Any]
    evidence: BaseModel | None = None
    summary: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_record_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)
    error: ToolError | None = None
    is_synthetic: Literal[True] = True
    is_read_only: Literal[True] = True
    executed_actions: tuple[str, ...] = Field(default=(), max_length=0)  # always empty: tools only retrieve
    timestamp: AwareDatetime

    @model_validator(mode="before")
    @classmethod
    def _ignore_derived_fields_when_round_tripping(cls, data: Any) -> Any:
        """record_found is derived from the evidence; accept it in a dump so model_validate(model_dump()) works."""
        if isinstance(data, dict) and "record_found" in data:
            return {key: value for key, value in data.items() if key != "record_found"}
        return data

    @field_validator("source_record_ids")
    @classmethod
    def _source_ids_must_not_be_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("source record IDs must not be blank")
        return value

    @model_validator(mode="after")
    def _status_error_and_evidence_must_agree(self) -> Self:
        if self.status is ToolStatus.SUCCESS:
            if self.error is not None:
                raise ValueError("a successful result cannot carry an error")
            if self.evidence is None:
                raise ValueError("a successful result must carry evidence")
        else:
            if self.error is None:
                raise ValueError("a non-successful result must explain the failure in error")
            if self.evidence is not None:
                raise ValueError("a non-successful result must not carry evidence")
            if self.source_record_ids:
                raise ValueError("a non-successful result must not claim source records")
        return self

    @property
    def evidence_count(self) -> int:
        """Number of source records returned as evidence."""
        return len(self.source_record_ids)

    def to_llm_evidence(self) -> dict[str, Any]:
        """Compact, JSON-safe view for a language model: no Python objects, paths or exceptions."""
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "summary": self.summary,
            "evidence": self.evidence.model_dump(mode="json") if self.evidence is not None else None,
            "source_record_ids": list(self.source_record_ids),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "error": self.error.message if self.error else None,
        }


class GetAlertDetailsResult(ToolResultBase):
    tool_name: Literal["get_alert_details"] = "get_alert_details"
    evidence: AlertEvidence | None = None

    @computed_field
    @property
    def record_found(self) -> bool:
        return self.evidence is not None


class SearchAuthenticationEventsResult(ToolResultBase):
    tool_name: Literal["search_authentication_events"] = "search_authentication_events"
    evidence: AuthenticationSearchEvidence | None = None

    @model_validator(mode="after")
    def _sources_must_match_returned_events(self) -> Self:
        if self.evidence is not None and tuple(e.event_id for e in self.evidence.events) != self.source_record_ids:
            raise ValueError("source record IDs must be exactly the returned event IDs")
        return self


class GetAssetCriticalityResult(ToolResultBase):
    tool_name: Literal["get_asset_criticality"] = "get_asset_criticality"
    evidence: AssetEvidence | None = None

    @computed_field
    @property
    def record_found(self) -> bool:
        return self.evidence is not None


class GetResponsePolicyResult(ToolResultBase):
    tool_name: Literal["get_response_policy"] = "get_response_policy"
    evidence: PolicyEvidence | None = None
    candidate_policy_ids: tuple[str, ...] = ()  # filled only for AMBIGUOUS lookups

    @computed_field
    @property
    def record_found(self) -> bool:
        return self.evidence is not None


# --------------------------------------------------------------------------
# Invocation audit record
# --------------------------------------------------------------------------


class ToolInvocationRecord(FrozenModel):
    """One observable tool call. Contains inputs and outcomes only: no secrets, no stack traces."""

    call_id: str = Field(min_length=1)
    tool_name: ToolName
    started_at: AwareDatetime
    completed_at: AwareDatetime
    duration_ms: float = Field(ge=0)
    request: dict[str, Any]
    status: ToolStatus
    source_record_ids: tuple[str, ...] = ()
    evidence_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()
    error_type: str | None = None

    @model_validator(mode="after")
    def _timestamps_must_agree_with_duration(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not be before started_at")
        elapsed_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        if abs(elapsed_ms - self.duration_ms) > DURATION_TOLERANCE_MS:
            raise ValueError("duration_ms must match the difference between the timestamps")
        return self


# --------------------------------------------------------------------------
# Consistency report
# --------------------------------------------------------------------------


class ConsistencyReport(FrozenModel):
    """Result of checking that the synthetic sources agree with each other."""

    issues: tuple[str, ...] = ()

    @property
    def consistent(self) -> bool:
        return not self.issues


# --------------------------------------------------------------------------
# Tool catalogue
# --------------------------------------------------------------------------


class ToolDefinition(FrozenModel):
    """Metadata a later LangChain wrapper can use for name, description and argument schema."""

    name: ToolName
    description: str = Field(min_length=1)
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    read_only: Literal[True] = True
    synthetic: Literal[True] = True
    safety_classification: Literal["READ_ONLY_SYNTHETIC_EVIDENCE_RETRIEVAL"] = "READ_ONLY_SYNTHETIC_EVIDENCE_RETRIEVAL"
    allowed_use: str = Field(min_length=1)
    prohibited_use: tuple[str, ...] = Field(min_length=1)

    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()


_COMMON_PROHIBITED = (
    "Making the final triage decision or priority.",
    "Executing, requesting or simulating any security action.",
    "Inferring or claiming confirmed compromise.",
)

TOOL_CATALOGUE: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="get_alert_details",
        description=(
            "Retrieve the factual synthetic alert record for an alert ID: type, severity, summary, target, "
            "source and the alert's evidence items. Use it first, to learn what the alert reports. "
            "Requires alert_id. It does not judge priority, infer impact or search other records."
        ),
        input_model=GetAlertDetailsInput,
        output_model=GetAlertDetailsResult,
        allowed_use="Read the alert's own recorded observations.",
        prohibited_use=_COMMON_PROHIBITED,
    ),
    ToolDefinition(
        name="search_authentication_events",
        description=(
            "Search synthetic authentication events by account, source, asset_id, alert_id, scenario_id, "
            "success flag and/or time window (criteria are ANDed; at least one is required; limit 1-100). "
            "Returns matching events plus counts of failed and successful events and the distinct sources. "
            "Use it to check how many failed or successful logins occurred. It does not look up IP reputation, "
            "identify attackers or conclude that an account is compromised."
        ),
        input_model=SearchAuthenticationEventsInput,
        output_model=SearchAuthenticationEventsResult,
        allowed_use="Count and inspect simulated authentication events.",
        prohibited_use=(*_COMMON_PROHIBITED, "External network, IP-reputation or identity-system lookups."),
    ),
    ToolDefinition(
        name="get_asset_criticality",
        description=(
            "Retrieve the synthetic business/security criticality of an asset: name, type, criticality and "
            "description. Requires asset_id. Use it to learn how important the affected asset is. It does not "
            "convert criticality into an incident status or priority."
        ),
        input_model=GetAssetCriticalityInput,
        output_model=GetAssetCriticalityResult,
        allowed_use="Read an asset's recorded criticality.",
        prohibited_use=_COMMON_PROHIBITED,
    ),
    ToolDefinition(
        name="get_response_policy",
        description=(
            "Retrieve the synthetic defensive response policy: permitted investigation actions, escalation "
            "criteria, restricted actions and the human-authorization requirement. Provide policy_id, or an "
            "alert_type/severity/asset_criticality context that matches exactly one policy. It does not execute "
            "any recommended action and never permits automatic disruptive action."
        ),
        input_model=GetResponsePolicyInput,
        output_model=GetResponsePolicyResult,
        allowed_use="Read authorized defensive guidance and restrictions.",
        prohibited_use=_COMMON_PROHIBITED,
    ),
)


def get_tool_catalogue() -> tuple[ToolDefinition, ...]:
    return TOOL_CATALOGUE
