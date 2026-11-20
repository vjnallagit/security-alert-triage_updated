"""SecurityToolService: read-only, deterministic evidence retrieval over local synthetic data.

Layers: data files -> existing data loader + validated models -> this service -> the thin
public functions in tools/security_tools.py.

The service RETRIEVES evidence. It never decides, never acts and never touches the network.
Expected outcomes (invalid input, not found, ambiguous, no matches) come back as structured
statuses. Unexpected data failures come back as status ERROR with a safe message (no paths,
no stack traces); the full cause chain is written to the "security_tools" logger for developers.

Every call is recorded in a bounded, instance-owned, in-memory log. Nothing is persisted.
"""

import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from models.schemas import AlertDetail, Asset, AuthenticationEvent, ResponsePolicy, SecurityScenario
from models.tool_models import (
    MAX_INVOCATION_LOG_ENTRIES,
    AlertEvidence,
    AssetEvidence,
    AuthenticationSearchEvidence,
    ConsistencyReport,
    GetAlertDetailsInput,
    GetAlertDetailsResult,
    GetAssetCriticalityInput,
    GetAssetCriticalityResult,
    GetResponsePolicyInput,
    GetResponsePolicyResult,
    PolicyContext,
    PolicyEvidence,
    SearchAuthenticationEventsInput,
    SearchAuthenticationEventsResult,
    TimeRange,
    ToolError,
    ToolInvocationRecord,
    ToolResultBase,
    ToolStatus,
)
from pydantic import BaseModel, ValidationError
from services.data_loader import DATA_DIR, DataLoadError, load_json_list

logger = logging.getLogger("security_tools")

ModelT = TypeVar("ModelT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=ToolResultBase)

_GENERIC_LIMITATION = "All data is synthetic and local; no real SIEM, IAM or threat-intelligence system is queried."
ALERT_LIMITATIONS = (
    _GENERIC_LIMITATION,
    "Reports only what the alert record states; impact is not inferred, so confirmed_impact is left unset.",
    "Does not prioritize the alert or recommend an action.",
)
EVENT_LIMITATIONS = (
    _GENERIC_LIMITATION,
    "Counts are over events whose success flag is false or true; event records carry no source-familiarity "
    "information, so unfamiliar sources are not determined.",
    "No IP reputation, geolocation or attacker attribution is performed; counts do not establish compromise.",
)
ASSET_LIMITATIONS = (
    _GENERIC_LIMITATION,
    "Reports recorded criticality only; it is not translated into an incident status or priority.",
)
POLICY_LIMITATIONS = (
    _GENERIC_LIMITATION,
    "applies_to is derived from the synthetic scenarios that reference each policy, because policy records "
    "do not declare their own applicability.",
    "Returns guidance only; no recommended action is executed and automatic disruptive action is never permitted.",
)


class ToolDataError(Exception):
    """Source data could not be read or validated. The original cause is chained."""


@dataclass(frozen=True)
class _Outcome:
    status: ToolStatus
    summary: str
    evidence: BaseModel | None = None
    source_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: ToolError | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_call_id() -> str:
    return f"CALL-{uuid.uuid4().hex[:12]}"


def _json_safe(value: Any) -> Any:
    """Make a request echo JSON-safe and bounded; invalid inputs may be arbitrary objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:200]


def _not_found(label: str) -> _Outcome:
    return _Outcome(
        ToolStatus.NOT_FOUND, f"No synthetic {label} matched the request.",
        error=ToolError(error_type="NotFound", message=f"No synthetic {label} matched the supplied identifier."),
    )


def _ambiguous(label: str, count: int) -> _Outcome:
    return _Outcome(
        ToolStatus.AMBIGUOUS, f"Several synthetic {label} records matched; none was chosen.",
        warnings=(f"{count} synthetic {label} records matched the request.",),
        error=ToolError(error_type="Ambiguous",
                        message=f"{count} synthetic {label} records matched; refine the request or fix the data."),
    )


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    message = "; ".join(parts) or "Input failed validation."
    return message if "Traceback" not in message else "Input failed validation."


class SecurityToolService:
    """Holds only injected data locations, a clock, an ID factory and its own bounded call log."""

    def __init__(
        self,
        alert_details_path: Path = DATA_DIR / "alert_details.json",
        authentication_events_path: Path = DATA_DIR / "authentication_events.json",
        assets_path: Path = DATA_DIR / "assets.json",
        policies_path: Path = DATA_DIR / "policies.json",
        scenarios_path: Path = DATA_DIR / "scenarios.json",
        *,
        max_log_entries: int = MAX_INVOCATION_LOG_ENTRIES,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = _new_call_id,
    ) -> None:
        if not 1 <= max_log_entries <= MAX_INVOCATION_LOG_ENTRIES:
            raise ValueError(f"max_log_entries must be between 1 and {MAX_INVOCATION_LOG_ENTRIES}")
        self._alert_path = Path(alert_details_path)
        self._event_path = Path(authentication_events_path)
        self._asset_path = Path(assets_path)
        self._policy_path = Path(policies_path)
        self._scenario_path = Path(scenarios_path)
        self._clock = clock
        self._id_factory = id_factory
        self._log: deque[ToolInvocationRecord] = deque(maxlen=max_log_entries)

    # --- public tools ----------------------------------------------------------

    def get_alert_details(self, alert_id: str, include_raw_evidence: bool = True) -> GetAlertDetailsResult:
        return self._execute(
            "get_alert_details", {"alert_id": alert_id, "include_raw_evidence": include_raw_evidence},
            GetAlertDetailsInput, self._alert_outcome, GetAlertDetailsResult, "alert_details", ALERT_LIMITATIONS,
        )

    def search_authentication_events(
        self,
        account: str | None = None,
        source: str | None = None,
        asset_id: str | None = None,
        alert_id: str | None = None,
        scenario_id: str | None = None,
        success: bool | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> SearchAuthenticationEventsResult:
        query = {
            "account": account, "source": source, "asset_id": asset_id, "alert_id": alert_id,
            "scenario_id": scenario_id, "success": success, "start_time": start_time, "end_time": end_time,
            "limit": limit,
        }
        return self._execute(
            "search_authentication_events", query, SearchAuthenticationEventsInput, self._events_outcome,
            SearchAuthenticationEventsResult, "authentication_events", EVENT_LIMITATIONS,
        )

    def get_asset_criticality(self, asset_id: str) -> GetAssetCriticalityResult:
        return self._execute(
            "get_asset_criticality", {"asset_id": asset_id}, GetAssetCriticalityInput, self._asset_outcome,
            GetAssetCriticalityResult, "assets", ASSET_LIMITATIONS,
        )

    def get_response_policy(
        self,
        policy_id: str | None = None,
        alert_type: str | None = None,
        severity: str | None = None,
        asset_criticality: str | None = None,
    ) -> GetResponsePolicyResult:
        query = {"policy_id": policy_id, "alert_type": alert_type, "severity": severity,
                 "asset_criticality": asset_criticality}
        return self._execute(
            "get_response_policy", query, GetResponsePolicyInput, self._policy_outcome,
            GetResponsePolicyResult, "policies", POLICY_LIMITATIONS,
        )

    # --- invocation audit ----------------------------------------------------------

    def get_invocation_log(self) -> tuple[ToolInvocationRecord, ...]:
        """A deep-copied snapshot, oldest first; changing it cannot affect the service's log."""
        return tuple(record.model_copy(deep=True) for record in self._log)

    def reset_invocation_log(self) -> None:
        self._log.clear()

    # --- consistency (developer/validation support, not a tool) -------------------------

    def check_consistency(self) -> ConsistencyReport:
        """Check that alerts, events, assets, policies and scenarios agree. Raises ToolDataError on bad files."""
        alerts = self._load(self._alert_path, AlertDetail, "alert details")
        events = self._load(self._event_path, AuthenticationEvent, "authentication events")
        assets = self._load(self._asset_path, Asset, "assets")
        policies = self._load(self._policy_path, ResponsePolicy, "policies")
        scenarios = self._load(self._scenario_path, SecurityScenario, "scenarios")
        issues: list[str] = []

        for label, ids in (("alert", [a.alert_id for a in alerts]), ("event", [e.event_id for e in events]),
                           ("asset", [a.asset_id for a in assets]), ("policy", [p.policy_id for p in policies]),
                           ("scenario", [s.scenario_id for s in scenarios])):
            issues.extend(f"Duplicate {label} ID: {i}" for i in sorted({i for i in ids if ids.count(i) > 1}))

        alert_by_id = {a.alert_id: a for a in alerts}
        asset_ids = {a.asset_id for a in assets}
        for e in events:
            alert = alert_by_id.get(e.alert_id)
            if alert is None:
                issues.append(f"Event {e.event_id} references missing alert {e.alert_id}")
            else:
                if e.account != alert.target:
                    issues.append(f"Event {e.event_id} account differs from alert {alert.alert_id} target")
                if e.source != alert.source:
                    issues.append(f"Event {e.event_id} source differs from alert {alert.alert_id} source")
            if e.asset_id not in asset_ids:
                issues.append(f"Event {e.event_id} references missing asset {e.asset_id}")

        asset_by_id = {a.asset_id: a for a in assets}
        policy_ids = {p.policy_id for p in policies}
        for s in scenarios:
            alert = alert_by_id.get(s.alert_id)
            if alert is None:
                issues.append(f"Scenario {s.scenario_id} references missing alert {s.alert_id}")
            else:
                if alert.alert_type != s.alert_type or alert.severity != s.severity:
                    issues.append(f"Scenario {s.scenario_id} alert type/severity differ from alert {alert.alert_id}")
                failed = sum(1 for e in events if e.alert_id == s.alert_id and not e.success)
                if failed != s.failed_login_count:
                    issues.append(f"Scenario {s.scenario_id} failed_login_count differs from its authentication events")
            asset = asset_by_id.get(s.asset_id)
            if asset is None:
                issues.append(f"Scenario {s.scenario_id} references missing asset {s.asset_id}")
            elif asset.criticality != s.asset_criticality:
                issues.append(f"Scenario {s.scenario_id} asset criticality differs from asset {asset.asset_id}")
            if s.policy_id not in policy_ids:
                issues.append(f"Scenario {s.scenario_id} references missing policy {s.policy_id}")
        return ConsistencyReport(issues=tuple(issues))

    # --- execution wrapper -------------------------------------------------------------

    def _execute(
        self,
        tool_name: str,
        raw_query: dict[str, Any],
        input_model: type[BaseModel],
        handler: Callable[[Any], _Outcome],
        result_model: type[ResultT],
        source_type: str,
        limitations: tuple[str, ...],
    ) -> ResultT:
        started_at = self._clock()
        started = time.perf_counter()
        call_id = self._id_factory()
        query = _json_safe(raw_query)
        try:
            parsed = input_model.model_validate(raw_query)
        except ValidationError as exc:
            outcome = _Outcome(
                ToolStatus.INVALID_INPUT, "The tool input was invalid; nothing was searched.",
                error=ToolError(error_type="InvalidInput", message=_validation_message(exc)),
            )
        else:
            query = parsed.model_dump(mode="json", exclude_none=True)
            try:
                outcome = handler(parsed)
            except ToolDataError as exc:
                logger.error("Tool %s failed (call %s): %s", tool_name, call_id, exc, exc_info=True)
                cause = type(exc.__cause__).__name__ if exc.__cause__ else "ToolDataError"
                outcome = _Outcome(
                    ToolStatus.ERROR, "The synthetic source data could not be used.",
                    error=ToolError(error_type=cause, message=str(exc)),
                )

        duration_ms = (time.perf_counter() - started) * 1000
        completed_at = started_at + timedelta(milliseconds=duration_ms)
        result = result_model(
            call_id=call_id, status=outcome.status, query=query, evidence=outcome.evidence, summary=outcome.summary,
            source_type=source_type, source_record_ids=outcome.source_ids, warnings=outcome.warnings,
            limitations=limitations, error=outcome.error, timestamp=completed_at, **outcome.extras,
        )
        self._log.append(ToolInvocationRecord(
            call_id=call_id, tool_name=tool_name, started_at=started_at, completed_at=completed_at,
            duration_ms=duration_ms, request=query, status=result.status, source_record_ids=result.source_record_ids,
            evidence_count=result.evidence_count, warnings=result.warnings,
            error_type=result.error.error_type if result.error else None,
        ))
        return result

    # --- data access ---------------------------------------------------------------------

    @staticmethod
    def _load(path: Path, model: type[ModelT], label: str) -> list[ModelT]:
        """Read and validate one source file through the shared loader. Never returns partial data."""
        try:
            return [model.model_validate(item) for item in load_json_list(path.name, path.parent)]
        except (DataLoadError, ValidationError) as exc:
            raise ToolDataError(f"Source data for {label} could not be read or validated.") from exc

    # --- tool logic ------------------------------------------------------------------------

    def _alert_outcome(self, query: GetAlertDetailsInput) -> _Outcome:
        matches = [r for r in self._load(self._alert_path, AlertDetail, "alert details") if r.alert_id == query.alert_id]
        if not matches:
            return _not_found("alert")
        if len(matches) > 1:
            return _ambiguous("alert", len(matches))
        record = matches[0]
        evidence = AlertEvidence(
            alert_id=record.alert_id, timestamp=record.timestamp, alert_type=record.alert_type,
            severity=record.severity, summary=record.summary, target=record.target, source=record.source,
            evidence_items=tuple(record.evidence) if query.include_raw_evidence else (),
        )
        return _Outcome(
            ToolStatus.SUCCESS,
            f"Alert {record.alert_id}: type {record.alert_type}, severity {record.severity.value}, "
            f"target {record.target}, source {record.source}; {len(record.evidence)} evidence item(s) on record.",
            evidence=evidence, source_ids=(record.alert_id,),
            warnings=() if query.include_raw_evidence else ("Evidence items were omitted at the caller's request.",),
        )

    def _events_outcome(self, query: SearchAuthenticationEventsInput) -> _Outcome:
        required_alert_ids = [query.alert_id] if query.alert_id else []
        if query.scenario_id:
            scenarios = [s for s in self._load(self._scenario_path, SecurityScenario, "scenarios")
                         if s.scenario_id == query.scenario_id]
            if not scenarios:
                return _not_found("scenario")
            if len(scenarios) > 1:
                return _ambiguous("scenario", len(scenarios))
            required_alert_ids.append(scenarios[0].alert_id)

        def matches(event: AuthenticationEvent) -> bool:
            return (
                all(event.alert_id == alert_id for alert_id in required_alert_ids)
                and (query.account is None or event.account == query.account)
                and (query.source is None or event.source == query.source)
                and (query.asset_id is None or event.asset_id == query.asset_id)
                and (query.success is None or event.success == query.success)
                and (query.start_time is None or event.timestamp >= query.start_time)
                and (query.end_time is None or event.timestamp <= query.end_time)
            )

        events = self._load(self._event_path, AuthenticationEvent, "authentication events")
        # Newest first; equal timestamps fall back to event ID so the order is always deterministic.
        matched = sorted((e for e in events if matches(e)), key=lambda e: (-e.timestamp.timestamp(), e.event_id))
        returned = tuple(matched[: query.limit])
        failed = sum(1 for e in matched if not e.success)
        evidence = AuthenticationSearchEvidence(
            matched_count=len(matched), returned_count=len(returned), events=returned,
            failed_login_count=failed, successful_login_count=len(matched) - failed,
            unique_sources=tuple(sorted({e.source for e in matched})),
            time_range=TimeRange(earliest=matched[-1].timestamp, latest=matched[0].timestamp) if matched else None,
            truncated=len(matched) > len(returned),
        )
        if not matched:
            summary = "No matching synthetic authentication events were found."
        else:
            summary = (f"{len(matched)} synthetic authentication event(s) matched: {failed} failed, "
                       f"{len(matched) - failed} successful; {len(evidence.unique_sources)} distinct source(s).")
        warnings = (f"Results truncated: {len(returned)} of {len(matched)} matching events returned.",) if evidence.truncated else ()
        return _Outcome(ToolStatus.SUCCESS, summary, evidence=evidence,
                        source_ids=tuple(e.event_id for e in returned), warnings=warnings)

    def _asset_outcome(self, query: GetAssetCriticalityInput) -> _Outcome:
        matches = [a for a in self._load(self._asset_path, Asset, "assets") if a.asset_id == query.asset_id]
        if not matches:
            return _not_found("asset")
        if len(matches) > 1:
            return _ambiguous("asset", len(matches))
        asset = matches[0]
        evidence = AssetEvidence(asset_id=asset.asset_id, asset_name=asset.asset_name, asset_type=asset.asset_type,
                                 criticality=asset.criticality, description=asset.description)
        return _Outcome(
            ToolStatus.SUCCESS,
            f"Asset {asset.asset_id} ({asset.asset_name}) is recorded with criticality {asset.criticality.value}.",
            evidence=evidence, source_ids=(asset.asset_id,),
        )

    def _policy_outcome(self, query: GetResponsePolicyInput) -> _Outcome:
        policies = self._load(self._policy_path, ResponsePolicy, "policies")
        scenarios = self._load(self._scenario_path, SecurityScenario, "scenarios")
        contexts: dict[str, set[PolicyContext]] = {}
        for s in scenarios:
            contexts.setdefault(s.policy_id, set()).add(
                PolicyContext(alert_type=s.alert_type, severity=s.severity, asset_criticality=s.asset_criticality)
            )

        warnings: tuple[str, ...] = ()
        if query.policy_id:
            matches = [p for p in policies if p.policy_id == query.policy_id]
            if any(v is not None for v in (query.alert_type, query.severity, query.asset_criticality)):
                warnings = ("Context selectors were ignored because policy_id was supplied.",)
        else:
            def applies(policy: ResponsePolicy) -> bool:
                return any(
                    (query.alert_type is None or c.alert_type == query.alert_type)
                    and (query.severity is None or c.severity == query.severity)
                    and (query.asset_criticality is None or c.asset_criticality == query.asset_criticality)
                    for c in contexts.get(policy.policy_id, ())
                )

            matches = [p for p in policies if applies(p)]

        if not matches:
            return _not_found("policy")
        if len(matches) > 1:
            outcome = _ambiguous("policy", len(matches))
            return _Outcome(outcome.status, outcome.summary, warnings=outcome.warnings, error=outcome.error,
                            extras={"candidate_policy_ids": tuple(sorted(p.policy_id for p in matches))})
        policy = matches[0]
        applies_to = tuple(sorted(contexts.get(policy.policy_id, ()),
                                  key=lambda c: (c.alert_type, c.severity.value, c.asset_criticality.value)))
        evidence = PolicyEvidence(
            policy_id=policy.policy_id, policy_name=policy.title, description=policy.description, applies_to=applies_to,
            recommended_investigation_actions=tuple(policy.recommended_actions),
            escalation_criteria=policy.escalation_guidance, restricted_actions=tuple(policy.prohibited_automated_actions),
            restriction_statement=policy.restriction,
            human_authorization_required=policy.human_authorization_required,
            automatic_disruptive_action_permitted=policy.automated_actions_permitted,
        )
        return _Outcome(
            ToolStatus.SUCCESS,
            f"Policy {policy.policy_id} ({policy.title}): {len(policy.recommended_actions)} permitted action(s); "
            "automatic disruptive action is not permitted and human authorization is required.",
            evidence=evidence, source_ids=(policy.policy_id,), warnings=warnings,
        )
