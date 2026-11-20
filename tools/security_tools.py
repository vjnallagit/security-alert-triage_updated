"""The four simulated, read-only security evidence tools.

These are thin adapters over SecurityToolService, shaped so a later LangChain wrapper can use
them directly: plain keyword arguments that match the input models in models/tool_models.py,
and typed, JSON-serializable results. They retrieve evidence; they never decide or act.

When no service is passed, a fresh default service over the committed synthetic data is
created for that call (no shared module-level state). Pass your own service to keep an
invocation log across calls or to point the tools at other data files.
"""

from datetime import datetime

from models.tool_models import (
    GetAlertDetailsResult,
    GetAssetCriticalityResult,
    GetResponsePolicyResult,
    SearchAuthenticationEventsResult,
    get_tool_catalogue,
)
from services.security_tool_service import SecurityToolService


def _service_or_default(service: SecurityToolService | None) -> SecurityToolService:
    return service if service is not None else SecurityToolService()


def get_alert_details(
    alert_id: str,
    include_raw_evidence: bool = True,
    service: SecurityToolService | None = None,
) -> GetAlertDetailsResult:
    """Retrieve the factual synthetic alert record for an alert ID (type, severity, summary, evidence)."""
    return _service_or_default(service).get_alert_details(alert_id, include_raw_evidence)


def search_authentication_events(
    account: str | None = None,
    source: str | None = None,
    asset_id: str | None = None,
    alert_id: str | None = None,
    scenario_id: str | None = None,
    success: bool | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 100,
    service: SecurityToolService | None = None,
) -> SearchAuthenticationEventsResult:
    """Search synthetic authentication events (criteria ANDed, at least one required) with counts."""
    return _service_or_default(service).search_authentication_events(
        account=account, source=source, asset_id=asset_id, alert_id=alert_id, scenario_id=scenario_id,
        success=success, start_time=start_time, end_time=end_time, limit=limit,
    )


def get_asset_criticality(
    asset_id: str,
    service: SecurityToolService | None = None,
) -> GetAssetCriticalityResult:
    """Retrieve the recorded synthetic criticality of an asset."""
    return _service_or_default(service).get_asset_criticality(asset_id)


def get_response_policy(
    policy_id: str | None = None,
    alert_type: str | None = None,
    severity: str | None = None,
    asset_criticality: str | None = None,
    service: SecurityToolService | None = None,
) -> GetResponsePolicyResult:
    """Retrieve the defensive response policy by ID, or by one unambiguous alert context."""
    return _service_or_default(service).get_response_policy(
        policy_id=policy_id, alert_type=alert_type, severity=severity, asset_criticality=asset_criticality,
    )


__all__ = [
    "get_alert_details",
    "search_authentication_events",
    "get_asset_criticality",
    "get_response_policy",
    "get_tool_catalogue",
]
