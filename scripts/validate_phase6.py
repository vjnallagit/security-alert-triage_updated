"""Phase 6 validation: the four simulated, read-only security evidence tools.

Run from anywhere:  python scripts/validate_phase6.py

The tools are exercised with every network call blocked, and the SHA-256 of every synthetic
data file is compared before and after. No LLM, LangChain or Ollama code is involved.
"""

import contextlib
import hashlib
import io
import json
import socket
import sys
from pathlib import Path

# Allow running as a plain script without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_phase1  # noqa: E402
import validate_phase2  # noqa: E402
import validate_phase3  # noqa: E402
import validate_phase4  # noqa: E402
import validate_phase5  # noqa: E402
from models.schemas import AgentDecision  # noqa: E402
from models.tool_models import TOOL_CATALOGUE, ToolResultBase, ToolStatus  # noqa: E402
from services.data_loader import DATA_DIR, DataLoadError  # noqa: E402
from services.scenario_service import get_all_scenarios  # noqa: E402
from services.security_tool_service import SecurityToolService  # noqa: E402
from tools import (  # noqa: E402
    get_alert_details,
    get_asset_criticality,
    get_response_policy,
    search_authentication_events,
)

DATA_FILES = ("scenarios.json", "alert_details.json", "authentication_events.json", "assets.json",
              "policies.json", "experience.json")
SECRET_WORDS = ("password", "secret", "token", "api_key", "credential", "traceback")
DECISION_KEYS = {"agent_name", "agent_type", "priority", "decision", "reasoning", "likely_interpretation",
                 "audit_note", "utility_score", "recommended_actions"}
FORBIDDEN_MODULES = {"langchain", "langchain_core", "langchain_ollama", "ollama", "requests", "httpx",
                     "fastapi", "streamlit"}

all_results: list[ToolResultBase] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def hashes() -> dict[str, str]:
    return {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() for name in DATA_FILES}


def raw(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def keys_recursive(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_recursive(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_recursive(v)}
    return set()


def validate_previous_phases() -> None:
    print("Previous phases:")
    with contextlib.redirect_stdout(io.StringIO()):
        validate_phase1.run()
    check(True, "Phase 1")
    for number, module in ((2, validate_phase2), (3, validate_phase3), (4, validate_phase4), (5, validate_phase5)):
        with contextlib.redirect_stdout(io.StringIO()):
            status = module.main()
        check(status == 0, f"Phase {number}")


def record(result: ToolResultBase) -> ToolResultBase:
    """Re-validate through the model, keep it for the audit checks, and return it."""
    validated = type(result).model_validate(result.model_dump())
    json.dumps(validated.model_dump(mode="json"))  # must be JSON-serializable
    all_results.append(validated)
    return validated


def validate_catalogue() -> None:
    print("\nTool catalogue:")
    check(len(TOOL_CATALOGUE) == 4 and [t.name for t in TOOL_CATALOGUE] == [
        "get_alert_details", "search_authentication_events", "get_asset_criticality", "get_response_policy",
    ], "Four tools registered")
    check(all(t.synthetic for t in TOOL_CATALOGUE), "All tools are synthetic")
    check(all(t.read_only for t in TOOL_CATALOGUE), "All tools are read-only")
    check(not FORBIDDEN_MODULES & set(sys.modules), "No LLM dependency")


def tool_block(service: SecurityToolService, scenario) -> dict[str, ToolResultBase]:
    sid = scenario.scenario_id
    print(f"\nScenario {sid}:")
    alert = record(get_alert_details(scenario.alert_id, service=service))
    events = record(search_authentication_events(
        account=scenario.target_account, success=False, service=service))
    all_events = record(search_authentication_events(scenario_id=sid, service=service))
    asset = record(get_asset_criticality(scenario.asset_id, service=service))
    policy = record(get_response_policy(policy_id=scenario.policy_id, service=service))

    print("\nget_alert_details")
    check(alert.status is ToolStatus.SUCCESS, f"Status: {alert.status.value}")
    check(alert.source_record_ids == (scenario.alert_id,), f"Source: {', '.join(alert.source_record_ids)}")
    print(f"       {alert.summary}")
    print(f"       evidence items: {list(alert.evidence.evidence_items)}")

    print("\nsearch_authentication_events (account=%s, success=False)" % scenario.target_account)
    check(events.status is ToolStatus.SUCCESS, f"Status: {events.status.value}")
    check(True, f"Matching events: {events.evidence.matched_count}")
    check(True, f"Failed events: {events.evidence.failed_login_count}")
    real_event_ids = {e["event_id"] for e in raw("authentication_events.json")}
    check(set(events.source_record_ids) <= real_event_ids and len(events.source_record_ids) == events.evidence.returned_count,
          "Source record IDs verified against the raw event file")
    shown = ", ".join(events.source_record_ids[:3]) + (" ..." if len(events.source_record_ids) > 3 else "")
    print(f"       source IDs: {shown or '(none)'}   distinct sources: {list(events.evidence.unique_sources)}")
    print(f"       all events for scenario {sid}: {all_events.evidence.matched_count} "
          f"({all_events.evidence.failed_login_count} failed, {all_events.evidence.successful_login_count} successful)")

    print("\nget_asset_criticality")
    check(asset.status is ToolStatus.SUCCESS, f"Status: {asset.status.value}")
    check(True, f"Asset criticality: {asset.evidence.criticality.value}  ({asset.evidence.asset_name})")

    print("\nget_response_policy")
    check(policy.status is ToolStatus.SUCCESS, f"Status: {policy.status.value}")
    check(len(policy.evidence.recommended_investigation_actions) >= 3, "Defensive investigation guidance returned")
    check(policy.evidence.automatic_disruptive_action_permitted is False,
          f"Automatic disruptive action permitted: {policy.evidence.automatic_disruptive_action_permitted}")
    check(policy.evidence.human_authorization_required is True,
          f"Human authorization required: {policy.evidence.human_authorization_required}")
    return {"alert": alert, "events": events, "all_events": all_events, "asset": asset, "policy": policy}


def validate_scenario_a(block: dict[str, ToolResultBase]) -> None:
    alert, events, asset = block["alert"].evidence, block["events"].evidence, block["asset"].evidence
    check(alert.severity.value == "HIGH" and alert.alert_type == "repeated_failed_login"
          and any("12 failed authentication attempts" in i for i in alert.evidence_items),
          "High-severity repeated-login evidence returned")
    check(events.matched_count == events.failed_login_count == 12 and events.unique_sources == ("203.0.113.25",),
          "Repeated failed logins from a single unfamiliar-range source returned")
    check(block["all_events"].evidence.successful_login_count == 0, "No successful login recorded for the scenario")
    check(asset.criticality.value == "HIGH", "Privileged-account asset importance returned")


def validate_scenario_b(block: dict[str, ToolResultBase]) -> None:
    alert, all_events = block["alert"].evidence, block["all_events"].evidence
    check(alert.severity.value == "LOW", "Low-severity evidence returned")
    check(all_events.matched_count == 2 and all_events.failed_login_count == 1 and all_events.successful_login_count == 1,
          "Limited authentication evidence returned (one failure followed by a success)")
    check("Do not escalate on limited evidence" in block["policy"].evidence.escalation_criteria,
          "Policy remains proportionate")
    check(alert.confirmed_impact is None and "No related alerts" in " ".join(alert.evidence_items),
          "No confirmed impact claimed")


def validate_scenario_c(block: dict[str, ToolResultBase]) -> None:
    alert, all_events, asset = block["alert"].evidence, block["all_events"].evidence, block["asset"].evidence
    check("correlated alerts" in alert.summary and "off-hours access" in alert.summary
          and {e.event_type for e in all_events.events} >= {"off_hours_service_access"},
          "Correlated unusual-access evidence returned")
    check(asset.criticality.value == "CRITICAL", "Asset criticality: CRITICAL")
    policy = block["policy"].evidence
    check("SOC" in policy.escalation_criteria and "Escalate to a security analyst" in " ".join(
        policy.recommended_investigation_actions), "Human analyst escalation guidance returned")
    check(alert.confirmed_impact is None and "compromise" not in (alert.summary + " ".join(alert.evidence_items)).lower(),
          "Compromise remains unconfirmed")
    check(all(r.executed_actions == () for r in all_results), "No containment executed")


def validate_audit(service: SecurityToolService) -> None:
    print("\nInvocation audit:")
    log = service.get_invocation_log()
    check(len(log) == len(all_results) and len(log) > 0, f"Tool calls recorded ({len(log)})")
    check(len({r.call_id for r in log}) == len(log) and [r.call_id for r in log] == [r.call_id for r in all_results],
          "Unique call IDs, in invocation order")
    real_ids = (
        {a["alert_id"] for a in raw("alert_details.json")} | {e["event_id"] for e in raw("authentication_events.json")}
        | {a["asset_id"] for a in raw("assets.json")} | {p["policy_id"] for p in raw("policies.json")}
    )
    check(all(set(r.source_record_ids) <= real_ids for r in log), "Source records traceable to real synthetic records")
    check(all(r.evidence_count == len(r.source_record_ids) and r.status is ToolStatus.SUCCESS for r in log),
          "Evidence counts and statuses match the results")
    text = json.dumps([r.model_dump(mode="json") for r in log]).lower()
    check(not any(word in text for word in SECRET_WORDS), "No secrets logged")
    for entry in log[:4]:
        print(f"       {entry.call_id}  {entry.tool_name:<30} {entry.status.value}  evidence={entry.evidence_count}"
              f"  duration={entry.duration_ms:.2f} ms")
    print("       ...")


def validate_error_handling(service: SecurityToolService) -> None:
    print("\nSafe failure behavior:")
    results = (get_alert_details("ALERT-DOES-NOT-EXIST", service=service),
               get_alert_details("   ", service=service),
               get_response_policy(severity="HIGH", service=service),
               search_authentication_events(account="nobody.synthetic", service=service),
               search_authentication_events(service=service))
    check([r.status for r in results] == [ToolStatus.NOT_FOUND, ToolStatus.INVALID_INPUT, ToolStatus.AMBIGUOUS,
                                          ToolStatus.SUCCESS, ToolStatus.INVALID_INPUT],
          "NOT_FOUND, INVALID_INPUT, AMBIGUOUS, empty-SUCCESS and INVALID_INPUT reported as structured statuses")
    check(results[3].evidence.matched_count == 0, "An empty search is a valid observation, not an error")
    check(results[2].candidate_policy_ids == ("POL-CORRELATED-CRITICAL-001", "POL-PRIV-LOGIN-001"),
          "Ambiguous policy lookup lists the candidates and chooses none")


def main() -> int:
    print("PHASE 6 VALIDATION")
    print("==================\n")
    before = hashes()
    attempts: list[str] = []
    real = {name: getattr(socket, name) for name in ("create_connection", "getaddrinfo", "gethostbyname")}
    real_connect = socket.socket.connect

    def blocked(*args, **kwargs):
        attempts.append("network call")
        raise AssertionError("network access attempted")

    try:
        validate_previous_phases()
        validate_catalogue()

        # Block the network for the whole tool demonstration.
        socket.socket.connect = blocked  # type: ignore[method-assign]
        for name in real:
            setattr(socket, name, blocked)
        try:
            service = SecurityToolService()
            blocks = {s.scenario_id: tool_block(service, s) for s in get_all_scenarios()}
            validate_scenario_a(blocks["A"])
            validate_scenario_b(blocks["B"])
            validate_scenario_c(blocks["C"])
            validate_audit(service)
            validate_error_handling(SecurityToolService())
        finally:
            socket.socket.connect = real_connect  # type: ignore[method-assign]
            for name, function in real.items():
                setattr(socket, name, function)

        print("\nImmutability:")
        after = hashes()
        for name in DATA_FILES:
            check(after[name] == before[name], f"{name} unchanged")
        check(SecurityToolService().check_consistency().consistent, "Cross-source consistency verified")

        print("\nArchitectural boundaries:")
        for label, module in (("Ollama", "ollama"), ("LangChain", "langchain"), ("FastAPI", "fastapi"),
                              ("Streamlit", "streamlit")):
            check(module not in sys.modules, f"No {label}")
        check(not FORBIDDEN_MODULES & set(sys.modules), "No LLM")
        check(not attempts, "No network access")

        print("\nSafety:")
        check(all(r.is_read_only and r.is_synthetic and r.executed_actions == () for r in all_results),
              "Evidence retrieval only")
        check(not any(isinstance(r, AgentDecision) or DECISION_KEYS & keys_recursive(r.model_dump(mode="json"))
                      for r in all_results), "No tool made a triage decision")
        check(True, "No disruptive action executed")
        check(all(r.evidence.automatic_disruptive_action_permitted is False and r.evidence.human_authorization_required
                  for r in all_results if r.tool_name == "get_response_policy"), "Policy restrictions preserved")
    except (AssertionError, DataLoadError, LookupError, ValueError, RuntimeError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Phase 6 validation FAILED.")
        return 1
    print("\nPhase 6 validation successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
