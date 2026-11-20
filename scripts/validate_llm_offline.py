"""Offline validation of the LLM agent architecture (Phase 7). Safe to run without Ollama.

Run from anywhere:  python scripts/validate_llm_offline.py

It uses a scripted FAKE tool-calling model with the REAL Phase 6 tools, the real orchestrator, real grounding
checks and real guardrails. Every network call is blocked while the scenarios run. This validates
orchestration and safety only; it does not validate any real Ollama model.
"""

import hashlib
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.llm_agent import LLMAgent  # noqa: E402
from config.llm_config import load_ollama_config  # noqa: E402
from models.schemas import AgentDecision  # noqa: E402
from services.data_loader import DATA_DIR  # noqa: E402
from services.llm_errors import LLMError  # noqa: E402
from services.scenario_service import get_all_scenarios  # noqa: E402
from devtools.mock_llm_runtime import happy_path_model  # noqa: E402  (development script: uses the dev mock)

DATA_FILES = ("scenarios.json", "alert_details.json", "authentication_events.json", "assets.json",
              "policies.json", "experience.json")
EXPECTED = {"A": ("HIGH", "INVESTIGATE"), "B": ("LOW", "COLLECT_MORE_EVIDENCE"),
            "C": ("CRITICAL", "ESCALATE_TO_SECURITY_ANALYST")}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def hashes() -> dict[str, str]:
    return {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() for name in DATA_FILES}


def evidence_for(decision: AgentDecision, tool_name: str) -> list[dict]:
    return [item for item in decision.metadata["structured_evidence"] if item["tool_name"] == tool_name]


def validate_previous_phases() -> None:
    print("Previous components:")
    # A subprocess keeps LangChain out of the process that Phase 6 inspects for forbidden imports.
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_phase6.py")],
                            capture_output=True, text=True, timeout=600)
    check(result.returncode == 0, "Phase 1-6 validations (including the existing evidence tools)")


def validate_scenario(decision: AgentDecision, scenario_id: str, scenario) -> None:
    print(f"\nScenario {scenario_id}:")
    called = [call["tool_name"] for call in decision.tool_calls]
    for name in dict.fromkeys(called):
        check(True, f"Fake model requested {name}")
    check(len(set(called)) >= 2, "At least two distinct tools called")
    check(all(call["status"] == "SUCCESS" for call in decision.tool_calls), "Real Phase 6 tools executed")
    check(all(r["passed"] for r in decision.metadata["grounding_validation"]), "Grounding validated against tool results")
    sources = set(decision.metadata["source_record_ids"])
    check({scenario.alert_id, scenario.asset_id, scenario.policy_id} <= sources, "Source IDs verified")
    priority, outcome = EXPECTED[scenario_id]
    check((decision.priority.value, decision.decision) == (priority, outcome), f"Final {priority} / {outcome}")

    search = evidence_for(decision, "search_authentication_events")[0]["asserted_facts"]
    asset = evidence_for(decision, "get_asset_criticality")[0]["asserted_facts"]
    if scenario_id == "A":
        check(search["failed_login_count"] == 12, "12 failed events grounded")
    elif scenario_id == "B":
        check(asset["asset_criticality"] == "LOW", "Limited evidence grounded (low-criticality asset)")
    else:
        check(asset["asset_criticality"] == "CRITICAL", "Critical asset grounded")
        check(bool(evidence_for(decision, "get_response_policy")), "Policy evidence grounded")
        check(not scenario.confirmed_impact and "not confirmed" in decision.uncertainty.lower(),
              "Compromise remains unconfirmed")


def validate_guardrails(decisions: list[AgentDecision]) -> None:
    print("\nGuardrails:")
    check(all(d.metadata["recommendation_only"] for d in decisions), "Recommendation-only")
    check(all(d.metadata["human_authorization_required_for_consequential_action"]
              and "human authorization" in d.audit_note for d in decisions), "Human authorization retained")
    check(all(d.metadata["executed_actions"] == [] for d in decisions), "No disruptive action executed")
    check(all(r["passed"] for d in decisions for r in d.metadata["guardrail_validation"]), "Safety guardrails passed")


def main() -> int:
    print("LLM OFFLINE VALIDATION")
    print("======================\n")
    before = hashes()
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def blocked(*args, **kwargs):
        attempts.append("network call")
        raise AssertionError("network access attempted")

    try:
        validate_previous_phases()

        print("\nRuntime:")
        config = load_ollama_config({})
        check(not config.integration_enabled, "Ollama integration disabled")
        socket.socket.connect = blocked  # type: ignore[method-assign]
        socket.create_connection = blocked  # type: ignore[assignment]
        decisions: list[AgentDecision] = []
        try:
            for scenario in get_all_scenarios():
                # Sequential turns make each model-chosen tool call visible in the trace.
                model = happy_path_model(scenario, parallel=False)
                decision = LLMAgent(model=model).decide(scenario)
                if not decisions:
                    check(decision.metadata["runtime_mode"] == "MOCK", "MOCK runtime (scripted fake model)")
                decisions.append(decision)
                validate_scenario(decision, scenario.scenario_id, scenario)
        finally:
            socket.socket.connect = real_connect  # type: ignore[method-assign]
            socket.create_connection = real_create

        validate_guardrails(decisions)

        print("\nIsolation:")
        check(not attempts, "No network connection attempted")
        check("ollama" not in sys.modules and "langchain_ollama" not in sys.modules,
              "Ollama connection not attempted (the Ollama client was never loaded)")
        after = hashes()
        for name in DATA_FILES:
            check(after[name] == before[name], f"{name} unchanged")
    except (AssertionError, LLMError, LookupError, ValueError, RuntimeError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Offline LLM validation FAILED.")
        return 1

    print("\nOffline LLM validation successful.")
    print("\nThis validates orchestration using a mock model.")
    print("It does not validate a real Ollama model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
