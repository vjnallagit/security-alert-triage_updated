"""Six-agent comparison validation. Runs without opening Streamlit and without Ollama.

Run from anywhere:  python scripts/validate_comparison.py

The LLM Agent runs on the scripted MOCK model. Every network call is blocked while the comparison runs, and
the SHA-256 of every synthetic data file is compared before and after. This validates the comparison
architecture; it does not validate a real Ollama model.
"""

import hashlib
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.comparison_models import AGENT_ORDER, AgentComparisonResult, AgentStatus  # noqa: E402
from models.llm_models import RuntimeMode  # noqa: E402
from models.schemas import AgentType  # noqa: E402
from services.comparison_service import AgentComparisonService  # noqa: E402  (type only)
from services.data_loader import DATA_DIR  # noqa: E402
from services.llm_runtime import build_comparison_service  # noqa: E402
from services.scenario_service import get_all_scenarios  # noqa: E402
from ui.formatters import runtime_label  # noqa: E402

EXPECTED = {
    "A": ("HIGH", "INVESTIGATE"),
    "B": ("LOW", "COLLECT_MORE_EVIDENCE"),
    "C": ("CRITICAL", "ESCALATE_TO_SECURITY_ANALYST"),
}
FORBIDDEN_MODULES = ("ollama", "langchain_ollama", "streamlit")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def data_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA_DIR.glob("*.json"))}


def print_table(result: AgentComparisonResult) -> None:
    for entry in result.results:
        name = entry.agent_name + (" [MOCK]" if entry.agent_type is AgentType.LLM else "")
        print(f"{name:<22} {entry.priority.value:<9} {entry.decision}")
    print()


def validate_scenario(result: AgentComparisonResult, scenario_id: str) -> None:
    print(f"Scenario {scenario_id}\n")
    print_table(result)
    check(tuple(e.agent_type for e in result.results) == AGENT_ORDER and result.all_succeeded, "Six agents evaluated")
    check({e.input_fingerprint for e in result.results} == {result.scenario_fingerprint}, "Same scenario fingerprint")
    required = all(
        e.priority and e.decision and e.evidence and e.uncertainty.strip() and e.likely_interpretation.strip()
        and e.recommended_actions and e.audit_note.strip() for e in result.results)
    check(required, "Required assignment output present")
    llm = result.entry(AgentType.LLM)
    check(result.llm_runtime_mode is RuntimeMode.MOCK and llm.llm_runtime_mode is RuntimeMode.MOCK
          and runtime_label(llm) == "MOCK", "LLM mode truthfully marked MOCK")
    priority, decision = EXPECTED[scenario_id]
    check(all((e.priority.value, e.decision) == (priority, decision) for e in result.results),
          f"All agents agree: {priority} / {decision} (taken from real agent execution)")
    print()


def main() -> int:
    print("SIX-AGENT COMPARISON VALIDATION")
    print("===============================\n")
    before = data_hashes()
    attempts: list[str] = []
    real_connect, real_create = socket.socket.connect, socket.create_connection

    def blocked(*args, **kwargs):
        attempts.append("network call")
        raise AssertionError("network access attempted")

    try:
        scenarios = get_all_scenarios()
        check([s.scenario_id for s in scenarios] == ["A", "B", "C"], "Loaded scenarios A, B and C")
        print()
        service: AgentComparisonService = build_comparison_service(
            RuntimeMode.MOCK, {"LLM_RUNTIME_MODE": "MOCK", "ENABLE_MOCK_LLM": "true"})  # explicit development gate
        socket.socket.connect = blocked  # type: ignore[method-assign]
        socket.create_connection = blocked  # type: ignore[assignment]
        try:
            for scenario in scenarios:
                validate_scenario(service.compare(scenario), scenario.scenario_id)

            print("Repeatability\n")
            scenario_a = scenarios[0]
            runs = [service.compare(scenario_a) for _ in range(3)]
            states = [r.entry(AgentType.MODEL_BASED_REFLEX).special_details["state_summary"] for r in runs]
            check(all(s["processed_scenario_count"] == 1 and s["account_count"] == 1 for s in states),
                  "Main comparison uses fresh Model-Based state")
            check(len({r.scenario_fingerprint for r in runs}) == 1, "Scenario fingerprint is stable across runs")
            check(all(r.entry(AgentType.MODEL_BASED_REFLEX).decision == "INVESTIGATE" for r in runs),
                  "Repeated clicks do not escalate the Model-Based Agent")
        finally:
            socket.socket.connect = real_connect  # type: ignore[method-assign]
            socket.create_connection = real_create

        after = data_hashes()
        check(after["experience.json"] == before["experience.json"], "Learning seed unchanged")
        check(after == before, "Source files unchanged")
        check(not attempts, "No network connection attempted")
        check(not any(module in sys.modules for module in FORBIDDEN_MODULES),
              "Ollama not contacted (no Ollama client or Streamlit loaded)")
        check(all(e.status is AgentStatus.SUCCESS for e in service.compare(scenario_a).results),
              "No agent reported an error")
    except (AssertionError, LookupError, ValueError, RuntimeError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Comparison validation FAILED.")
        return 1

    print("\nComparison validation successful.")
    print("\nThe LLM Agent ran on a deterministic MOCK model. This does not validate a real Ollama model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
