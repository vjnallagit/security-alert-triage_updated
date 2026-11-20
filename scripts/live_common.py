"""Shared helpers for the live Ollama scripts. Nothing here runs on import."""

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.llm_config import OllamaConfig, load_ollama_config  # noqa: E402
from models.schemas import SecurityScenario  # noqa: E402
from services.llm_errors import LLMError  # noqa: E402
from services.llm_orchestrator import OrchestrationResult  # noqa: E402

NEXT_CHECKS = (
    "ollama --version",
    "ollama list            (is the model in OLLAMA_MODEL pulled?)",
    "ollama serve           (or restart the Ollama application)",
)


def debug_enabled(argv: list[str]) -> bool:
    return "--debug" in argv or os.environ.get("OLLAMA_DEBUG", "").strip().lower() == "true"


def require_integration_mode() -> bool:
    if os.environ.get("RUN_OLLAMA_INTEGRATION", "").strip().lower() == "true":
        return True
    print("Live Ollama mode is not enabled. This script talks to a local Ollama server.")
    print("Enable it explicitly by setting RUN_OLLAMA_INTEGRATION=true in the environment")
    print("(Linux shell: export RUN_OLLAMA_INTEGRATION=true).")
    print("Run it on the machine hosting the local Ollama service; see docs/local_ollama_setup.md.")
    return False


def load_live_config() -> OllamaConfig:
    return load_ollama_config(integration_enabled=True)


def print_safe_config(config: OllamaConfig) -> None:
    print("Configuration (no secrets are used or printed):")
    for key, value in config.safe_summary().items():
        print(f"  {key}: {value}")


def report_failure(error: Exception, config: OllamaConfig | None, debug: bool) -> None:
    """High-level message and next steps; a stack trace only when debug mode is explicitly enabled."""
    print(f"\n[FAIL] {type(error).__name__}: {error}")
    if config is not None:
        print(f"Configured base URL: {config.base_url}")
    if isinstance(error, LLMError):
        print("Next checks:")
        for line in NEXT_CHECKS:
            print(f"  {line}")
    if debug:
        traceback.print_exception(error)
    else:
        print("(Re-run with --debug or OLLAMA_DEBUG=true to see the technical detail.)")


def describe_run(scenario: SecurityScenario, result: OrchestrationResult, elapsed_seconds: float) -> None:
    recommendation, trace = result.recommendation, result.trace
    print(f"\n=== Scenario {scenario.scenario_id}: {scenario.title} ===")
    print(f"Model: {trace.model_name} ({trace.runtime_mode.value}) | elapsed {elapsed_seconds:.1f}s | "
          f"model turns {trace.iterations}")
    print("Tool-call sequence chosen by the model:")
    results = {r.tool_call_id: r for r in trace.tool_results}
    for number, call in enumerate(trace.tool_calls, start=1):
        outcome = results.get(call.tool_call_id)
        print(f"  {number}. {call.tool_name}({call.arguments}) -> {call.status.value}")
        if outcome and outcome.summary:
            print(f"       evidence: {outcome.summary}")
        print(f"       source IDs: {', '.join(call.source_record_ids) or '(none)'}")
    print(f"Distinct relevant evidence tools: {', '.join(result.collector.evidence_tool_names)}")
    print(f"Source record IDs: {', '.join(trace.source_record_ids)}")
    print(f"Priority: {recommendation.priority.value}")
    print(f"Decision: {recommendation.decision}")
    print(f"Uncertainty: {recommendation.uncertainty}")
    print(f"Likely interpretation: {recommendation.likely_interpretation}")
    print("Recommended actions:")
    for action in recommendation.recommended_actions:
        print(f"  - {action}")
    print(f"Audit note: {recommendation.audit_note}")
    checks = result.grounding_results + result.guardrail_results
    print(f"Grounding and guardrail checks passed: {sum(c.passed for c in checks)}/{len(checks)}")
