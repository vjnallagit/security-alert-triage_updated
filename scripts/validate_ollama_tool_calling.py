"""Validate GENUINE model-selected tool calling on Scenario A (needs a local Ollama runtime).

    export RUN_OLLAMA_INTEGRATION=true OLLAMA_MODEL=qwen3:4b
    python scripts/validate_ollama_tool_calling.py [--debug]

Nothing is pre-scripted: the live model must choose its own tool calls. The script then requires at least two
distinct relevant tools, validates grounding and guardrails, and shows the final structured recommendation.
"""

import sys
import time

import live_common  # noqa: E402  (also puts the project root on sys.path)
from services.llm_orchestrator import LLMOrchestrator  # noqa: E402
from services.ollama_chat_adapter import OllamaChatModelAdapter  # noqa: E402
from services.scenario_service import get_scenario  # noqa: E402


def main(argv: list[str]) -> int:
    print("OLLAMA TOOL-CALLING VALIDATION (Scenario A)")
    print("===========================================\n")
    if not live_common.require_integration_mode():
        return 2
    debug, config = live_common.debug_enabled(argv), None
    try:
        config = live_common.load_live_config()
        live_common.print_safe_config(config)
        scenario = get_scenario("A")
        started = time.perf_counter()
        result = LLMOrchestrator(OllamaChatModelAdapter(config), config).run(scenario)
        live_common.describe_run(scenario, result, time.perf_counter() - started)
        distinct = result.collector.evidence_tool_names
        if len(distinct) < 2:
            raise AssertionError(f"Only {len(distinct)} distinct relevant tool(s) were used; two are required.")
        failed = [c.name for c in result.grounding_results + result.guardrail_results if not c.passed]
        if failed:
            raise AssertionError(f"Checks failed: {', '.join(failed)}")
    except Exception as exc:
        live_common.report_failure(exc, config, debug)
        return 1
    print("\n[PASS] The live model selected at least two distinct relevant tools")
    print("[PASS] Grounding validated   [PASS] Guardrails validated   [PASS] No action executed")
    print("\nOllama tool-calling validation successful for Scenario A.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
