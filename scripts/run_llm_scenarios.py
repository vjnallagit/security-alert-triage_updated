"""Run Scenarios A, B and C through the LIVE Ollama LLM agent (needs a local Ollama runtime).

    export RUN_OLLAMA_INTEGRATION=true OLLAMA_MODEL=qwen3:4b
    python scripts/run_llm_scenarios.py [--output-dir <folder>] [--debug]

Shows the model's actual tool-call sequence, evidence, source IDs and final recommendation for each scenario.
Performs no remediation and does not modify source data. An optional JSON report is written to the folder
you choose (never inside the project's data directory); it contains no secrets.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import live_common  # noqa: E402  (also puts the project root on sys.path)
from agents.llm_agent import LLMAgent  # noqa: E402
from services.data_loader import DATA_DIR  # noqa: E402
from services.llm_orchestrator import LLMOrchestrator  # noqa: E402
from services.ollama_chat_adapter import OllamaChatModelAdapter  # noqa: E402
from services.scenario_service import get_all_scenarios  # noqa: E402


def data_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA_DIR.glob("*.json"))}


def safe_output_dir(raw: str) -> Path:
    target = Path(raw).expanduser().resolve()
    if target == DATA_DIR.resolve() or DATA_DIR.resolve() in target.parents:
        raise ValueError("The report folder must not be inside the project's data directory.")
    target.mkdir(parents=True, exist_ok=True)
    return target


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", help="Optional folder for a JSON report.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    print("LLM AGENT: LIVE SCENARIO RUN")
    print("============================\n")
    if not live_common.require_integration_mode():
        return 2
    debug, config = live_common.debug_enabled(argv), None
    before = data_hashes()
    report: dict[str, object] = {"generated_at": datetime.now(UTC).isoformat(), "scenarios": {}}
    failures = 0
    try:
        config = live_common.load_live_config()
        live_common.print_safe_config(config)
        report["configuration"] = config.safe_summary()
        out_dir = safe_output_dir(args.output_dir) if args.output_dir else None
        adapter = OllamaChatModelAdapter(config)
        for scenario in get_all_scenarios():
            started = time.perf_counter()
            try:
                result = LLMOrchestrator(adapter, config).run(scenario)
            except Exception as exc:  # one failing scenario must not hide the others
                failures += 1
                live_common.report_failure(exc, config, debug)
                report["scenarios"][scenario.scenario_id] = {"error": f"{type(exc).__name__}: {exc}"}  # type: ignore[index]
                continue
            elapsed = time.perf_counter() - started
            live_common.describe_run(scenario, result, elapsed)
            decision = LLMAgent._to_decision(scenario, result)
            report["scenarios"][scenario.scenario_id] = {  # type: ignore[index]
                "elapsed_seconds": round(elapsed, 2), "decision": decision.model_dump(mode="json")}
        if data_hashes() != before:
            raise AssertionError("A source data file changed during the run.")
        if out_dir is not None:
            path = out_dir / f"llm_scenarios_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nReport written to {path}")
    except Exception as exc:
        live_common.report_failure(exc, config, debug)
        return 1
    print("\nNo remediation was performed and no source data changed.")
    print(f"Live scenario run finished with {failures} failed scenario(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
