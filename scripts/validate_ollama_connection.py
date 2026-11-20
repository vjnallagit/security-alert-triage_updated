"""Check the local Ollama connection (needs a local Ollama runtime).

    export RUN_OLLAMA_INTEGRATION=true OLLAMA_MODEL=qwen3:4b
    python scripts/validate_ollama_connection.py [--debug]

Sends one harmless plain prompt and checks that tools can be bound. It runs no triage and no scenario data.
"""

import sys

import live_common  # noqa: E402  (also puts the project root on sys.path)
from services.llm_errors import LLMError  # noqa: E402
from services.ollama_chat_adapter import OllamaChatModelAdapter  # noqa: E402
from tools.langchain_security_tools import SecurityToolbox  # noqa: E402


def main(argv: list[str]) -> int:
    print("OLLAMA CONNECTION VALIDATION")
    print("============================\n")
    if not live_common.require_integration_mode():
        return 2
    debug, config = live_common.debug_enabled(argv), None
    try:
        config = live_common.load_live_config()
        live_common.print_safe_config(config)
        adapter = OllamaChatModelAdapter(config)  # ChatOllama is created only when first used
        reply = adapter.ask("Reply with the single word: ready")
        if not reply.strip():
            raise LLMError("The model returned an empty response.")
        print(f"\n[PASS] Basic prompt answered ({len(reply.strip())} characters)")
        adapter.bind_tools(SecurityToolbox().tools)
        print("[PASS] bind_tools succeeded for the four security tools")
    except Exception as exc:  # reported with next checks; no stack trace unless debug is on
        live_common.report_failure(exc, config, debug)
        return 1
    print("\nOllama connection validation successful.")
    print("This does not yet prove reliable tool calling; run validate_ollama_tool_calling.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
