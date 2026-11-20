"""Submission-readiness validation. Runs without Ollama, on any platform:  python scripts/validate_submission_readiness.py

Checks that the repository is ready to copy and run: production code is independent of tests and
the dev mock, the runtime rules hold, live failures cannot fall back to a mock, thinking content is discarded,
nothing sensitive is committed, and no live result has been fabricated. It does NOT validate a real Ollama model.
"""

import ast
import hashlib
import logging
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.llm_config import OllamaConfig  # noqa: E402
from config.runtime_config import LLMRuntimeMode, RuntimeConfig, load_runtime_config  # noqa: E402
from models.llm_models import RuntimeMode  # noqa: E402
from models.schemas import AgentType  # noqa: E402
from services.data_loader import DATA_DIR  # noqa: E402
from services.llm_errors import LLMConfigurationError, LLMConnectionError  # noqa: E402
from services.llm_orchestrator import sanitize_reply  # noqa: E402
from services.llm_runtime import build_comparison_service, inspect_runtime  # noqa: E402
from services.scenario_service import get_scenario  # noqa: E402

PRODUCTION_DIRS = ("agents", "config", "models", "services", "tools", "ui")
SUBMISSION_ENV = {
    "LLM_RUNTIME_MODE": "LIVE", "ENABLE_MOCK_LLM": "false", "RUN_OLLAMA_INTEGRATION": "true",
    "SUBMISSION_MODE": "true", "OLLAMA_MODEL": "qwen3:4b",
}
REQUIRED_FILES = (
    "app.py", "requirements.txt", "README.md", ".env.example", ".gitignore", "pytest.ini",
    ".streamlit/config.toml", "docs/local_ollama_setup.md", "docs/test_results.md",
    "devtools/mock_llm_runtime.py", "services/llm_runtime.py", "config/runtime_config.py",
    "tests/fakes/fake_tool_calling_model.py", "tests/test_ollama_integration.py",
    "scripts/validate_ollama_connection.py", "scripts/validate_ollama_tool_calling.py", "scripts/run_llm_scenarios.py",
)
SECRET_ASSIGNMENT = re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret|bearer)\b\s*[=:]\s*[^\s<>$\"'`]+")
HOST_OR_USER = re.compile(r"[A-Za-z]:\\Users\\|/home/(?!<)|\b[\w.-]+@[\w-]+\.(?:com|org|net|edu|ac\.in)\b", re.I)
ALLOWED_IPS = ("127.0.0.1", "0.0.0.0", "203.0.113.")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def production_files() -> list[Path]:
    return [ROOT / "app.py", *(p for d in PRODUCTION_DIRS for p in sorted((ROOT / d).glob("*.py")))]


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def data_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA_DIR.glob("*.json"))}


def env_example() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def text_files() -> list[Path]:
    paths = [ROOT / "README.md", ROOT / ".env.example", ROOT / "requirements.txt", ROOT / ".streamlit" / "config.toml"]
    return [*paths, *sorted((ROOT / "docs").glob("*.md"))]


def count_collected_integration_tests() -> int:
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_ollama_integration.py", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    found = re.search(r"(\d+)(?:/\d+)? tests? collected", run.stdout)
    return int(found.group(1)) if found else 0


def main() -> int:
    logging.getLogger("services.comparison_service").setLevel(logging.ERROR)  # hide the expected simulated failure
    print("SUBMISSION READINESS VALIDATION")
    print("===============================\n")
    before = data_hashes()
    try:
        # Production/test separation and the explicit mock gate.
        offenders = [
            p.name for p in production_files()
            if "tests" in imported_roots(p) or "tests.fakes" in p.read_text(encoding="utf-8")
            or ("devtools" in imported_roots(p))  # the mock is loaded by name, only behind the gate in llm_runtime
        ]
        check(not offenders, "Production runtime imports no test fakes")
        check("devtools" not in sys.modules, "MOCK retained only for explicit development/testing (never loaded here)")

        # Submission mode.
        live = load_runtime_config(SUBMISSION_ENV)
        check(live.mode is LLMRuntimeMode.LIVE and live.submission_mode, "Submission mode requires LIVE")
        for rejected in ({**SUBMISSION_ENV, "LLM_RUNTIME_MODE": "MOCK", "ENABLE_MOCK_LLM": "true"},
                         {**SUBMISSION_ENV, "LLM_RUNTIME_MODE": "DISABLED"}):
            try:
                load_runtime_config(rejected)
            except LLMConfigurationError:
                continue
            raise AssertionError("submission mode accepted a non-LIVE runtime")
        check(True, "Submission mode rejects MOCK")
        status = inspect_runtime(SUBMISSION_ENV)
        check(status.config_error is None and status.available == (RuntimeMode.OLLAMA,),
              "Streamlit live-mode path exists (submission offers OLLAMA only)")
        app_text = (ROOT / "app.py").read_text(encoding="utf-8")
        components = (ROOT / "ui" / "components.py").read_text(encoding="utf-8")
        check("build_comparison_service" in app_text and "Retry LLM Agent" in components and "spinner" in app_text,
              "Streamlit live path has a spinner, a retry and an injected runtime")

        # Committed defaults and Ollama settings.
        example = env_example()
        check(example["RUN_OLLAMA_INTEGRATION"] == "false" and example["SUBMISSION_MODE"] == "false"
              and example["ENABLE_MOCK_LLM"] == "false" and RuntimeConfig().run_ollama_integration is False
              and OllamaConfig().integration_enabled is False, "Default committed configuration does not enable integration")
        check(OllamaConfig().base_url == "http://127.0.0.1:11434" == example["OLLAMA_BASE_URL"], "Ollama endpoint is localhost")
        guide = (ROOT / "docs" / "local_ollama_setup.md").read_text(encoding="utf-8")
        check("qwen3:4b" in guide and example["OLLAMA_MODEL"] == "qwen3:4b", "qwen3:4b configured and documented")
        check(OllamaConfig(request_timeout_seconds=300).request_timeout_seconds == 300
              and example["OLLAMA_REQUEST_TIMEOUT_SECONDS"] == "300" and "300 seconds" in guide,
              "CPU timeout documented as 300 seconds")

        # No live-to-mock fallback (a simulated unreachable adapter: no Ollama is contacted).
        import agents.llm_agent as llm_agent_module

        class Unreachable:
            def __init__(self, config: OllamaConfig) -> None:
                self.runtime_mode, self.model_name = RuntimeMode.OLLAMA, config.model

            def bind_tools(self, _tools: object) -> object:
                raise LLMConnectionError("simulated: server unreachable")

        original = llm_agent_module.OllamaChatModelAdapter
        llm_agent_module.OllamaChatModelAdapter = Unreachable  # type: ignore[assignment,misc]
        try:
            result = build_comparison_service(RuntimeMode.OLLAMA, SUBMISSION_ENV).compare(get_scenario("A"))
        finally:
            llm_agent_module.OllamaChatModelAdapter = original  # type: ignore[assignment,misc]
        llm = result.entry(AgentType.LLM)
        check(llm.status.value == "ERROR" and llm.decision_record is None and len(result.successful) == 5
              and "devtools" not in sys.modules, "No mock fallback in LIVE (error shown, five classical results kept)")

        # Thinking protection.
        from langchain_core.messages import AIMessage

        leaked = AIMessage(content="<think>SECRET-THOUGHT</think>ok", additional_kwargs={"reasoning_content": "SECRET-THOUGHT"},
                           response_metadata={"thinking": "SECRET-THOUGHT"})
        check("SECRET-THOUGHT" not in repr(sanitize_reply(leaked).model_dump()), "Thinking content excluded")

        # Tests retained.
        counts = {}
        for name in ("test_llm_orchestrator.py", "test_llm_guardrails.py", "test_llm_agent.py"):
            tree = ast.parse((ROOT / "tests" / name).read_text(encoding="utf-8"))
            counts[name] = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))
        check(all(c > 10 for c in counts.values()), "Mock tests retained for offline regression")
        collected = count_collected_integration_tests()
        check(collected == 24, f"Integration tests retained ({collected} collected; skipped unless explicitly enabled)")

        # Documentation, secrets and honesty.
        results = (ROOT / "docs" / "test_results.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        check("Pending live run" in results and "validation passed" not in readme.lower(), "Live results not fabricated")
        check("Pending live Ollama validation" in readme, "README describes pending live validation truthfully")
        check(all((ROOT / "docs" / n).exists() for n in ("local_ollama_setup.md", "test_results.md")),
              "Local Ollama setup guide and test_results.md exist")
        secrets = [f"{p.name}: {m.group(0)}" for p in text_files() for m in SECRET_ASSIGNMENT.finditer(p.read_text(encoding="utf-8"))]
        check(not secrets, "No credentials in configuration or documentation")
        sensitive = [p.name for p in text_files() if HOST_OR_USER.search(p.read_text(encoding="utf-8"))]
        stray_ips = [p.name for p in text_files() for ip in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", p.read_text(encoding="utf-8"))
                     if not ip.startswith(ALLOWED_IPS)]
        check(not sensitive and not stray_ips, "No machine-specific paths, user IDs, hostnames or stray IP addresses")
        check(not (ROOT / ".env").exists(), "No .env file is committed")

        # Immutability and readiness to copy.
        missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
        check(not missing, "All files needed to run the project are present")
        check(data_hashes() == before, "Source data unchanged")
    except (AssertionError, KeyError, LLMConfigurationError, ValueError, OSError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Submission-readiness validation FAILED.")
        return 1

    print("\nSubmission-readiness validation successful.")
    print("\nThe repository is ready to copy.")
    print("Real Ollama validation remains required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
