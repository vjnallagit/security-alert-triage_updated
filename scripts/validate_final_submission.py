"""Final submission validation. Runs without a live model, on any platform:
    python scripts/validate_final_submission.py

It checks that the five classical agents, the LLM agent, the four tools, the three scenarios, the comparison
service, the Streamlit app and the documentation are all present and consistent, that no environment-specific
wording or secrets are in the repository, that source data is unchanged, and that no live result is claimed
without a live run. It does NOT validate a real Ollama model.
"""

import ast
import getpass
import hashlib
import inspect
import logging
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import (  # noqa: E402
    GoalBasedAgent,
    LearningAgent,
    LLMAgent,
    ModelBasedReflexAgent,
    SimpleReflexAgent,
    UtilityBasedAgent,
)
from agents.base import BaseAgent  # noqa: E402
from config.llm_config import OllamaConfig  # noqa: E402
from models.comparison_models import AGENT_ORDER, scenario_fingerprint  # noqa: E402
from models.learning_models import FeedbackOutcome  # noqa: E402
from models.llm_models import RuntimeMode  # noqa: E402
from models.schemas import AgentType, SecurityScenario  # noqa: E402
from services.comparison_service import AgentComparisonService, LearningFeedbackDemo  # noqa: E402
from services.data_loader import DATA_DIR  # noqa: E402
from services.evidence_collector import evaluate_grounding  # noqa: E402
from services.llm_guardrails import evaluate_guardrails  # noqa: E402
from services.llm_runtime import build_comparison_service  # noqa: E402
from services.scenario_service import get_all_scenarios  # noqa: E402
from tools.langchain_security_tools import TOOL_NAMES  # noqa: E402

EXPECTED_TOOLS = ("get_alert_details", "search_authentication_events", "get_asset_criticality", "get_response_policy")
CLASSICAL = {
    "agents/simple_reflex.py": SimpleReflexAgent, "agents/model_based_reflex.py": ModelBasedReflexAgent,
    "agents/goal_based.py": GoalBasedAgent, "agents/utility_based.py": UtilityBasedAgent,
    "agents/learning_agent.py": LearningAgent,
}
README_SECTIONS = (
    "1. Overview", "2. What the Application Produces", "3. Architecture", "4. Shared Security Scenarios",
    "5. Agent Comparison", "6. Five Classical Agents", "7. Security Evidence Tools", "8. LLM Agent",
    "9. Local Ollama Setup", "10. Streamlit Interface", "11. Safety", "12. Testing", "13. Running the Application",
    "14. Screenshots", "15. Limitations", "16. Project Structure", "17. Assignment Rubric Mapping",
)
RUBRIC_ROWS = ("Five classical agents", "LLM + tool calling", "Shared scenarios/testing", "Streamlit/comparison",
               "Learning/evaluation", "Documentation/demo")
OUTPUT_FIELDS = ("priority", "decision", "reasoning_summary", "evidence", "uncertainty", "likely_interpretation",
                 "recommended_actions", "audit_note")
# Line-ending-normalised SHA-256 prefixes (a Windows or Linux checkout gives the same values).
PINNED_DATA = {
    "alert_details.json": "3f2075f226e7b4e7", "assets.json": "765340ef0b625030",
    "authentication_events.json": "d2267c262468b686", "experience.json": "1f331499efc4d323",
    "policies.json": "6b4d8be50b390afb", "scenarios.json": "c1a8784807d01372",
}
# Built from fragments so this file does not itself contain the identifiers it searches for.
CASE_INSENSITIVE_TERMS = ["la" + "ptop", "inv" + "esco", "corp" + "orate", "ub" + "untu", "ama" + "zon",
                          "ep" + "yc", "v" + "cpu", "uni" + "versity"]
CASE_SENSITIVE_TERMS = [r"\bBI" + r"TS\b", r"\bV" + r"M\b"]
MARKETING_WORDS = ("robust", "production-grade", "enterprise-grade", "world-class", "non-negotiable", "20+ years",
                   "principal engineer", "this demonstrates", "the system ensures", "guaranteed")
SECRET_ASSIGNMENT = re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret|bearer)\b\s*[=:]\s*[^\s<>$\"'`]+")
SECRET_LITERAL = re.compile(r"(?i)\b\w*(api[_-]?key|token|password|passwd|secret)\w*\s*=\s*[\"'][^\"'\s]{8,}[\"']")
PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
GENERIC_PATH = re.compile(r"[A-Za-z]:\\Users\\|/home/(?!<)")
ALLOWED_IPS = ("127.0.0.1", "0.0.0.0", "203.0.113.", "198.51.100.", "192.0.2.")
SKIP_PARTS = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".ini", ".example", ".json", ".cfg"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def repo_text_files() -> list[Path]:
    return [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES and not SKIP_PARTS & set(p.parts)]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def data_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA_DIR.glob("*.json"))}


def normalised_prefixes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
            for p in sorted(DATA_DIR.glob("*.json"))}


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_function_names(relative: str, prefix: str) -> list[str]:
    tree = ast.parse(read(relative))
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith(prefix)]


def collected_integration_tests() -> int:
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_ollama_integration.py", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    found = re.search(r"(\d+)(?:/\d+)? tests? collected", run.stdout)
    return int(found.group(1)) if found else 0


def validate_agents_and_tools(scenarios: list[SecurityScenario]) -> None:
    check(all((ROOT / r).exists() and issubclass(c, BaseAgent) for r, c in CLASSICAL.items()), "Five classical agents")
    check(LLMAgent.__bases__ == (BaseAgent,), "LLM agent")
    check(TOOL_NAMES == EXPECTED_TOOLS and len(TOOL_NAMES) == 4, "Four evidence tools")
    check([s.scenario_id for s in scenarios] == ["A", "B", "C"], "Three shared scenarios")
    parameters = list(inspect.signature(BaseAgent.decide).parameters.values())
    check(all(type(s) is SecurityScenario for s in scenarios) and parameters[1].annotation is SecurityScenario,
          "Shared scenario contract (one SecurityScenario type for every agent)")


def validate_comparison(scenarios: list[SecurityScenario]) -> None:
    check(tuple(AGENT_ORDER) == (AgentType.SIMPLE_REFLEX, AgentType.MODEL_BASED_REFLEX, AgentType.GOAL_BASED,
                                 AgentType.UTILITY_BASED, AgentType.LEARNING, AgentType.LLM), "Six-agent comparison")
    # The scripted stand-in is used only to check the six-agent output contract without a live model.
    service = build_comparison_service(RuntimeMode.MOCK, {"LLM_RUNTIME_MODE": "MOCK", "ENABLE_MOCK_LLM": "true"})
    for scenario in scenarios:
        result = service.compare(scenario)
        check(result.all_succeeded and len(result.results) == 6
              and {e.input_fingerprint for e in result.results} == {scenario_fingerprint(scenario)},
              f"Scenario {scenario.scenario_id}: six agents received an identical scenario fingerprint")
        for entry in result.results:
            missing = [f for f in OUTPUT_FIELDS if not getattr(entry, f)]
            if missing:
                raise AssertionError(f"{entry.agent_name} on {scenario.scenario_id} is missing: {missing}")
        print(f"[PASS] Assignment output contract: all six agents on {scenario.scenario_id}")
    disabled = AgentComparisonService(llm_mode=None, llm_unavailable_reason="off").compare(scenarios[0])
    check(len(disabled.successful) == 5, "Five classical results survive an unavailable LLM Agent")


def validate_learning() -> None:
    before = data_hashes()["experience.json"]
    scenario = get_all_scenarios()[0]
    demo = LearningFeedbackDemo()
    try:
        start = demo.decide(scenario).decision
        step = None
        for _ in range(4):
            step = demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
        changed = demo.decide(scenario).decision
    finally:
        demo.close()
    check(step is not None and step.after_failure == step.before_failure + 1 and start == "INVESTIGATE"
          and changed == "ESCALATE_TO_SECURITY_ANALYST", "Learning demonstration (history updated, decision changes)")
    check(hasattr(LearningAgent, "record_feedback") and hasattr(LearningAgent, "learn"), "Learning feedback mechanisms")
    check(data_hashes()["experience.json"] == before, "Committed experience file untouched by the demonstration")


def validate_llm_rules() -> None:
    requirements = read("requirements.txt").lower()
    check("langchain-core" in requirements and "langchain-ollama" in requirements and "streamlit" in requirements,
          "Local Ollama/LangChain configuration (dependencies declared)")
    try:
        OllamaConfig(minimum_tool_calls=1)
        enforced = False
    except ValueError:
        enforced = True
    minimum_tool_tests = test_function_names("tests/test_llm_orchestrator.py", "test_mt_")
    check(enforced and len(minimum_tool_tests) >= 8, "2+ relevant tool enforcement (rule exists and is tested)")
    grounding_tests = test_function_names("tests/test_llm_orchestrator.py", "test_lg_")
    check(callable(evaluate_grounding) and len(grounding_tests) >= 10, "Evidence grounding (validator exists and is tested)")
    guardrail_tests = test_function_names("tests/test_llm_guardrails.py", "test_gr_")
    check(callable(evaluate_guardrails) and len(guardrail_tests) >= 10, "Safety guardrails (rules exist and are tested)")
    check(collected_integration_tests() == 24, "24 integration tests retained (collected, not run)")


def validate_documents() -> None:
    check((ROOT / "app.py").exists() and (ROOT / "ui" / "components.py").exists(), "Streamlit UI")
    readme = read("README.md")
    headings = re.findall(r"^## (.+)$", readme, re.M)
    missing = [s for s in README_SECTIONS if s not in headings]
    check(not missing, f"README (all {len(README_SECTIONS)} sections)" + (f" missing: {missing}" if missing else ""))
    lowered = readme.lower()
    check(not [w for w in MARKETING_WORDS if w in lowered], "README wording is plain (no marketing phrasing)")
    check(all(row in readme for row in RUBRIC_ROWS), "README rubric mapping")
    check("estimated marks" not in lowered and "full marks" not in lowered, "No marks are claimed")
    check((ROOT / "docs" / "submission_summary.md").exists(), "Submission summary")
    check((ROOT / "docs" / "demo_guide.md").exists(), "Demo guide")
    check((ROOT / "screenshots").is_dir(), "Screenshot directory")
    check((ROOT / "docs" / "test_results.md").exists() and (ROOT / "docs" / "local_ollama_setup.md").exists(),
          "Test documentation")


def validate_truthfulness() -> None:
    readme, results, summary = read("README.md"), read("docs/test_results.md"), read("docs/submission_summary.md")
    check("Pending live Ollama validation" in readme and "Pending live Ollama validation" in summary,
          "Offline/live distinction is stated truthfully")
    lines = results.splitlines()
    live_rows = [ln for ln in lines if ln.startswith("|") and any(
        key in ln for key in ("validate_ollama", "-m integration", "run_llm_scenarios"))]
    scenario_rows = [ln for ln in lines if re.match(r"\| [ABC] \|", ln)]
    check(len(live_rows) == 4 and len(scenario_rows) == 3 and all("Pending live run" in ln for ln in live_rows + scenario_rows),
          "No live results fabricated (every live row says 'Pending live run' until a real run)")
    images = [p for p in (ROOT / "screenshots").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    print(f"[INFO] {len(images)} screenshot image(s) present; images must come from a real run")


def validate_cleanliness() -> None:
    production = [ROOT / "app.py", *(p for d in ("agents", "config", "models", "services", "tools", "ui")
                                     for p in sorted((ROOT / d).glob("*.py")))]
    offenders = [p.name for p in production if {"tests", "devtools"} & imported_roots(p)
                 or "tests.fakes" in p.read_text(encoding="utf-8")]
    check(not offenders, "No production import from tests/ or the development stand-in")

    case_sensitive = re.compile("|".join(CASE_SENSITIVE_TERMS))
    case_insensitive = re.compile("|".join(CASE_INSENSITIVE_TERMS), re.IGNORECASE)
    user_names = {n.lower() for n in (getpass.getuser(), Path.home().name) if len(n) >= 4}
    hits = []
    for path in repo_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if case_sensitive.search(text) or case_insensitive.search(text) or any(u in text.lower() for u in user_names):
            hits.append(str(path.relative_to(ROOT)))
    check(not hits, "No environment-specific identifiers" + (f" (found in: {', '.join(hits)})" if hits else ""))

    problems = []
    for path in repo_text_files():
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PRIVATE_KEY.search(text):
            problems.append(f"{relative}: private key")
        # tests hold deliberately fake redaction fixtures, and source code names path patterns for redaction
        if relative.parts[0] != "tests" and relative.suffix != ".json":
            if relative.suffix == ".py":
                problems += [f"{relative}: {m.group(0)}" for m in SECRET_LITERAL.finditer(text)]
            else:
                problems += [f"{relative}: {m.group(0)}" for m in SECRET_ASSIGNMENT.finditer(text)]
                if GENERIC_PATH.search(text):
                    problems.append(f"{relative}: machine path")
        if relative.suffix == ".md" or relative.name == ".env.example":
            problems += [f"{relative}: {ip}" for ip in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
                         if not ip.startswith(ALLOWED_IPS)]
    check(not problems, "No credentials, machine paths or stray IP addresses" + (f": {problems[:3]}" if problems else ""))
    check(not (ROOT / ".env").exists(), "No .env file is committed (.env.example only)")


def main() -> int:
    logging.getLogger("services.comparison_service").setLevel(logging.ERROR)
    print("FINAL SUBMISSION VALIDATION")
    print("===========================\n")
    before = data_hashes()
    attempts: list[str] = []
    real_connect, real_create = socket.socket.connect, socket.create_connection

    def blocked(*args, **kwargs):
        attempts.append("network call")
        raise AssertionError("network access attempted")

    try:
        scenarios = get_all_scenarios()
        validate_agents_and_tools(scenarios)
        socket.socket.connect = blocked  # type: ignore[method-assign]
        socket.create_connection = blocked  # type: ignore[assignment]
        try:
            validate_comparison(scenarios)
            validate_learning()
        finally:
            socket.socket.connect = real_connect  # type: ignore[method-assign]
            socket.create_connection = real_create
        check(not attempts, "No network connection attempted (no live model needed)")
        validate_llm_rules()
        validate_documents()
        validate_truthfulness()
        validate_cleanliness()
        check(data_hashes() == before and normalised_prefixes() == PINNED_DATA, "Source data unchanged")
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Final submission validation FAILED.")
        return 1

    print("\nFinal submission structure validated.")
    print("\nLive Ollama integration results must still come from an actual live run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
