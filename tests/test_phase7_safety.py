"""Phase 7 safety and boundary tests: import safety, no hidden network/side effects, data untouched, scope limits."""

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from agents.llm_agent import LLMAgent
from config.llm_config import DEFAULT_BASE_URL, load_ollama_config
from services.llm_prompts import SYSTEM_PROMPT, build_scenario_message
from services.scenario_service import get_all_scenarios
from tests.fakes.fake_tool_calling_model import happy_path_model
from tools.langchain_security_tools import TOOL_NAMES, SecurityToolbox

ROOT = Path(__file__).resolve().parent.parent
PHASE7_MODULES = [
    "config/llm_config.py", "models/llm_models.py", "services/llm_errors.py", "services/evidence_collector.py",
    "services/llm_guardrails.py", "services/llm_prompts.py", "services/ollama_chat_adapter.py",
    "services/llm_orchestrator.py", "tools/langchain_security_tools.py", "agents/llm_agent.py",
]
CLASSICAL_AGENT_FILES = ["simple_reflex.py", "model_based_reflex.py", "goal_based.py", "utility_based.py", "learning_agent.py"]
DATA_FILES = sorted((ROOT / "data").glob("*.json"))
LIVE_ONLY = {"ollama", "langchain_ollama", "httpx", "requests"}


def imported_roots(path: Path, *, top_level_only: bool) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    roots: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_importing_the_package_does_not_load_langchain_ollama_or_httpx() -> None:
    code = (
        "import sys, agents, config.llm_config, services.llm_orchestrator, services.ollama_chat_adapter\n"
        "bad = [m for m in ('langchain', 'langchain_core', 'langchain_ollama', 'ollama', 'httpx', 'requests') "
        "if m in sys.modules]\nprint('LOADED', bad)\nassert not bad"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_phase_7_modules_opens_no_network_connection() -> None:
    code = (
        "import socket\n"
        "def refuse(*a, **k): raise SystemExit('network attempted at import time')\n"
        "socket.socket.connect = refuse; socket.create_connection = refuse\n"
        "import agents, config.llm_config, services.llm_orchestrator, services.ollama_chat_adapter, "
        "tools.langchain_security_tools\nfrom agents import LLMAgent\nLLMAgent()\nprint('ok')"
    )
    result = run_python(code)
    assert result.returncode == 0 and "ok" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("relative", PHASE7_MODULES)
def test_phase_7_modules_import_live_libraries_only_lazily(relative: str) -> None:
    assert not imported_roots(ROOT / relative, top_level_only=True) & (LIVE_ONLY | {"langchain_core", "langchain"})


def test_only_the_adapter_may_import_the_live_model_clients() -> None:
    offenders = [p for p in PHASE7_MODULES if imported_roots(ROOT / p, top_level_only=False) & LIVE_ONLY
                 and p != "services/ollama_chat_adapter.py"]
    assert offenders == []


@pytest.mark.parametrize("name", CLASSICAL_AGENT_FILES)
def test_classical_agents_do_not_touch_the_llm_layer(name: str) -> None:
    source = (ROOT / "agents" / name).read_text(encoding="utf-8")
    assert not imported_roots(ROOT / "agents" / name, top_level_only=False) & (LIVE_ONLY | {"langchain_core", "langchain"})
    for fragment in ("llm_orchestrator", "llm_agent", "ollama_chat_adapter", "llm_config", "langchain_security_tools"):
        assert fragment not in source


def test_phase_6_tool_layer_does_not_import_the_llm_layer() -> None:
    for relative in ("services/security_tool_service.py", "models/tool_models.py"):
        roots = imported_roots(ROOT / relative, top_level_only=False)
        assert not roots & (LIVE_ONLY | {"langchain_core", "langchain"}), relative
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "llm_orchestrator" not in source and "llm_agent" not in source and "llm_guardrails" not in source


@pytest.mark.parametrize("relative", PHASE7_MODULES)
def test_phase_7_modules_contain_no_process_execution_or_dynamic_code(relative: str) -> None:
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned_names = {"eval", "exec", "print", "__import__"}
    banned_attributes = {"system", "popen", "Popen", "rmtree", "remove", "unlink", "check_output"}
    called_names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    called_attributes = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not called_names & banned_names and not called_attributes & banned_attributes
    banned_imports = {"subprocess", "shutil", "socket", "ctypes"} | ({"os"} if relative != "config/llm_config.py" else set())
    assert not imported_roots(path, top_level_only=False) & banned_imports


def test_the_llm_layer_adds_no_ui_or_orchestration_frameworks() -> None:
    # Streamlit now exists, but only in app.py and ui/: the agent, service, tool, model and config layers stay UI-free.
    forbidden = {"streamlit", "langgraph", "fastapi", "flask", "chromadb", "faiss", "pandas", "numpy"}
    for path in [*ROOT.glob("agents/*.py"), *ROOT.glob("services/*.py"), *ROOT.glob("tools/*.py"),
                 *ROOT.glob("models/*.py"), *ROOT.glob("config/*.py")]:
        assert not imported_roots(path, top_level_only=False) & forbidden, path
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert not any(name in requirements for name in ("langgraph", "openai", "anthropic", "fastapi"))


def test_default_configuration_points_only_at_localhost() -> None:
    assert load_ollama_config({}).base_url == DEFAULT_BASE_URL and "127.0.0.1" in DEFAULT_BASE_URL


def test_exactly_four_read_only_tools_are_exposed_and_none_can_act() -> None:
    names = SecurityToolbox().tool_names
    assert names == TOOL_NAMES and len(names) == 4
    for verb in ("block", "disable", "quarantine", "terminate", "delete", "isolate", "firewall", "contain", "kill"):
        assert not any(verb in name for name in names)


def test_prompts_forbid_actions_and_do_not_leak_answers() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "never block" in lowered and "human" in lowered and "at least two different tools" in lowered
    for scenario in get_all_scenarios():
        message = build_scenario_message(scenario)
        assert "expected_characteristics" not in message
        assert all(note not in message for note in scenario.expected_characteristics)


def _digest(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_running_the_llm_agent_never_modifies_any_data_file() -> None:
    before = _digest(DATA_FILES)
    for scenario in get_all_scenarios():
        LLMAgent(model=happy_path_model(scenario)).decide(scenario)
    assert _digest(DATA_FILES) == before


def test_no_secrets_or_real_hosts_are_present_in_phase_7_files() -> None:
    markers = ("api_key", "apikey", "bearer ", "begin private", "sk-")
    for relative in PHASE7_MODULES:
        text = (ROOT / relative).read_text(encoding="utf-8").lower()
        assert not [m for m in markers if m in text], relative
