"""UI safety tests: no remediation capability, committed data protected, mock/live status truthful, nothing sensitive shown."""

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

import ui.formatters as fmt
from services.data_loader import DATA_DIR
from tests.app_helpers import (
    ROOT,
    all_text,
    comparison_rows,
    expander,
    run_all_agents,
    run_app,
    select_scenario,
)
from tests.fakes.service import mock_service

# Development/offline settings: MOCK runtime, explicitly enabled. (Submission-mode UI tests are in test_submission_config.py.)
pytestmark = pytest.mark.usefixtures("dev_mock_environment")

UI_FILES = [ROOT / "app.py", *sorted((ROOT / "ui").glob("*.py"))]
SERVICE_FILES = [ROOT / "services" / "comparison_service.py", ROOT / "models" / "comparison_models.py"]
DATA_FILES = sorted(DATA_DIR.glob("*.json"))

# Identifiers that would indicate an action capability. (`st.container` is deliberately not matched.)
REMEDIATION_NAMES = re.compile(
    r"(block_|_block\b|disable_|_disable\b|quarantine|terminate|kill_|firewall|isolate_|lock_?out|"
    r"containment|contain_|exploit|attack|retaliat|hack_?back|remediat|delete_user|shutdown)", re.IGNORECASE)
FILE_MUTATION_CALLS = {"open", "write_text", "write_bytes", "unlink", "rmtree", "remove", "rename",
                       "mkdir", "touch", "to_csv", "to_json", "dump", "chmod"}
# With none of these importable, the UI layer has no way to read or write files, reach a network or run processes.
NETWORK_AND_PROCESS_MODULES = {"subprocess", "socket", "requests", "httpx", "urllib", "ollama", "langchain",
                               "langchain_core", "langchain_ollama", "os", "shutil", "ctypes", "pathlib",
                               "tempfile", "json", "pickle", "sqlite3"}
SENSITIVE_TEXT = re.compile(r"(password|passwd|secret|api[_-]?key|bearer |token=|traceback|[A-Za-z]:\\\\|/Users/|/home/)",
                            re.IGNORECASE)


def tree_of(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def identifiers(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree_of(path)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.arg)):
            names.add(getattr(node, "name", None) or getattr(node, "arg", ""))
    return names


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree_of(path)):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def data_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_FILES}


# ------------------------------------------------------------------------------ static boundaries


@pytest.mark.parametrize("path", [*UI_FILES, *SERVICE_FILES], ids=lambda p: p.name)
def test_no_ui_or_comparison_code_can_block_disable_quarantine_firewall_contain_or_attack(path: Path) -> None:
    offenders = sorted(n for n in identifiers(path) if REMEDIATION_NAMES.search(n))
    assert offenders == [], offenders


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_ui_code_performs_no_file_writes_network_or_process_calls(path: Path) -> None:
    called = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
              for n in ast.walk(tree_of(path)) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert not called & FILE_MUTATION_CALLS, called & FILE_MUTATION_CALLS
    assert not imported_roots(path) & NETWORK_AND_PROCESS_MODULES
    assert not called & {"eval", "exec", "system", "popen", "Popen"}


def test_ui_code_never_touches_the_committed_experience_store_or_scenario_files() -> None:
    forbidden = {"DEFAULT_EXPERIENCE_PATH", "ExperienceRepository", "record_outcome", "DATA_DIR", "load_json_object",
                 "get_experience", "EXPERIENCE_PATH"}
    for path in UI_FILES:
        assert not identifiers(path) & forbidden, path.name


def test_ui_cannot_reach_the_scenario_or_policy_data_for_writing() -> None:
    for path in UI_FILES:
        text = path.read_text(encoding="utf-8")
        assert "scenarios.json" not in text and "policies.json" not in text and "alert_details.json" not in text


def test_stateful_agents_and_traces_are_never_cached() -> None:
    for path in UI_FILES:
        text = path.read_text(encoding="utf-8")
        assert "cache_resource" not in text and "cache_data" not in text, path.name


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_streamlit_is_confined_to_the_ui_layer(path: Path) -> None:
    assert path.parent in (ROOT, ROOT / "ui")
    for service_file in SERVICE_FILES:
        assert "streamlit" not in imported_roots(service_file)
    assert "streamlit" not in imported_roots(ROOT / "ui" / "formatters.py")


def test_ui_does_not_import_agent_decision_code() -> None:
    # The UI renders results; it must not instantiate or call any agent directly.
    for path in UI_FILES:
        for node in ast.walk(tree_of(path)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("agents"), path.name
    for path in UI_FILES:
        assert not any(name.endswith("Agent") and name != "AgentComparisonService" and name not in {"LLM Agent"}
                       for name in identifiers(path)
                       if name in {"SimpleReflexAgent", "ModelBasedReflexAgent", "GoalBasedAgent",
                                   "UtilityBasedAgent", "LearningAgent", "LLMAgent"}), path.name


def test_source_dependency_declared_without_visualisation_extras() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "streamlit" in requirements
    assert not any(name in requirements for name in ("pandas", "plotly", "altair", "numpy"))
    for kept in ("pydantic", "pytest", "langchain-core", "langchain-ollama"):
        assert kept in requirements


# ------------------------------------------------------------------------------ runtime behaviour


def test_a_full_ui_session_never_modifies_any_data_file() -> None:
    before = data_hashes()
    at = run_app()
    for scenario_id in "ABC":
        at = run_all_agents(select_scenario(at, scenario_id))
        expander(at, "How Model-Based").button(key="mb_run_twice").click()
        expander(at, "How Feedback").button(key="learn_failure").click()
        at = at.run()
        expander(at, "How Feedback").button(key="learn_success").click()
        at = at.run()
    at = at.sidebar.button(key="reset_demos").click().run()
    assert not at.exception
    assert data_hashes() == before


def test_mock_mode_never_claims_to_be_ollama() -> None:
    at = run_all_agents(run_app())
    text = all_text(at)
    assert "LIVE OLLAMA" not in text and "selected by the configured Ollama model" not in text
    assert [r["Runtime"] for r in comparison_rows(at)][-1] == "MOCK"
    assert "Runtime: Mock / Offline Validation" in text and "not enabled" in text  # live is not offered
    assert len(at.sidebar.radio) == 0  # a single available runtime needs no selector


def test_the_mock_warning_is_visible_before_and_after_running() -> None:
    at = run_app()
    assert fmt.MOCK_BANNER in [w.value for w in at.warning]
    at = run_all_agents(at)
    assert fmt.MOCK_BANNER in [w.value for w in at.warning]


def _enable_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_OLLAMA_INTEGRATION", "true")
    monkeypatch.setenv("OLLAMA_MODEL", "example-model")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


def test_live_mode_is_offered_only_when_explicitly_enabled_and_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_live(monkeypatch)
    at = run_app()
    assert at.sidebar.radio(key="llm_mode_option").options == [fmt.MOCK_MODE_OPTION, fmt.LIVE_MODE_OPTION]
    monkeypatch.delenv("OLLAMA_MODEL")
    assert len(run_app().sidebar.radio) == 0  # live not configured: only the mock remains, so no selector


def test_live_failure_shows_an_explicit_error_and_never_falls_back_to_the_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.llm_agent as llm_agent_module
    import devtools.mock_llm_runtime as fakes  # the single implementation the mock factory calls
    from models.llm_models import RuntimeMode
    from services.llm_errors import LLMConnectionError

    class Unreachable:
        def __init__(self, config: Any) -> None:
            self.runtime_mode, self.model_name = RuntimeMode.OLLAMA, config.model

        def bind_tools(self, _tools: Any) -> Any:
            raise LLMConnectionError("Could not reach the local Ollama server.")

    mock_calls: list[Any] = []

    _enable_live(monkeypatch)
    monkeypatch.setattr(llm_agent_module, "OllamaChatModelAdapter", Unreachable)
    monkeypatch.setattr(fakes, "happy_path_model", lambda *a, **k: mock_calls.append(a))
    at = run_app()
    at.sidebar.radio(key="llm_mode_option").set_value(fmt.LIVE_MODE_OPTION)
    at = run_all_agents(at.run())
    assert not at.exception
    assert mock_calls == []  # the mock model was never built in live mode
    rows = comparison_rows(at)
    assert rows[-1]["Decision"] == "ERROR" and rows[-1]["Runtime"] == "UNAVAILABLE"
    assert [r["Decision"] for r in rows[:5]] == ["INVESTIGATE"] * 5
    text = all_text(at)
    assert "Live Ollama is unavailable" in text
    assert "LIVE OLLAMA" not in text and fmt.MOCK_BANNER not in [w.value for w in at.warning]
    assert "Tools were selected by the configured Ollama model" not in text
    assert "Traceback" not in text


def test_a_mock_result_is_not_shown_after_switching_to_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_live(monkeypatch)
    at = run_all_agents(run_app())  # a MOCK comparison exists
    at.sidebar.radio(key="llm_mode_option").set_value(fmt.LIVE_MODE_OPTION)
    at = at.run()
    assert not [e for e in at.expander if e.label.startswith("LLM Agent - ")]
    assert any("Press **Run All Agents**" in i.value for i in at.info)


def test_no_credentials_or_raw_exceptions_are_rendered() -> None:
    at = run_all_agents(run_app())
    for scenario_id in "BC":
        at = run_all_agents(select_scenario(at, scenario_id))
    assert not at.exception
    leaked = SENSITIVE_TEXT.findall(all_text(at))
    assert leaked == []


def test_session_state_holds_only_results_and_demo_objects_never_secrets() -> None:
    at = run_all_agents(run_app())
    expander(at, "How Model-Based").button(key="mb_run_twice").click()
    at = at.run()
    keys = set(at.session_state)
    results_and_demos = {"comparison", "mb_demo", "learning_demo"}
    widget_keys = {"scenario_id", "llm_mode_option", "run_all", "retry_llm", "reset_demos", "mb_run_twice",
                   "learn_failure", "learn_success", "learn_reset"}
    assert keys <= results_and_demos | widget_keys, keys
    assert not any(re.search(r"(secret|token|password|credential|api)", k, re.I) for k in keys)


def test_the_ui_offers_no_buttons_that_act_on_accounts_hosts_or_networks() -> None:
    at = run_app()
    labels = " | ".join(b.label for b in at.button)
    assert not re.search(r"(block|disable|quarantine|terminate|firewall|contain|isolate|attack|delete)", labels, re.I)
    assert {b.label for b in at.button} >= {"Run All Agents", "Reset Demonstrations"}


def test_comparison_service_still_runs_when_the_ui_is_not_involved() -> None:
    from models.schemas import AgentType
    from services.scenario_service import get_scenario

    result = mock_service().compare(get_scenario("A"))
    assert result.entry(AgentType.LLM).llm_runtime_mode is not None and result.all_succeeded
