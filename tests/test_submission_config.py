"""Submission-hardening tests: runtime configuration (SC-01..SC-20), thinking protection, the dev-mock gate,
live single-flight/retry, and the submission-mode UI. Nothing here contacts Ollama: LIVE paths use a simulated adapter."""

import ast
import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import services.comparison_service as comparison_module
import ui.formatters as fmt
from agents import LLMAgent
from config.llm_config import DEFAULT_BASE_URL, OllamaConfig, load_ollama_config
from config.runtime_config import LLMRuntimeMode, RuntimeConfig, load_runtime_config
from models.llm_models import RuntimeMode
from models.schemas import AgentType
from services.comparison_service import AgentStatus
from services.data_loader import DATA_DIR
from services.llm_errors import LLMConfigurationError, LLMConnectionError
from services.llm_orchestrator import THINKING_FIELD_NAMES, LLMOrchestrator, sanitize_reply
from services.llm_runtime import build_comparison_service, inspect_runtime, live_unavailable_message
from services.scenario_service import get_scenario
from tests.app_helpers import ROOT, all_text, comparison_rows, run_all_agents, run_app
from tests.fakes.fake_tool_calling_model import ScriptedChatModel, answer_step, standard_calls, tool_calls_message
from tests.fakes.service import THINKING_MARKER, SimulatedLiveAdapter, mock_service

pytestmark = pytest.mark.usefixtures("clean_runtime_environment")

SUBMISSION_ENV = {
    "LLM_RUNTIME_MODE": "LIVE", "ENABLE_MOCK_LLM": "false", "RUN_OLLAMA_INTEGRATION": "true",
    "SUBMISSION_MODE": "true", "OLLAMA_MODEL": "qwen3:4b",
}
DEV_ENV = {"LLM_RUNTIME_MODE": "MOCK", "ENABLE_MOCK_LLM": "true"}
PRODUCTION_DIRS = ("agents", "config", "models", "services", "tools", "ui")
DATA_FILES = sorted(DATA_DIR.glob("*.json"))


def production_files() -> list[Path]:
    files = [ROOT / "app.py"]
    for directory in PRODUCTION_DIRS:
        files += sorted((ROOT / directory).glob("*.py"))
    return files


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def data_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_FILES}


class Unreachable:
    """A live adapter whose server cannot be reached."""

    created = 0

    def __init__(self, config: Any) -> None:
        type(self).created += 1
        self.runtime_mode, self.model_name = RuntimeMode.OLLAMA, config.model

    def bind_tools(self, _tools: Any) -> Any:
        raise LLMConnectionError("Could not reach the local Ollama server.")


@pytest.fixture
def submission_environment(clean_runtime_environment: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name, value in SUBMISSION_ENV.items():
        clean_runtime_environment.setenv(name, value)
    return clean_runtime_environment


@pytest.fixture
def mock_must_not_run(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Records any attempt to build the development mock."""
    import devtools.mock_llm_runtime as mock_module

    attempts: list[Any] = []
    monkeypatch.setattr(mock_module, "mock_llm_agent_factory", lambda *a, **k: attempts.append(a))
    monkeypatch.setattr(mock_module, "happy_path_model", lambda *a, **k: attempts.append(a))
    return attempts


# ------------------------------------------------------------------------------ runtime configuration


def test_sc_01_submission_mode_requires_live() -> None:
    config = load_runtime_config(SUBMISSION_ENV)
    assert config.mode is LLMRuntimeMode.LIVE and config.submission_mode and config.result_mode is RuntimeMode.OLLAMA
    unset_mode = {k: v for k, v in SUBMISSION_ENV.items() if k != "LLM_RUNTIME_MODE"}
    assert load_runtime_config(unset_mode).mode is LLMRuntimeMode.LIVE  # LIVE is the default
    with pytest.raises(LLMConfigurationError, match="requires LLM_RUNTIME_MODE=LIVE"):
        load_runtime_config({**SUBMISSION_ENV, "LLM_RUNTIME_MODE": "DISABLED"})


@pytest.mark.parametrize("enable_mock", ["true", "false"])
def test_sc_02_submission_mode_rejects_mock(enable_mock: str) -> None:
    env = {**SUBMISSION_ENV, "LLM_RUNTIME_MODE": "MOCK", "ENABLE_MOCK_LLM": enable_mock}
    with pytest.raises(LLMConfigurationError, match="MOCK"):
        load_runtime_config(env)
    with pytest.raises(ValidationError):
        RuntimeConfig(mode=LLMRuntimeMode.MOCK, enable_mock=True, submission_mode=True)
    status = inspect_runtime(env)
    assert status.available == () and status.default is None and status.config_error
    with pytest.raises(LLMConfigurationError):
        build_comparison_service(RuntimeMode.MOCK, env)  # the mock cannot be built in submission mode


def test_sc_03_live_requires_integration_enabled() -> None:
    env = {**SUBMISSION_ENV, "RUN_OLLAMA_INTEGRATION": "false", "SUBMISSION_MODE": "false"}
    with pytest.raises(LLMConfigurationError, match="RUN_OLLAMA_INTEGRATION=true"):
        load_runtime_config(env)
    with pytest.raises(LLMConfigurationError):
        load_runtime_config({"LLM_RUNTIME_MODE": "LIVE"})  # nothing is enabled implicitly


def test_sc_04_live_requires_a_model() -> None:
    env = {k: v for k, v in SUBMISSION_ENV.items() if k != "OLLAMA_MODEL"}
    with pytest.raises(LLMConfigurationError, match="model"):
        load_runtime_config(env)
    with pytest.raises(LLMConfigurationError):
        load_runtime_config({**SUBMISSION_ENV, "OLLAMA_MODEL": "   "})


def test_sc_05_mock_requires_explicit_enablement_and_unknown_modes_fail() -> None:
    with pytest.raises(LLMConfigurationError, match="ENABLE_MOCK_LLM=true"):
        load_runtime_config({"LLM_RUNTIME_MODE": "MOCK"})
    assert load_runtime_config(DEV_ENV).mock_allowed
    with pytest.raises(LLMConfigurationError, match="LLM_RUNTIME_MODE must be one of"):
        load_runtime_config({"LLM_RUNTIME_MODE": "AUTO"})
    with pytest.raises(LLMConfigurationError, match="true or false"):
        load_runtime_config({**DEV_ENV, "ENABLE_MOCK_LLM": "maybe"})
    assert load_runtime_config({"LLM_RUNTIME_MODE": "disabled"}).mode is LLMRuntimeMode.DISABLED  # case-insensitive
    assert not RuntimeConfig(mode=LLMRuntimeMode.DISABLED).mock_allowed


def test_the_mock_is_built_only_behind_the_gate_and_never_loaded_otherwise() -> None:
    code = (
        "import sys\n"
        "from models.llm_models import RuntimeMode\n"
        "from services.llm_runtime import inspect_runtime, build_comparison_service\n"
        "env = {'LLM_RUNTIME_MODE': 'MOCK', 'ENABLE_MOCK_LLM': 'true'}\n"
        "inspect_runtime(env)\n"
        "assert 'devtools' not in sys.modules, 'inspecting must not load the mock'\n"
        "live_env = {'LLM_RUNTIME_MODE': 'LIVE', 'RUN_OLLAMA_INTEGRATION': 'true', 'OLLAMA_MODEL': 'qwen3:4b'}\n"
        "build_comparison_service(RuntimeMode.OLLAMA, live_env)\n"
        "assert 'devtools' not in sys.modules, 'the live path must not load the mock'\n"
        "build_comparison_service(RuntimeMode.MOCK, env)\n"
        "assert 'devtools.mock_llm_runtime' in sys.modules\nprint('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0 and "ok" in result.stdout, result.stdout + result.stderr


def test_inspect_runtime_reports_what_may_be_offered() -> None:
    dev = inspect_runtime(DEV_ENV)
    assert dev.available == (RuntimeMode.MOCK,) and dev.default is RuntimeMode.MOCK and not dev.submission_mode
    both = inspect_runtime({**DEV_ENV, "RUN_OLLAMA_INTEGRATION": "true", "OLLAMA_MODEL": "qwen3:4b"})
    assert both.available == (RuntimeMode.MOCK, RuntimeMode.OLLAMA) and both.default is RuntimeMode.MOCK
    submission = inspect_runtime(SUBMISSION_ENV)
    assert submission.available == (RuntimeMode.OLLAMA,) and submission.default is RuntimeMode.OLLAMA
    assert submission.model_name == "qwen3:4b" and submission.submission_mode and submission.config_error is None
    assert inspect_runtime({"LLM_RUNTIME_MODE": "DISABLED"}).available == ()
    nothing_set = inspect_runtime({})  # default LIVE, but integration is not enabled: nothing is offered
    assert nothing_set.available == () and "RUN_OLLAMA_INTEGRATION" in (nothing_set.config_error or "")


def test_unknown_runtime_never_infers_live_from_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def refuse(*_a: Any, **_k: Any) -> None:
        raise AssertionError("the runtime inspection opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert inspect_runtime({}).available == () and inspect_runtime(SUBMISSION_ENV).available


# ------------------------------------------------------------------------------ failure behaviour


def test_sc_06_live_failure_never_returns_a_mock_result(
    monkeypatch: pytest.MonkeyPatch, mock_must_not_run: list[Any]
) -> None:
    import agents.llm_agent as llm_agent_module

    monkeypatch.setattr(llm_agent_module, "OllamaChatModelAdapter", Unreachable)
    service = build_comparison_service(RuntimeMode.OLLAMA, SUBMISSION_ENV)
    result = service.compare(get_scenario("A"))
    llm = result.entry(AgentType.LLM)
    assert llm.status is AgentStatus.ERROR and llm.decision_record is None
    assert mock_must_not_run == []
    assert all(e.llm_runtime_mode is not RuntimeMode.MOCK for e in result.results)
    assert len(result.successful) == 5


def test_sc_06_a_failed_live_run_never_loads_the_mock_module() -> None:
    code = (
        "import sys\n"
        "import agents.llm_agent as m\n"
        "from models.llm_models import RuntimeMode\n"
        "from services.llm_errors import LLMConnectionError\n"
        "from services.llm_runtime import build_comparison_service\n"
        "from services.scenario_service import get_scenario\n"
        "class Down:\n"
        "    def __init__(self, c): self.runtime_mode, self.model_name = RuntimeMode.OLLAMA, c.model\n"
        "    def bind_tools(self, t): raise LLMConnectionError('down')\n"
        "m.OllamaChatModelAdapter = Down\n"
        "env = {'LLM_RUNTIME_MODE': 'LIVE', 'RUN_OLLAMA_INTEGRATION': 'true', 'OLLAMA_MODEL': 'qwen3:4b'}\n"
        "r = build_comparison_service(RuntimeMode.OLLAMA, env).compare(get_scenario('A'))\n"
        "assert r.entry(m.LLMAgent and __import__('models.schemas', fromlist=['x']).AgentType.LLM).status.value == 'ERROR'\n"
        "assert 'devtools' not in sys.modules and 'tests' not in sys.modules\nprint('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0 and "ok" in result.stdout, result.stdout + result.stderr


def test_the_comparison_service_itself_has_no_mock_model() -> None:
    result = comparison_module.AgentComparisonService(llm_mode=RuntimeMode.MOCK).compare(get_scenario("A"))
    llm = result.entry(AgentType.LLM)  # MOCK without an injected model is an error, never a silent result
    assert llm.status is AgentStatus.ERROR and "no model" in llm.error_message
    assert len(result.successful) == 5
    disabled = comparison_module.AgentComparisonService(llm_mode=None, llm_unavailable_reason="off").compare(get_scenario("A"))
    assert disabled.llm_runtime_mode is None and disabled.entry(AgentType.LLM).status is AgentStatus.ERROR


def test_only_one_live_request_runs_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.llm_agent as llm_agent_module

    Unreachable.created = 0
    monkeypatch.setattr(llm_agent_module, "OllamaChatModelAdapter", Unreachable)
    config = OllamaConfig(model="qwen3:4b", integration_enabled=True)
    service = comparison_module.AgentComparisonService(llm_mode=RuntimeMode.OLLAMA, ollama_config=config)
    assert comparison_module._LIVE_LLM_LOCK.acquire(blocking=False)
    try:
        busy = service.compare(get_scenario("A")).entry(AgentType.LLM)
    finally:
        comparison_module._LIVE_LLM_LOCK.release()
    assert busy.status is AgentStatus.ERROR and "already running" in busy.error_message
    assert Unreachable.created == 0  # the second request never reached the model
    service.compare(get_scenario("A"))  # the lock was released: the next request proceeds
    assert Unreachable.created == 1


def test_rerun_agent_retries_only_the_llm_agent_for_the_same_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.llm_agent as llm_agent_module

    calls: list[AgentType] = []
    from agents import GoalBasedAgent, LearningAgent, ModelBasedReflexAgent, SimpleReflexAgent, UtilityBasedAgent
    from services.experience_repository import ExperienceRepository

    def counted(agent_type: AgentType, build: Any) -> Any:
        return lambda scenario: (calls.append(agent_type), build())[1]

    factories = {
        AgentType.SIMPLE_REFLEX: counted(AgentType.SIMPLE_REFLEX, SimpleReflexAgent),
        AgentType.MODEL_BASED_REFLEX: counted(AgentType.MODEL_BASED_REFLEX, ModelBasedReflexAgent),
        AgentType.GOAL_BASED: counted(AgentType.GOAL_BASED, GoalBasedAgent),
        AgentType.UTILITY_BASED: counted(AgentType.UTILITY_BASED, UtilityBasedAgent),
        AgentType.LEARNING: counted(AgentType.LEARNING, lambda: LearningAgent(ExperienceRepository())),
    }
    config = OllamaConfig(model="qwen3:4b", integration_enabled=True)
    service = comparison_module.AgentComparisonService(
        llm_mode=RuntimeMode.OLLAMA, ollama_config=config, agent_factories=factories)
    scenario = get_scenario("A")
    monkeypatch.setattr(llm_agent_module, "OllamaChatModelAdapter", Unreachable)
    failed = service.compare(scenario)
    assert failed.entry(AgentType.LLM).status is AgentStatus.ERROR and len(calls) == 5
    monkeypatch.setattr(llm_agent_module, "OllamaChatModelAdapter", SimulatedLiveAdapter)
    retried = service.rerun_agent(failed, AgentType.LLM)
    assert len(calls) == 5  # none of the five classical agents ran again
    assert retried.entry(AgentType.LLM).status is AgentStatus.SUCCESS and retried.all_succeeded
    assert retried.entry(AgentType.LLM).llm_runtime_mode is RuntimeMode.OLLAMA
    for agent_type in AGENT_TYPES_WITHOUT_LLM:
        assert retried.entry(agent_type) is failed.entry(agent_type)  # kept exactly as it was
    assert retried.scenario_fingerprint == failed.scenario_fingerprint
    with pytest.raises(ValueError):
        service.rerun_agent(failed.model_copy(update={"llm_runtime_mode": RuntimeMode.MOCK}), AgentType.LLM)


AGENT_TYPES_WITHOUT_LLM = [t for t in AgentType if t not in (AgentType.LLM,)]


# ------------------------------------------------------------------------------ thinking protection


def _reply_with_thinking() -> Any:
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=f"<think>{THINKING_MARKER}</think>visible text", tool_calls=[
            {"name": "get_alert_details", "args": {"alert_id": "ALERT-A-001"}, "id": "c1", "type": "tool_call"}],
        additional_kwargs={"thinking": THINKING_MARKER, "keep": "ok", "nested": {"reasoning_content": THINKING_MARKER}},
        response_metadata={"thinking": THINKING_MARKER, "chain_of_thought": THINKING_MARKER},
    )


def test_sc_13_thinking_fields_are_discarded_by_the_sanitiser() -> None:
    clean = sanitize_reply(_reply_with_thinking())
    dump = json.dumps(clean.model_dump(mode="json"), default=str)
    assert THINKING_MARKER not in dump and clean.content == "visible text"
    assert clean.additional_kwargs == {"keep": "ok", "nested": {}}
    assert clean.response_metadata == {} and clean.tool_calls[0]["name"] == "get_alert_details"
    assert {"thinking", "reasoning_content", "chain_of_thought"} <= THINKING_FIELD_NAMES
    assert THINKING_MARKER not in str(sanitize_reply(type("R", (), {"content": [{"type": "thinking", "thinking": THINKING_MARKER},
                                                                        {"type": "text", "text": "hi"}]})()).content)


def test_sc_13_thinking_never_enters_the_decision_trace_conversation_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    scenario = get_scenario("A")
    calls = standard_calls(scenario)

    def with_thinking(step: Any) -> Any:
        from langchain_core.messages import AIMessage

        def wrapped(messages: Any) -> Any:
            reply = step(messages) if callable(step) else step
            return AIMessage(
                content=f"<think>{THINKING_MARKER}</think>{reply.content}", tool_calls=reply.tool_calls,
                additional_kwargs={"reasoning_content": THINKING_MARKER, "thinking": THINKING_MARKER},
                response_metadata={"thinking": THINKING_MARKER})
        return wrapped

    model = ScriptedChatModel([with_thinking(tool_calls_message(*calls)), with_thinking(answer_step(scenario))])
    with caplog.at_level(logging.DEBUG):
        decision = LLMAgent(model=model, config=OllamaConfig()).decide(scenario)
    assert decision.agent_type is AgentType.LLM
    everything = json.dumps(decision.model_dump(mode="json"), default=str)
    assert THINKING_MARKER not in everything  # AgentDecision, metadata and execution trace
    for turn in model.invocations:  # what was sent back to the model on later turns
        assert THINKING_MARKER not in json.dumps([m.model_dump(mode="json") for m in turn], default=str)
    assert THINKING_MARKER not in caplog.text
    assert THINKING_MARKER not in repr(model.invocations[-1])


def test_sc_13_a_thinking_only_prefix_does_not_break_json_parsing() -> None:
    scenario = get_scenario("B")
    from langchain_core.messages import AIMessage

    final = answer_step(scenario)

    def think_then_json(messages: Any) -> Any:
        return AIMessage(content=f"<think>{THINKING_MARKER} unfinished")  # never closed, nothing after it

    model = ScriptedChatModel([tool_calls_message(*standard_calls(scenario)), think_then_json, final])
    result = LLMOrchestrator(model, OllamaConfig()).run(scenario)
    assert result.trace.final_output_valid  # the empty reply was repaired once; thinking text never leaked
    assert THINKING_MARKER not in json.dumps(result.trace.model_dump(mode="json"), default=str)


# ------------------------------------------------------------------------------ submission-mode UI


def run_submission_app(monkeypatch: pytest.MonkeyPatch, adapter: type, **flags: Any) -> Any:
    import agents.llm_agent as llm_agent_module

    monkeypatch.setattr(llm_agent_module, "OllamaChatModelAdapter", adapter)
    for key, value in flags.items():
        setattr(adapter, key, value)
    return run_app()


def test_sc_07_submission_app_hides_mock_controls(
    submission_environment: pytest.MonkeyPatch, mock_must_not_run: list[Any]
) -> None:
    at = run_submission_app(submission_environment, SimulatedLiveAdapter)
    assert not at.exception and len(at.sidebar.radio) == 0  # no runtime selector
    text = all_text(at)
    assert "Runtime: OLLAMA" in text and "Model: qwen3:4b" in text
    for forbidden in ("mock", "deterministic", "pending", "Real Ollama", "offline MOCK"):
        assert forbidden.lower() not in text.lower(), forbidden
    assert not at.warning and fmt.MOCK_BANNER not in text
    labels = {b.label for b in at.button}
    assert {"Run All Agents", "Reset Demonstrations"} <= labels and not any("mock" in b.lower() for b in labels)
    assert mock_must_not_run == []


def test_sc_08_submission_app_contains_no_mock_result(
    submission_environment: pytest.MonkeyPatch, mock_must_not_run: list[Any]
) -> None:
    Unreachable.created = 0
    at = run_all_agents(run_submission_app(submission_environment, Unreachable))
    assert not at.exception
    rows = comparison_rows(at)
    assert [r["Decision"] for r in rows[:5]] == ["INVESTIGATE"] * 5
    assert rows[5]["Decision"] == "ERROR" and rows[5]["Runtime"] == "UNAVAILABLE"
    text = all_text(at)
    assert "mock" not in text.lower() and "MOCK" not in " ".join(r["Runtime"] for r in rows)
    assert mock_must_not_run == []


def test_sc_09_ollama_runtime_is_shown_only_after_a_real_live_result(
    submission_environment: pytest.MonkeyPatch,
) -> None:
    at = run_submission_app(submission_environment, SimulatedLiveAdapter, leak_thinking=False)
    assert "LIVE OLLAMA" not in all_text(at)  # nothing is claimed before a result exists
    at = run_all_agents(at)
    rows = comparison_rows(at)
    assert rows[5]["Runtime"] == "OLLAMA" and rows[5]["Decision"] == "INVESTIGATE"
    assert any(fmt.LIVE_SUCCESS_TEXT == s.value for s in at.success)
    assert at.session_state["comparison"].entry(AgentType.LLM).llm_runtime_mode is RuntimeMode.OLLAMA


def test_sc_10_live_error_retains_the_five_classical_results_and_shows_the_required_message(
    submission_environment: pytest.MonkeyPatch,
) -> None:
    at = run_all_agents(run_submission_app(submission_environment, Unreachable))
    expected = ("Live Ollama is unavailable. Verify that Ollama is running locally, that qwen3:4b is installed, "
                "and that the environment configuration is correct.")
    assert expected == live_unavailable_message("qwen3:4b")
    assert expected in [e.value for e in at.error]
    assert len([e for e in at.expander if " - HIGH / INVESTIGATE" in e.label]) == 5
    text = all_text(at)
    for leaked in ("Traceback", "C:\\", "/home/", "OLLAMA_MODEL=", "Could not reach"):
        assert leaked not in text, leaked


def test_a_failed_live_run_offers_a_retry_that_reruns_only_the_llm_agent(
    submission_environment: pytest.MonkeyPatch,
) -> None:
    import agents.llm_agent as llm_agent_module

    at = run_all_agents(run_submission_app(submission_environment, Unreachable))
    assert [b.label for b in at.sidebar.button if b.label == "Retry LLM Agent"] == ["Retry LLM Agent"]
    before = comparison_rows(at)[:5]
    submission_environment.setattr(llm_agent_module, "OllamaChatModelAdapter", SimulatedLiveAdapter)
    at = at.sidebar.button(key="retry_llm").click().run()
    assert not at.exception
    rows = comparison_rows(at)
    assert rows[:5] == before and rows[5]["Runtime"] == "OLLAMA"
    assert not [b for b in at.sidebar.button if b.label == "Retry LLM Agent"]  # nothing left to retry


def test_sc_14_no_chain_of_thought_is_displayed(submission_environment: pytest.MonkeyPatch) -> None:
    at = run_all_agents(run_submission_app(submission_environment, SimulatedLiveAdapter, leak_thinking=True))
    SimulatedLiveAdapter.leak_thinking = False
    assert not at.exception and comparison_rows(at)[5]["Runtime"] == "OLLAMA"
    assert THINKING_MARKER not in all_text(at)
    assert THINKING_MARKER not in json.dumps(at.session_state["comparison"].model_dump(mode="json"), default=str)


def test_sc_20_submission_mode_leaves_source_data_unchanged(submission_environment: pytest.MonkeyPatch) -> None:
    before = data_hashes()
    at = run_all_agents(run_submission_app(submission_environment, SimulatedLiveAdapter, leak_thinking=False))
    assert not at.exception and data_hashes() == before


def test_the_live_wait_message_says_cpu_inference_can_take_several_minutes() -> None:
    assert "several minutes" in fmt.LIVE_RUNNING_MESSAGE and "CPU" in fmt.LIVE_RUNNING_MESSAGE


def test_development_ui_keeps_the_truthful_mock_labels(
    clean_runtime_environment: pytest.MonkeyPatch,
) -> None:
    for name, value in DEV_ENV.items():
        clean_runtime_environment.setenv(name, value)
    at = run_all_agents(run_app())
    assert fmt.MOCK_BANNER in [w.value for w in at.warning]
    assert comparison_rows(at)[5]["Runtime"] == "MOCK" and "LIVE OLLAMA" not in all_text(at)
    assert "Pending live Ollama validation." in all_text(at)


def test_with_no_environment_at_all_the_page_explains_what_to_set_and_still_runs_classical_agents(
    clean_runtime_environment: pytest.MonkeyPatch,
) -> None:
    at = run_all_agents(run_app())  # nothing exported: the default LIVE runtime is not enabled
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "RUN_OLLAMA_INTEGRATION=true" in errors
    rows = comparison_rows(at)
    assert [r["Decision"] for r in rows[:5]] == ["INVESTIGATE"] * 5 and rows[5]["Decision"] == "ERROR"
    assert fmt.MOCK_BANNER not in all_text(at) and "MOCK" not in " ".join(r["Runtime"] for r in rows)  # no silent mock


def test_an_invalid_environment_shows_a_clear_error_and_still_runs_the_classical_agents(
    clean_runtime_environment: pytest.MonkeyPatch,
) -> None:
    clean_runtime_environment.setenv("SUBMISSION_MODE", "true")
    clean_runtime_environment.setenv("LLM_RUNTIME_MODE", "MOCK")
    clean_runtime_environment.setenv("ENABLE_MOCK_LLM", "true")
    at = run_all_agents(run_app())
    assert not at.exception
    assert any("requires LLM_RUNTIME_MODE=LIVE" in e.value for e in at.error)
    rows = comparison_rows(at)
    assert [r["Decision"] for r in rows[:5]] == ["INVESTIGATE"] * 5 and rows[5]["Decision"] == "ERROR"
    assert "MOCK" not in " ".join(r["Runtime"] for r in rows)


# ------------------------------------------------------------------------------ production / test separation


def test_sc_11_no_production_module_imports_test_fakes() -> None:
    for path in production_files():
        text = path.read_text(encoding="utf-8")
        assert "tests.fakes" not in text and "fake_tool_calling_model" not in text, path.name
        assert "tests" not in imported_roots(path), path.name


def test_sc_12_no_production_module_imports_tests_or_devtools_directly() -> None:
    for path in production_files():
        assert not imported_roots(path) & {"tests", "devtools"}, path.name  # devtools is loaded by name, behind the gate
    llm_runtime = (ROOT / "services" / "llm_runtime.py").read_text(encoding="utf-8")
    assert "import_module" in llm_runtime and llm_runtime.count("devtools.mock_llm_runtime") == 1
    for other in production_files():
        if other.name != "llm_runtime.py":
            assert "devtools" not in other.read_text(encoding="utf-8").replace("devtools/", ""), other.name


def test_the_dev_mock_itself_imports_nothing_from_tests() -> None:
    assert "tests" not in imported_roots(ROOT / "devtools" / "mock_llm_runtime.py")
    assert "DEVELOPMENT / TEST ONLY" in (ROOT / "devtools" / "mock_llm_runtime.py").read_text(encoding="utf-8")


def test_the_test_fake_is_a_reexport_not_a_second_implementation() -> None:
    import devtools.mock_llm_runtime as dev
    import tests.fakes.fake_tool_calling_model as fake

    assert fake.ScriptedChatModel is dev.ScriptedChatModel and fake.happy_path_model is dev.happy_path_model
    assert len((ROOT / "tests" / "fakes" / "fake_tool_calling_model.py").read_text(encoding="utf-8").splitlines()) < 12


def test_mock_service_helper_never_injects_a_mock_into_a_live_service() -> None:
    live = mock_service(llm_mode=RuntimeMode.OLLAMA)
    entry = live.compare(get_scenario("A")).entry(AgentType.LLM)
    assert entry.status is AgentStatus.ERROR and entry.llm_runtime_mode is RuntimeMode.OLLAMA


# ------------------------------------------------------------------------------ committed defaults and secrets


def env_example() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_sc_15_committed_defaults_do_not_enable_integration() -> None:
    assert RuntimeConfig().run_ollama_integration is False and RuntimeConfig().submission_mode is False
    assert RuntimeConfig().enable_mock is False and OllamaConfig().integration_enabled is False
    example = env_example()
    assert example["RUN_OLLAMA_INTEGRATION"] == "false" and example["SUBMISSION_MODE"] == "false"
    assert example["ENABLE_MOCK_LLM"] == "false" and example["LLM_RUNTIME_MODE"] == "LIVE"
    with pytest.raises(LLMConfigurationError):
        load_runtime_config({})  # the committed defaults alone never start a live run
    assert load_ollama_config({}).integration_enabled is False


def test_sc_16_ollama_base_url_defaults_to_localhost() -> None:
    assert OllamaConfig().base_url == DEFAULT_BASE_URL == "http://127.0.0.1:11434"
    assert env_example()["OLLAMA_BASE_URL"] == DEFAULT_BASE_URL


def test_sc_17_remote_hosts_are_rejected_unless_explicitly_allowed() -> None:
    with pytest.raises(ValidationError, match="localhost"):
        OllamaConfig(base_url="http://203.0.113.9:11434")
    assert OllamaConfig(base_url="http://203.0.113.9:11434", allow_remote_host=True).allow_remote_host
    assert env_example()["OLLAMA_ALLOW_REMOTE_HOST"] == "false"
    with pytest.raises(LLMConfigurationError):
        load_ollama_config({"OLLAMA_BASE_URL": "http://203.0.113.9:11434"})


def test_sc_18_configuration_repr_and_logs_expose_no_secrets(caplog: pytest.LogCaptureFixture) -> None:
    env = {**SUBMISSION_ENV, "OLLAMA_API_KEY": "SECRET-KEY-123", "HF_TOKEN": "TOKEN-456", "PASSWORD": "PW-789"}
    with caplog.at_level(logging.DEBUG):
        runtime = load_runtime_config(env)
        ollama = load_ollama_config(env)
        shown = " ".join([repr(runtime), repr(ollama), str(ollama.safe_summary()), repr(inspect_runtime(env))])
    for secret in ("SECRET-KEY-123", "TOKEN-456", "PW-789"):
        assert secret not in shown and secret not in caplog.text
    assert not [f for f in (*RuntimeConfig.model_fields, *OllamaConfig.model_fields)
                if re.search(r"(key|token|password|secret|credential)", f, re.I)]


def test_sc_19_no_remediation_capability_was_introduced() -> None:
    pattern = re.compile(r"(block_|_block\b|disable_|quarantine|terminate|kill_|firewall|isolate_|remediat|attack|exploit)", re.I)
    for relative in ("config/runtime_config.py", "services/llm_runtime.py", "devtools/mock_llm_runtime.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
            n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert not [n for n in names if pattern.search(n)], relative
    policies = json.loads((DATA_DIR / "policies.json").read_text(encoding="utf-8"))
    assert all(p["automated_actions_permitted"] is False and p["human_authorization_required"] is True
               for p in (policies if isinstance(policies, list) else policies.get("policies", policies)))


def test_the_live_timeout_supports_300_seconds() -> None:
    assert OllamaConfig().request_timeout_seconds == 300.0
    assert OllamaConfig(request_timeout_seconds=600).request_timeout_seconds == 600
    assert load_ollama_config({"OLLAMA_REQUEST_TIMEOUT_SECONDS": "300"}).request_timeout_seconds == 300.0
    assert env_example()["OLLAMA_REQUEST_TIMEOUT_SECONDS"] == "300" and env_example()["OLLAMA_MODEL"] == "qwen3:4b"
    from services.ollama_chat_adapter import _default_chat_ollama_factory

    chat = _default_chat_ollama_factory(OllamaConfig(model="qwen3:4b"))  # constructs only; no connection
    assert type(chat).__name__ == "ChatOllama"


def test_documentation_files_exist_and_do_not_fabricate_live_results() -> None:
    guide = (ROOT / "docs" / "local_ollama_setup.md").read_text(encoding="utf-8")
    for required in ("qwen3:4b", "127.0.0.1:11434", "300", "source .venv/bin/activate"):
        assert required in guide, required
    results = (ROOT / "docs" / "test_results.md").read_text(encoding="utf-8")
    assert "Pending live run" in results and "Pending live Ollama validation" in results
    assert "passed validation" not in (ROOT / "README.md").read_text(encoding="utf-8")
