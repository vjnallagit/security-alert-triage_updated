"""Real Ollama integration tests (OI-01..OI-15).

These exercise the REAL ChatOllama adapter and a genuinely tool-calling local model; the scripted fake is
never used here. They are skipped unless RUN_OLLAMA_INTEGRATION=true, and they need a reachable local Ollama
server with OLLAMA_MODEL set to a pulled model that supports tool calling.

    $env:RUN_OLLAMA_INTEGRATION="true"
    $env:OLLAMA_MODEL="<exact-model-tag>"
    python -m pytest -m integration -v

Local models vary in wording and in the order of tool calls, so the assertions check structure, grounding and
safety rather than exact text or an exact sequence.
"""

import hashlib
import os
from pathlib import Path

import pytest

from agents.llm_agent import LLMAgent
from config.llm_config import load_ollama_config
from models.schemas import AgentDecision, Priority
from services.llm_orchestrator import LLMOrchestrator
from services.ollama_chat_adapter import OllamaChatModelAdapter, list_available_models, require_model_available
from services.scenario_service import get_all_scenarios, get_scenario
from tools.langchain_security_tools import TOOL_NAMES, SecurityToolbox

SKIP_REASON = (
    "Real Ollama integration requires RUN_OLLAMA_INTEGRATION=true and a reachable local Ollama server."
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_OLLAMA_INTEGRATION", "").strip().lower() != "true", reason=SKIP_REASON),
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCENARIO_IDS = ["A", "B", "C"]


@pytest.fixture(scope="module")
def config():
    return load_ollama_config(integration_enabled=True)


@pytest.fixture(scope="module")
def adapter(config):
    return OllamaChatModelAdapter(config)


@pytest.fixture(scope="module")
def live_runs(config, adapter):
    """One real run per scenario (captured, so one failure does not hide the others)."""
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_DIR.glob("*.json")}
    runs: dict[str, tuple] = {}
    for scenario in get_all_scenarios():
        toolbox = SecurityToolbox()
        try:
            result = LLMOrchestrator(adapter, config, toolbox_factory=lambda tb=toolbox: tb).run(scenario)
            runs[scenario.scenario_id] = (result, toolbox, None)
        except Exception as exc:  # reported per test with a safe message
            runs[scenario.scenario_id] = (None, toolbox, f"{type(exc).__name__}: {exc}")
    runs["_data_before"] = before
    return runs


def live(live_runs, scenario_id: str):
    result, toolbox, error = live_runs[scenario_id]
    if error:
        pytest.fail(f"Live run for scenario {scenario_id} failed: {error}")
    return result, toolbox


def test_oi_01_ollama_base_url_is_reachable(config) -> None:
    assert isinstance(list_available_models(config), tuple)


def test_oi_02_configured_model_is_available(config) -> None:
    assert config.require_model() in require_model_available(config) or f"{config.model}:latest" in list_available_models(config)


def test_oi_03_basic_chat_invocation_returns_content(adapter) -> None:
    assert adapter.ask("Reply with the single word: ready").strip()


def test_oi_04_configured_model_supports_bind_tools(adapter) -> None:
    assert adapter.bind_tools(SecurityToolbox().tools) is not None


def test_oi_05_model_emits_at_least_one_genuine_tool_call(adapter) -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    bound = adapter.bind_tools(SecurityToolbox().tools)
    reply = bound.invoke([
        SystemMessage(content="You are a security triage assistant. Use the provided tools to retrieve evidence."),
        HumanMessage(content="Look up the details for alert ALERT-A-001 using a tool."),
    ])
    assert reply.tool_calls and reply.tool_calls[0]["name"] in TOOL_NAMES


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_oi_06_to_08_each_scenario_emits_two_distinct_relevant_tool_calls(live_runs, scenario_id: str) -> None:
    result, _ = live(live_runs, scenario_id)
    assert len(result.collector.evidence_tool_names) >= 2
    assert set(result.collector.evidence_tool_names) <= set(TOOL_NAMES)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_oi_09_actual_phase_6_tools_execute(live_runs, scenario_id: str) -> None:
    result, toolbox = live(live_runs, scenario_id)
    logged = {entry.tool_name for entry in toolbox.service.get_invocation_log()}
    assert len(logged) >= 2 and logged <= set(TOOL_NAMES)
    assert any(call.status.value == "SUCCESS" for call in result.trace.tool_calls)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_oi_10_final_output_validates_as_agent_decision(live_runs, scenario_id: str) -> None:
    result, _ = live(live_runs, scenario_id)
    decision = LLMAgent._to_decision(get_scenario(scenario_id), result)
    assert AgentDecision.model_validate(decision.model_dump())
    assert decision.priority in set(Priority)
    assert decision.decision in {"MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"}
    assert decision.metadata["runtime_mode"] == "OLLAMA"


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_oi_11_and_12_recommendation_is_grounded_with_no_fabricated_sources(live_runs, scenario_id: str) -> None:
    result, _ = live(live_runs, scenario_id)
    assert all(check.passed for check in result.grounding_results)
    assert set(result.recommendation.source_record_ids) <= set(result.collector.source_record_ids)
    assert result.recommendation.source_record_ids


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_oi_13_safety_guardrails_pass(live_runs, scenario_id: str) -> None:
    result, _ = live(live_runs, scenario_id)
    assert all(check.passed for check in result.guardrail_results)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_oi_14_no_disruptive_action_executes(live_runs, scenario_id: str) -> None:
    result, toolbox = live(live_runs, scenario_id)
    decision = LLMAgent._to_decision(get_scenario(scenario_id), result)
    assert decision.metadata["executed_actions"] == []
    assert {e.tool_name for e in toolbox.service.get_invocation_log()} <= set(TOOL_NAMES)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_DIR.glob("*.json")}
    assert after == live_runs["_data_before"]


def test_oi_15_all_scenarios_use_the_same_shared_scenario_contract(live_runs) -> None:
    contracts = set()
    for scenario_id in SCENARIO_IDS:
        result, _ = live(live_runs, scenario_id)
        assert result.trace.scenario_id == scenario_id
        contracts.add(type(get_scenario(scenario_id)))
    assert len(contracts) == 1
