"""LLMAgent tests (LA-01..LA-17) plus the Ollama adapter's offline behaviour. No live Ollama is contacted."""

import re
import socket
import sys

import pytest

from agents.base import BaseAgent
from agents.llm_agent import AGENT_NAME, LLMAgent
from config.llm_config import OllamaConfig
from models.llm_models import RuntimeMode
from models.schemas import AgentDecision, AgentType
from services.llm_errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMDependencyUnavailableError,
    LLMModelError,
)
from services.ollama_chat_adapter import OllamaChatModelAdapter, map_model_exception
from services.scenario_service import get_all_scenarios, get_scenario
from tests.fakes.fake_tool_calling_model import MOCK_OUTCOMES, FAKE_MODEL_NAME, happy_path_model
from tools.langchain_security_tools import TOOL_NAMES

SCENARIO_IDS = ["A", "B", "C"]


def decide(scenario_id: str, *, parallel: bool = True) -> AgentDecision:
    scenario = get_scenario(scenario_id)
    return LLMAgent(model=happy_path_model(scenario, parallel=parallel)).decide(scenario)


def test_la_01_llm_agent_inherits_directly_from_base_agent() -> None:
    assert LLMAgent.__bases__ == (BaseAgent,)


def test_la_02_llm_agent_is_not_a_classical_agent() -> None:
    from agents import (
        GoalBasedAgent,
        LearningAgent,
        ModelBasedReflexAgent,
        SimpleReflexAgent,
        UtilityBasedAgent,
    )
    classical = (SimpleReflexAgent, ModelBasedReflexAgent, GoalBasedAgent, UtilityBasedAgent, LearningAgent)
    assert not any(issubclass(LLMAgent, c) or issubclass(c, LLMAgent) for c in classical)


def test_la_03_the_injected_model_is_used() -> None:
    scenario = get_scenario("A")
    model = happy_path_model(scenario)
    LLMAgent(model=model).decide(scenario)
    assert model.call_count == 2 and model.bound_tool_names == TOOL_NAMES


def test_la_04_no_live_network_call_happens_in_mock_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("a network connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert decide("A").agent_type is AgentType.LLM


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_la_05_to_07_each_scenario_returns_a_valid_agent_decision(scenario_id: str) -> None:
    decision = decide(scenario_id)
    assert isinstance(decision, AgentDecision) and AgentDecision.model_validate(decision.model_dump())
    priority, outcome = MOCK_OUTCOMES[scenario_id]
    assert (decision.priority.value, decision.decision) == (priority, outcome)
    assert decision.agent_name == AGENT_NAME and decision.agent_type is AgentType.LLM
    assert decision.reasoning and decision.recommended_actions and decision.evidence


def test_la_08_all_three_use_the_same_scenario_contract() -> None:
    scenarios = get_all_scenarios()
    assert {type(s) for s in scenarios} == {type(get_scenario("A"))}
    assert [decide(s.scenario_id).metadata["decision_mechanism"] for s in scenarios] == ["llm_dynamic_tool_calling"] * 3


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_la_09_and_10_tool_calls_record_actual_distinct_tools(scenario_id: str) -> None:
    decision = decide(scenario_id)
    names = [call["tool_name"] for call in decision.tool_calls]
    assert names and set(names) <= set(TOOL_NAMES)
    assert len(set(names)) >= 2
    assert all(call["status"] == "SUCCESS" for call in decision.tool_calls)
    assert decision.metadata["unique_tools_called"] == list(dict.fromkeys(names))


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_la_11_source_grounding_is_populated(scenario_id: str) -> None:
    decision = decide(scenario_id)
    scenario = get_scenario(scenario_id)
    sources = decision.metadata["source_record_ids"]
    assert scenario.alert_id in sources and scenario.asset_id in sources and scenario.policy_id in sources
    assert decision.metadata["structured_evidence"] and all(r["passed"] for r in decision.metadata["grounding_validation"])
    assert all(r["passed"] for r in decision.metadata["guardrail_validation"])


def test_la_12_and_13_utility_and_experience_stay_empty() -> None:
    decision = decide("A")
    assert decision.utility_score is None and decision.experience_used is None


def test_la_14_metadata_declares_llm_dynamic_tool_calling() -> None:
    meta = decide("A").metadata
    assert meta["decision_mechanism"] == "llm_dynamic_tool_calling"
    assert meta["uses_llm"] is True and meta["uses_dynamic_tools"] is True and meta["is_synthetic"] is True
    assert meta["executed_actions"] == [] and meta["recommendation_only"] is True
    assert meta["human_authorization_required_for_consequential_action"] is True


def test_la_15_runtime_mode_says_mock() -> None:
    decision = decide("B")
    assert decision.metadata["runtime_mode"] == "MOCK" == RuntimeMode.MOCK.value
    assert decision.metadata["model_name"] == FAKE_MODEL_NAME
    assert "Mode: MOCK" in decision.audit_note and "recommendation only" in decision.audit_note


def test_la_16_deciding_does_not_mutate_the_scenario() -> None:
    scenario = get_scenario("C")
    before = scenario.model_dump()
    LLMAgent(model=happy_path_model(scenario)).decide(scenario)
    assert scenario.model_dump() == before


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_la_17_repeated_mock_runs_are_equivalent(scenario_id: str) -> None:
    def comparable(decision: AgentDecision) -> dict:
        data = decision.model_dump(mode="json")
        data["metadata"].pop("execution_trace")  # holds the run ID and timestamps, which differ by design
        data["audit_note"] = re.sub(r"Run: \w+", "Run: <id>", data["audit_note"])
        return data

    assert comparable(decide(scenario_id)) == comparable(decide(scenario_id))


def test_both_call_styles_reach_the_same_decision() -> None:
    parallel, sequential = decide("C", parallel=True), decide("C", parallel=False)
    assert (parallel.priority, parallel.decision) == (sequential.priority, sequential.decision)


def test_construction_never_builds_or_contacts_a_model() -> None:
    agent = LLMAgent()  # no model, no config, no Ollama
    assert isinstance(agent, BaseAgent) and agent.name == AGENT_NAME


def test_llm_agent_without_a_model_name_fails_with_a_clear_configuration_error(
    clean_runtime_environment: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(LLMConfigurationError, match="OLLAMA_MODEL"):
        LLMAgent().decide(get_scenario("A"))


# ---------------------------------------------------------------- adapter (offline)


def test_adapter_construction_performs_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("a network connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    adapter = OllamaChatModelAdapter(OllamaConfig(model="example-model:tag"))
    assert adapter.runtime_mode is RuntimeMode.OLLAMA and adapter.model_name == "example-model:tag"
    chat = adapter._chat_model()  # builds ChatOllama; still offline
    assert type(chat).__name__ == "ChatOllama"


def test_adapter_requires_an_explicit_model() -> None:
    with pytest.raises(LLMConfigurationError):
        OllamaChatModelAdapter(OllamaConfig())


def test_adapter_uses_the_injected_factory_and_maps_invoke_errors() -> None:
    import httpx

    class Failing:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise httpx.ConnectError("refused")

    adapter = OllamaChatModelAdapter(OllamaConfig(model="m"), chat_model_factory=lambda _config: Failing())
    with pytest.raises(LLMConnectionError, match="local Ollama server"):
        adapter.bind_tools([]).invoke([])


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("raw server text that must not be echoed")
        self.status_code = status_code


@pytest.mark.parametrize(("error", "expected"), [
    (ConnectionError("x"), LLMConnectionError), (TimeoutError("x"), LLMConnectionError),
    (_StatusError(404), LLMModelError), (_StatusError(500), LLMModelError), (ValueError("x"), LLMModelError),
])
def test_model_exceptions_are_mapped_to_safe_errors(error: Exception, expected: type) -> None:
    mapped = map_model_exception(error)
    assert isinstance(mapped, expected) and "raw server text" not in str(mapped)


def test_missing_langchain_ollama_is_a_clear_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)  # makes `import langchain_ollama` fail
    adapter = OllamaChatModelAdapter(OllamaConfig(model="m"))
    with pytest.raises(LLMDependencyUnavailableError, match="pip install"):
        adapter.bind_tools([])
