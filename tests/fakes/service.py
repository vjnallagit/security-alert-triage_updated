"""Test helper: a comparison service whose MOCK runtime has the scripted model INJECTED explicitly.

Production code never loads a mock by itself; tests wire it in here, exactly as the injection design requires.
LIVE (OLLAMA) services are deliberately NOT given a mock, so a test can never pass because of a silent fallback.
"""

from typing import Any

from devtools.mock_llm_runtime import happy_path_model, mock_llm_agent_factory
from langchain_core.messages import AIMessage
from models.llm_models import RuntimeMode
from models.schemas import AgentType
from services.comparison_service import AgentComparisonService

THINKING_MARKER = "TOP-SECRET-THOUGHT-7731"


def mock_service(**kwargs: Any) -> AgentComparisonService:
    kwargs.setdefault("llm_mode", RuntimeMode.MOCK)

    factories = dict(kwargs.pop("agent_factories", None) or {})

    if kwargs["llm_mode"] is RuntimeMode.MOCK:
        factories.setdefault(
            AgentType.LLM,
            mock_llm_agent_factory,
        )

    return AgentComparisonService(
        agent_factories=factories,
        **kwargs,
    )


class SimulatedLiveAdapter:
    """TEST ONLY: stands in for OllamaChatModelAdapter so LIVE code paths
    run without any Ollama server.

    It reports runtime OLLAMA, so a test can check that the UI labels a
    simulated live result correctly.

    With leak_thinking it also attaches reasoning to every reply, to prove
    that content is discarded.
    """

    runtime_mode = RuntimeMode.OLLAMA
    leak_thinking = False
    created = 0

    def __init__(self, config: Any) -> None:
        type(self).created += 1
        self.model_name = config.model
        self._inner: Any = None

    def bind_tools(self, _tools: Any) -> "SimulatedLiveAdapter":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        import json

        if len(messages) == 2:
            from services.scenario_service import get_scenario

            content = messages[1].content
            json_start = content.find("{")

            if json_start == -1:
                raise ValueError(
                    "Scenario JSON not found in LLM message."
                )

            scenario_data = json.loads(content[json_start:])
            scenario = get_scenario(
                scenario_data["scenario_id"]
            )

            self._inner = happy_path_model(
                scenario,
                parallel=False,
            )

        reply = self._inner.invoke(messages)

        if not type(self).leak_thinking:
            return reply

        return AIMessage(
            content=(
                f"<think>{THINKING_MARKER}</think>"
                f"{reply.content}"
            ),
            tool_calls=reply.tool_calls,
            additional_kwargs={
                "reasoning_content": THINKING_MARKER,
                "thinking": THINKING_MARKER,
            },
            response_metadata={
                "thinking": THINKING_MARKER,
                "reasoning": THINKING_MARKER,
            },
        )