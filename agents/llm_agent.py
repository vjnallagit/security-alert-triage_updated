"""LLM-based agent: a language model decides which evidence tools to call, then recommends a triage action.

Unlike the five classical agents (fixed rules, state, goals, utilities or learned statistics), the decision
here comes from a model that dynamically chooses tool calls and reasons over their results. This class only
wires the pieces together and converts the validated recommendation into the shared AgentDecision contract:

    LLMOrchestrator   -> runs the bounded tool-calling loop and validates grounding and guardrails
    ToolCallingChatModel -> Ollama in real use, a scripted fake in offline tests

Constructing the agent never contacts Ollama; the model is created when `decide()` is first called.
"""

from agents.base import BaseAgent
from agents.common import HUMAN_AUTHORIZATION_STATEMENT, NO_ACTION_STATEMENT, safety_metadata
from config.llm_config import OllamaConfig, load_ollama_config
from models.schemas import AgentDecision, AgentType, SecurityScenario
from services.llm_orchestrator import LLMOrchestrator, OrchestrationResult
from services.ollama_chat_adapter import OllamaChatModelAdapter, ToolCallingChatModel

AGENT_NAME = "LLM Agent"


class LLMAgent(BaseAgent):
    name = AGENT_NAME

    def __init__(self, model: ToolCallingChatModel | None = None, config: OllamaConfig | None = None) -> None:
        self._model = model
        self._config = config

    def decide(self, scenario: SecurityScenario) -> AgentDecision:
        config = self._config or load_ollama_config()
        model = self._model or OllamaChatModelAdapter(config)
        result = LLMOrchestrator(model, config).run(scenario)
        return self._to_decision(scenario, result)

    @staticmethod
    def _to_decision(scenario: SecurityScenario, result: OrchestrationResult) -> AgentDecision:
        recommendation, trace = result.recommendation, result.trace
        unique_tools = tuple(dict.fromkeys(call.tool_name for call in trace.tool_calls))
        evidence = [
            f"[{item.tool_name} / {item.tool_call_id}] {item.claim}"
            + (f" (sources: {', '.join(item.source_record_ids)})" if item.source_record_ids else " (no matching records)")
            for item in recommendation.evidence
        ]
        reasoning = (
            f"The language model called {len(trace.tool_calls)} tool(s) ({', '.join(unique_tools)}) and "
            f"recommended {recommendation.decision} at {recommendation.priority.value} priority. Its answer was "
            f"accepted only after every cited record was checked against the real tool results and the safety "
            f"guardrails passed. Interpretation: {recommendation.likely_interpretation}"
        )
        audit_note = (
            f"Scenario {scenario.scenario_id} | Agent: {AGENT_NAME} | Mode: {trace.runtime_mode.value} | "
            f"Model: {trace.model_name} | Run: {trace.run_id} | {recommendation.audit_note} | "
            f"{NO_ACTION_STATEMENT} {HUMAN_AUTHORIZATION_STATEMENT}"
        )
        return AgentDecision(
            agent_name=AGENT_NAME, agent_type=AgentType.LLM, priority=recommendation.priority,
            decision=recommendation.decision, reasoning=reasoning, evidence=evidence,
            uncertainty=recommendation.uncertainty, likely_interpretation=recommendation.likely_interpretation,
            recommended_actions=list(recommendation.recommended_actions), audit_note=audit_note,
            tool_calls=[call.model_dump(mode="json") for call in trace.tool_calls],
            utility_score=None, experience_used=None,
            metadata={
                **safety_metadata(),
                "decision_mechanism": "llm_dynamic_tool_calling",
                "runtime_mode": trace.runtime_mode.value, "model_name": trace.model_name,
                "tool_call_count": len(trace.tool_calls), "unique_tools_called": list(unique_tools),
                "source_record_ids": list(trace.source_record_ids),
                "execution_trace": trace.model_dump(mode="json"),
                "grounding_validation": [r.model_dump(mode="json") for r in result.grounding_results],
                "guardrail_validation": [r.model_dump(mode="json") for r in result.guardrail_results],
                "structured_evidence": [item.model_dump(mode="json") for item in recommendation.evidence],
                "uses_llm": True, "uses_dynamic_tools": True, "is_synthetic": True,
            },
        )
