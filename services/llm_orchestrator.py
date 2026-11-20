"""The LLM tool-calling loop: model -> tool calls -> observations -> final answer -> validation.

One `run()` handles one scenario and keeps all its state in local variables, so a single orchestrator can be
reused safely. The loop is bounded by `max_tool_iterations`. The model decides WHICH tools to call; this
code only validates and executes those calls (read-only, through Phase 6), feeds the results back, and then
accepts the final JSON answer only if deterministic grounding and guardrail checks pass.

Nothing here decides a triage outcome, parses prose, or fills in missing evidence.
"""

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from config.llm_config import OllamaConfig
from models.llm_models import (
    CallStatus,
    GuardrailResult,
    LLMExecutionTrace,
    LLMToolCallRecord,
    LLMTriageRecommendation,
)
from models.schemas import SecurityScenario
from services.evidence_collector import EvidenceCollector, evaluate_grounding
from services.llm_errors import (
    GroundingValidationError,
    InsufficientToolEvidenceError,
    LLMError,
    LLMIterationLimitError,
    LLMOutputValidationError,
)
from services.llm_guardrails import enforce_guardrails, evaluate_guardrails
from services.llm_prompts import (
    SYSTEM_PROMPT,
    build_repair_message,
    build_scenario_message,
    build_tool_correction_message,
)
from services.ollama_chat_adapter import ToolCallingChatModel, map_model_exception
from tools.langchain_security_tools import SecurityToolbox

_THINK_BLOCK = re.compile(r"<think>.*?</think>|<think>.*\Z", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
# Fields some models/clients attach to a reply. They are never kept, shown, traced, logged or sent back.
THINKING_FIELD_NAMES = frozenset({
    "thinking", "thought", "thoughts", "reasoning", "reasoning_content", "chain_of_thought", "chainofthought",
})


def _without_thinking_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_thinking_fields(v) for k, v in value.items()
                if str(k).lower() not in THINKING_FIELD_NAMES}
    if isinstance(value, list):
        return [_without_thinking_fields(item) for item in value]
    return value


def sanitize_reply(reply: Any) -> Any:
    """A copy of the model reply with all thinking content removed.

    Keeps only the visible text, the tool calls (id, name, arguments) and non-thinking extras. Response
    metadata is dropped entirely. The original reply object is discarded by the caller.
    """
    from langchain_core.messages import AIMessage

    text = _THINK_BLOCK.sub("", content_to_text(getattr(reply, "content", ""))).strip()
    return AIMessage(
        content=text,
        tool_calls=list(getattr(reply, "tool_calls", None) or []),
        invalid_tool_calls=list(getattr(reply, "invalid_tool_calls", None) or []),
        additional_kwargs=_without_thinking_fields(dict(getattr(reply, "additional_kwargs", None) or {})),
    )


@dataclass(frozen=True)
class OrchestrationResult:
    recommendation: LLMTriageRecommendation
    trace: LLMExecutionTrace
    collector: EvidenceCollector
    grounding_results: tuple[GuardrailResult, ...]
    guardrail_results: tuple[GuardrailResult, ...]


class _AnswerRejected(Exception):
    """Internal: the final answer failed parsing or grounding and may be repaired once."""

    def __init__(self, problems: tuple[str, ...], error: type[LLMError]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
        self.error = error


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def content_to_text(content: Any) -> str:
    """Model content is a string or a list of content blocks; return its text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", "")) if isinstance(block, dict) else ""
            for block in content
        )
    return ""


def extract_json_object(text: str) -> dict[str, Any]:
    """Strict JSON only. Removes a `<think>` block and one surrounding code fence, nothing else."""
    cleaned = _THINK_BLOCK.sub("", text).strip()
    if (fenced := _CODE_FENCE.match(cleaned)) is not None:
        cleaned = fenced.group(1).strip()
    if not cleaned:
        raise _AnswerRejected(("the final answer was empty; return the JSON object",), LLMOutputValidationError)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise _AnswerRejected((f"the final answer is not valid JSON ({exc.msg} at position {exc.pos})",),
                              LLMOutputValidationError) from exc
    if not isinstance(parsed, dict):
        raise _AnswerRejected(("the final answer must be a single JSON object",), LLMOutputValidationError)
    return parsed


def parse_recommendation(text: str) -> LLMTriageRecommendation:
    data = extract_json_object(text)
    try:
        return LLMTriageRecommendation.model_validate(data)
    except ValidationError as exc:
        problems = tuple(
            f"{'.'.join(str(p) for p in e['loc']) or 'answer'}: {e['msg']}"
            for e in exc.errors(include_input=False, include_url=False)
        )
        raise _AnswerRejected(problems[:8], LLMOutputValidationError) from exc


def _evidence_count(payload: dict[str, Any]) -> int:
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        return 0
    events = evidence.get("events")
    return len(events) if isinstance(events, list) else 1


def _rejection_payload(tool_name: str, status: CallStatus, message: str) -> dict[str, Any]:
    return {
        "tool_name": tool_name, "status": status.value, "summary": message, "evidence": None,
        "source_record_ids": [], "warnings": [], "limitations": ["No evidence was retrieved for this call."],
        "error": message, "is_synthetic": True, "is_read_only": True, "executed_actions": [],
    }


class LLMOrchestrator:
    def __init__(
        self, model: ToolCallingChatModel, config: OllamaConfig | None = None, *,
        toolbox_factory: Callable[[], SecurityToolbox] = SecurityToolbox,
        clock: Callable[[], datetime] = _utc_now, run_id_factory: Callable[[], str] = _new_run_id,
    ) -> None:
        self._model = model
        self._config = config or OllamaConfig()
        self._toolbox_factory = toolbox_factory
        self._clock = clock
        self._run_id_factory = run_id_factory

    def run(self, scenario: SecurityScenario) -> OrchestrationResult:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        started_at, run_id = self._clock(), self._run_id_factory()
        toolbox = self._toolbox_factory()  # a fresh toolbox (and audit log) for every run
        collector = EvidenceCollector(scenario, self._config.minimum_tool_calls)
        call_records: list[LLMToolCallRecord] = []
        warnings: list[str] = []
        guard_results: tuple[GuardrailResult, ...] = ()
        iterations = 0
        correction_used = repair_used = False

        def trace(valid: bool, errors: tuple[str, ...] = ()) -> LLMExecutionTrace:
            return LLMExecutionTrace(
                run_id=run_id, scenario_id=scenario.scenario_id, runtime_mode=self._model.runtime_mode,
                model_name=self._model.model_name, started_at=started_at, completed_at=self._clock(),
                iterations=iterations, tool_calls=tuple(call_records), tool_results=collector.result_records(),
                source_record_ids=collector.source_record_ids, final_output_valid=valid,
                guardrail_results=guard_results, errors=errors, warnings=tuple(warnings))

        def fail(error: LLMError) -> LLMError:
            error.trace = trace(False, (str(error)[:300],))
            return error

        try:
            bound = self._model.bind_tools(toolbox.tools)
        except LLMError as exc:
            raise fail(exc)
        except Exception as exc:
            raise fail(map_model_exception(exc)) from exc  # type: ignore[arg-type]

        messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=build_scenario_message(scenario))]

        def handle_call(name: Any, args: Any, call_id: Any, *, malformed: bool = False) -> None:
            tool_name = name if isinstance(name, str) else ""
            safe_id = call_id if isinstance(call_id, str) and call_id and collector.get(call_id) is None else None
            if safe_id is None:
                safe_id = f"call_{run_id}_{len(call_records) + 1}"
                warnings.append(f"A tool call had a missing or repeated id; assigned '{safe_id}'.")
            validated, rejection = toolbox.prepare(tool_name, None if malformed else args)
            if rejection is not None:
                execution_args, payload, status = rejection.arguments, rejection.payload, rejection.status
            elif collector.has_executed(tool_name, validated or {}):
                message = "This exact call was already made; use the earlier result instead of repeating it."
                execution_args, status = validated or {}, CallStatus.DUPLICATE_CALL
                payload = _rejection_payload(tool_name, status, message)
            else:
                execution_args = validated or {}
                try:
                    execution = toolbox.run_prepared(tool_name, execution_args)
                    payload, status = execution.payload, execution.status
                except Exception:
                    status = CallStatus.ERROR
                    payload = _rejection_payload(tool_name, status, "The tool failed unexpectedly; no evidence was retrieved.")
            observation = collector.record(safe_id, tool_name or "(unnamed)", execution_args, payload, status)
            call_records.append(LLMToolCallRecord(
                tool_call_id=safe_id, tool_name=tool_name or "(unnamed)", arguments=execution_args,
                iteration=iterations, status=status, source_record_ids=observation.source_record_ids,
                evidence_count=_evidence_count(payload)))
            messages.append(ToolMessage(content=json.dumps(payload, sort_keys=True), tool_call_id=safe_id, name=tool_name or None))

        while True:
            if iterations >= self._config.max_tool_iterations:
                raise fail(LLMIterationLimitError(
                    f"No valid final answer after {self._config.max_tool_iterations} model turns."))
            iterations += 1
            try:
                reply = bound.invoke(messages)
            except LLMError as exc:
                raise fail(exc)
            except Exception as exc:
                raise fail(map_model_exception(exc)) from exc  # type: ignore[arg-type]
            reply = sanitize_reply(reply)  # thinking content never enters the conversation, trace or output
            messages.append(reply)

            requested = list(getattr(reply, "tool_calls", None) or [])
            malformed = list(getattr(reply, "invalid_tool_calls", None) or [])
            if requested or malformed:
                for call in requested:
                    handle_call(call.get("name"), call.get("args"), call.get("id"))
                for call in malformed:
                    handle_call(call.get("name"), None, call.get("id"), malformed=True)
                continue

            if not collector.has_minimum_evidence():
                if correction_used:
                    raise fail(InsufficientToolEvidenceError(
                        f"The model answered without evidence from {self._config.minimum_tool_calls} distinct "
                        f"relevant tools (had: {', '.join(collector.evidence_tool_names) or 'none'})."))
                correction_used = True
                messages.append(HumanMessage(content=build_tool_correction_message(
                    self._config.minimum_tool_calls, collector.evidence_tool_names)))
                continue

            try:
                recommendation = parse_recommendation(content_to_text(reply.content))
                grounding, problems = evaluate_grounding(
                    recommendation, collector, scenario, self._config.minimum_tool_calls)
                if problems:
                    guard_results = grounding
                    raise _AnswerRejected(problems, GroundingValidationError)
            except _AnswerRejected as rejected:
                if repair_used:
                    raise fail(rejected.error(f"The corrected answer was still invalid: {rejected}"[:400])) from rejected
                repair_used = True
                warnings.append("The first final answer was rejected; one repair was requested.")
                messages.append(HumanMessage(content=build_repair_message(rejected.problems)))
                continue

            guardrails = evaluate_guardrails(recommendation, scenario)
            guard_results = grounding + guardrails
            try:
                enforce_guardrails(guardrails)
            except LLMError as violation:
                raise fail(violation)
            return OrchestrationResult(recommendation, trace(True), collector, grounding, guardrails)
