"""LangChain wrappers around the four Phase 6 security evidence tools.

These are thin adapters: filtering, validation, source lookup, summaries and audit all stay in
SecurityToolService (Phase 6). Each wrapper returns the compact JSON from `to_llm_evidence()` plus the
safety declarations, so the model sees status, evidence, source record IDs, warnings and limitations.
The tools retrieve evidence only; they never make a decision or take an action.

LangChain is imported lazily, so importing this module needs no LangChain package and does no I/O.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from models.llm_models import CallStatus
from models.tool_models import TOOL_CATALOGUE, ToolDefinition, ToolResultBase
from pydantic import BaseModel, ValidationError
from services.llm_errors import LLMDependencyUnavailableError
from services.security_tool_service import SecurityToolService

# Tool names must stay exactly the Phase 6 public names.
TOOL_NAMES = tuple(definition.name for definition in TOOL_CATALOGUE)

# Wording the live model uses to decide which tool to call.
TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_alert_details": (
        "Retrieve the synthetic factual alert record for an alert ID. Use this when alert type, severity, "
        "target, source or recorded alert evidence is needed. This tool does not determine final priority "
        "or execute actions."
    ),
    "search_authentication_events": (
        "Search synthetic authentication events using one or more filters such as account, source, asset, "
        "scenario, success state or time range. Use this to verify repeated failures, successful activity, "
        "source patterns and event counts. This tool does not confirm compromise."
    ),
    "get_asset_criticality": (
        "Retrieve the synthetic asset record and its business/security criticality. Use this when the "
        "consequence of an alert depends on the importance of the affected asset. Criticality is evidence, "
        "not a final incident priority."
    ),
    "get_response_policy": (
        "Retrieve synthetic defensive response guidance, escalation criteria and prohibited actions. Use "
        "this before recommending investigation, escalation or any potentially consequential response. "
        "The tool never executes the policy."
    ),
}


@dataclass(frozen=True)
class ToolExecution:
    """What happened when the model asked for a tool: the JSON-safe payload it will see and its status."""

    tool_name: str
    arguments: dict[str, Any]
    payload: dict[str, Any]
    status: CallStatus

    @property
    def content(self) -> str:
        return json.dumps(self.payload, sort_keys=True)


def tool_payload(result: ToolResultBase) -> dict[str, Any]:
    """Compact evidence for the model: Phase 6's `to_llm_evidence()` plus explicit safety declarations."""
    payload = result.to_llm_evidence()
    payload.update(
        is_synthetic=result.is_synthetic, is_read_only=result.is_read_only,
        executed_actions=list(result.executed_actions), audit_call_id=result.call_id,
    )
    return payload


def _make_runner(service: SecurityToolService, tool_name: str) -> Callable[..., str]:
    method = getattr(service, tool_name)

    def run(**kwargs: Any) -> str:
        return json.dumps(tool_payload(method(**kwargs)), sort_keys=True)

    run.__name__ = tool_name
    return run


def build_langchain_tools(service: SecurityToolService) -> tuple[Any, ...]:
    """One LangChain StructuredTool per Phase 6 tool, with the Phase 6 Pydantic model as its argument schema."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise LLMDependencyUnavailableError(
            "langchain-core is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return tuple(
        StructuredTool.from_function(
            func=_make_runner(service, definition.name), name=definition.name,
            description=TOOL_DESCRIPTIONS[definition.name], args_schema=definition.input_model,
        )
        for definition in TOOL_CATALOGUE
    )


def _error_payload(tool_name: str, status: CallStatus, message: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "tool_name": tool_name, "status": status.value, "summary": message, "evidence": None,
        "source_record_ids": [], "warnings": [], "limitations": ["No evidence was retrieved for this call."],
        "error": message, "is_synthetic": True, "is_read_only": True, "executed_actions": [], **(extra or {}),
    }


def _safe_arguments(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    return {str(k): (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)[:200])
            for k, v in arguments.items()}


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "Invalid tool arguments: " + "; ".join(parts)


class SecurityToolbox:
    """The four LangChain tools plus a safe dispatcher. Unknown tools and bad arguments never execute."""

    def __init__(self, service: SecurityToolService | None = None) -> None:
        self._service = service or SecurityToolService()
        self._tools = build_langchain_tools(self._service)
        self._by_name = {tool.name: tool for tool in self._tools}
        self._definitions: dict[str, ToolDefinition] = {d.name: d for d in TOOL_CATALOGUE}

    @property
    def tools(self) -> tuple[Any, ...]:
        return self._tools

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    @property
    def service(self) -> SecurityToolService:
        return self._service

    def prepare(self, tool_name: str, arguments: Any) -> tuple[dict[str, Any] | None, ToolExecution | None]:
        """Validate a model-requested call without running it.

        Returns (validated_arguments, None) when it may run, or (None, error_execution) for an unknown tool,
        a non-object argument payload, or arguments that do not match the tool's Pydantic input schema
        (including unexpected arguments). Nothing is evaluated or executed here.
        """
        if tool_name not in self._by_name:
            message = f"Unknown tool. Available tools: {', '.join(self.tool_names)}."
            return None, ToolExecution(
                tool_name, _safe_arguments(arguments),
                _error_payload(str(tool_name)[:60], CallStatus.UNKNOWN_TOOL, message,
                               {"available_tools": list(self.tool_names)}), CallStatus.UNKNOWN_TOOL)
        if not isinstance(arguments, dict):
            message = "Invalid tool arguments: arguments must be a JSON object."
            return None, ToolExecution(tool_name, {}, _error_payload(tool_name, CallStatus.INVALID_INPUT, message),
                                       CallStatus.INVALID_INPUT)
        model: type[BaseModel] = self._definitions[tool_name].input_model
        try:
            validated = model.model_validate(arguments)  # also rejects unexpected arguments
        except ValidationError as exc:
            return None, ToolExecution(
                tool_name, _safe_arguments(arguments),
                _error_payload(tool_name, CallStatus.INVALID_INPUT, _validation_message(exc)), CallStatus.INVALID_INPUT)
        return validated.model_dump(mode="json", exclude_none=True), None

    def run_prepared(self, tool_name: str, validated_arguments: dict[str, Any]) -> ToolExecution:
        """Run a call that `prepare` accepted, through the LangChain tool and the Phase 6 service."""
        model: type[BaseModel] = self._definitions[tool_name].input_model
        python_arguments = model.model_validate(validated_arguments).model_dump(exclude_none=True)
        payload = json.loads(self._by_name[tool_name].invoke(python_arguments))
        return ToolExecution(tool_name, validated_arguments, payload, CallStatus(payload["status"]))

    def execute(self, tool_name: str, arguments: Any) -> ToolExecution:
        """Validate and run one model-requested call. Never raises for bad input."""
        validated, rejection = self.prepare(tool_name, arguments)
        return rejection if rejection is not None else self.run_prepared(tool_name, validated or {})
