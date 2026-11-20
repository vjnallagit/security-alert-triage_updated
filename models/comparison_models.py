"""Typed result of running all six agents on the same scenario.

These models only carry results. They hold no decision logic: every priority, decision and explanation
comes from a real `AgentDecision` that an agent produced, which is preserved unchanged in `decision_record`.
"""

import hashlib
import json
from enum import StrEnum
from typing import Any, Self

from pydantic import AwareDatetime, Field, model_validator

from models.goal_models import FrozenModel
from models.llm_models import RuntimeMode
from models.schemas import AgentDecision, AgentType, Priority, SecurityScenario

# The comparison order, which is also the order agents appear in the UI.
AGENT_ORDER: tuple[AgentType, ...] = (
    AgentType.SIMPLE_REFLEX,
    AgentType.MODEL_BASED_REFLEX,
    AgentType.GOAL_BASED,
    AgentType.UTILITY_BASED,
    AgentType.LEARNING,
    AgentType.LLM,
)

AGENT_DISPLAY_NAMES: dict[AgentType, str] = {
    AgentType.SIMPLE_REFLEX: "Simple Reflex",
    AgentType.MODEL_BASED_REFLEX: "Model-Based Reflex",
    AgentType.GOAL_BASED: "Goal-Based",
    AgentType.UTILITY_BASED: "Utility-Based",
    AgentType.LEARNING: "Learning Agent",
    AgentType.LLM: "LLM Agent",
}

AGENT_MECHANISMS: dict[AgentType, str] = {
    AgentType.SIMPLE_REFLEX: "Current percept + condition/action rules",
    AgentType.MODEL_BASED_REFLEX: "Current percept + internal state + rules",
    AgentType.GOAL_BASED: "Goal evaluation + defensive plan selection",
    AgentType.UTILITY_BASED: "Expected utility + safe action ranking",
    AgentType.LEARNING: "Historical outcome feedback + adaptation",
    AgentType.LLM: "Local Ollama + dynamic evidence function calling",
}

# Which AgentDecision.metadata entries each agent's detail panel needs. Everything else stays in the
# preserved decision record, so the UI never has to dump raw metadata.
SPECIAL_DETAIL_KEYS: dict[AgentType, tuple[str, ...]] = {
    AgentType.SIMPLE_REFLEX: ("matched_rule_id", "matched_rule_name", "current_percept_only", "uses_internal_state"),
    AgentType.MODEL_BASED_REFLEX: ("matched_rule_id", "matched_rule_name", "state_summary", "uses_internal_state"),
    AgentType.GOAL_BASED: (
        "goal", "candidate_actions_considered", "rejected_actions", "acceptable_actions_not_selected",
        "selected_plan", "selection_policy",
    ),
    AgentType.UTILITY_BASED: (
        "action_ranking", "selected_action", "selected_expected_utility", "key_trade_off",
        "expected_utility_formula", "assumptions_note",
    ),
    AgentType.LEARNING: (
        "learning_pattern", "candidate_experience", "selected_experience", "experience_sufficient",
        "selection_basis", "smoothing", "min_experience_count", "limitation_note", "problem_generator_used",
    ),
    AgentType.LLM: (
        "runtime_mode", "model_name", "tool_call_count", "unique_tools_called", "source_record_ids",
        "execution_trace", "grounding_validation", "guardrail_validation", "structured_evidence",
    ),
}


class AgentStatus(StrEnum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


def scenario_fingerprint(scenario: SecurityScenario) -> str:
    """SHA-256 of the canonical scenario JSON. Contains only the scenario, never any agent output."""
    canonical = json.dumps(scenario.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AgentComparisonEntry(FrozenModel):
    """One agent's row in the comparison."""

    agent_type: AgentType
    agent_name: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    status: AgentStatus
    input_fingerprint: str = Field(min_length=1, description="Fingerprint of the scenario this agent was given.")
    execution_time_ms: float | None = Field(default=None, ge=0)
    priority: Priority | None = None
    decision: str | None = None
    reasoning_summary: str = ""
    evidence: tuple[str, ...] = ()
    uncertainty: str = ""
    likely_interpretation: str = ""
    recommended_actions: tuple[str, ...] = ()
    audit_note: str = ""
    # Only the LLM agent has a runtime mode; it is read from the real result, never assumed.
    llm_runtime_mode: RuntimeMode | None = None
    special_details: dict[str, Any] = {}
    error_message: str | None = None
    decision_record: AgentDecision | None = Field(default=None, description="The agent's original, unmodified output.")

    @model_validator(mode="after")
    def _status_matches_content(self) -> Self:
        if self.status is AgentStatus.ERROR:
            if not self.error_message:
                raise ValueError("an ERROR entry needs a safe error_message")
            if self.decision_record is not None:
                raise ValueError("an ERROR entry must not carry a decision")
            return self
        required = {
            "priority": self.priority, "decision": self.decision, "reasoning": self.reasoning_summary,
            "evidence": self.evidence, "uncertainty": self.uncertainty,
            "likely_interpretation": self.likely_interpretation, "recommended_actions": self.recommended_actions,
            "audit_note": self.audit_note, "decision_record": self.decision_record,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"a SUCCESS entry must provide the required assignment fields: {', '.join(missing)}")
        return self


class AgentFailure(FrozenModel):
    agent_type: AgentType
    agent_name: str
    message: str = Field(min_length=1)


class AgentComparisonResult(FrozenModel):
    scenario: SecurityScenario
    scenario_fingerprint: str = Field(min_length=1)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    results: tuple[AgentComparisonEntry, ...]
    errors: tuple[AgentFailure, ...] = ()
    llm_runtime_mode: RuntimeMode | None = Field(
        default=None, description="MOCK or OLLAMA; None when the LLM runtime is disabled or unconfigured.")

    @model_validator(mode="after")
    def _entries_are_consistent(self) -> Self:
        types = [entry.agent_type for entry in self.results]
        if len(set(types)) != len(types):
            raise ValueError("each agent may appear only once")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not be before started_at")
        failed = {entry.agent_type for entry in self.results if entry.status is AgentStatus.ERROR}
        if failed != {failure.agent_type for failure in self.errors}:
            raise ValueError("errors must list exactly the agents whose entry has ERROR status")
        return self

    def entry(self, agent_type: AgentType) -> AgentComparisonEntry:
        return next(entry for entry in self.results if entry.agent_type is agent_type)

    @property
    def successful(self) -> tuple[AgentComparisonEntry, ...]:
        return tuple(entry for entry in self.results if entry.status is AgentStatus.SUCCESS)

    @property
    def all_succeeded(self) -> bool:
        return not self.errors


class ModelBasedObservation(FrozenModel):
    """One call of the demonstration Model-Based agent."""

    observation_number: int = Field(ge=1)
    priority: Priority
    decision: str
    matched_rule_id: str
    account_observations: int = Field(ge=0)
    previous_decision: str | None = None


class FeedbackStep(FrozenModel):
    """One recorded outcome in the Learning demonstration, with the history before and after."""

    scenario_id: str
    pattern: str
    action: str
    outcome: str
    decision_before: str
    decision_after: str
    before_success: int = Field(ge=0)
    before_failure: int = Field(ge=0)
    before_smoothed_rate: float
    after_success: int = Field(ge=0)
    after_failure: int = Field(ge=0)
    after_smoothed_rate: float
