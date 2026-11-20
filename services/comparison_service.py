"""Runs all six agents on one scenario and reports the results side by side.

This service contains NO decision logic. It builds agents, hands each the same SecurityScenario, records what
they return and isolates failures. Every priority and decision comes from a real agent run.

State policy for the main comparison (so repeated clicks are repeatable and fair):
  * Simple Reflex, Goal-Based and Utility-Based: a fresh instance per run (the default Utility profile).
  * Model-Based Reflex: a fresh instance per run, so state never accumulates across clicks.
  * Learning: reads the committed seed experience read-only (`decide` never writes).
  * LLM: the real Ollama adapter in LIVE mode. A MOCK model exists only for tests/development and must be
    INJECTED through `agent_factories` (see services/llm_runtime.py); this module never loads one itself.
    A live failure is reported as an error; it never falls back to a mock.

The Model-Based and Learning demonstrations are separate, explicitly stateful helpers. They never touch the
main comparison and the Learning demo only ever writes to a temporary copy of the seed.
"""

import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents import (
    GoalBasedAgent,
    LearningAgent,
    LLMAgent,
    ModelBasedReflexAgent,
    SimpleReflexAgent,
    UtilityBasedAgent,
)
from agents.base import BaseAgent
from config.llm_config import OllamaConfig, load_ollama_config
from models.comparison_models import (
    AGENT_DISPLAY_NAMES,
    AGENT_MECHANISMS,
    AGENT_ORDER,
    SPECIAL_DETAIL_KEYS,
    AgentComparisonEntry,
    AgentComparisonResult,
    AgentFailure,
    AgentStatus,
    FeedbackStep,
    ModelBasedObservation,
    scenario_fingerprint,
)
from models.learning_models import FeedbackOutcome
from models.llm_models import RuntimeMode
from models.schemas import AgentDecision, AgentType, SecurityScenario
from services.experience_repository import DEFAULT_EXPERIENCE_PATH, ExperienceRepository
from services.llm_errors import LLMConfigurationError, LLMConnectionError, LLMError
from services.simple_llm_agent import run_llm_triage

logger = logging.getLogger(__name__)

AgentFactory = Callable[[SecurityScenario], BaseAgent]

LLM_UNAVAILABLE = "LLM Agent unavailable in the selected runtime mode."
_SENSITIVE_TEXT = (
    (re.compile(r"Traceback.*", re.DOTALL | re.IGNORECASE), "[details hidden]"),
    (re.compile(r"File \"[^\"]*\"[^\n]*"), "[details hidden]"),
    (re.compile(r"[A-Za-z]:[\\/][^\s\"']*"), "[path hidden]"),
    (re.compile(r"(?:/Users|/home|/var|/etc|/tmp)/[^\s\"']*"), "[path hidden]"),
    (re.compile(r"(?i)\b\w*(?:api[_-]?key|token|password|secret|credential)\w*\s*[=:]\s*\S+"), "[secret hidden]"),
)


def redact_sensitive_text(text: str) -> str:
    """Remove stack traces, file paths and secret-looking assignments from text that may be shown to a user."""
    for pattern, replacement in _SENSITIVE_TEXT:
        text = pattern.sub(replacement, text)
    return text.strip()


def safe_error_message(agent_type: AgentType, error: Exception) -> str:
    """A short, user-safe explanation. The technical detail stays in the developer log only."""
    name = AGENT_DISPLAY_NAMES[agent_type]
    if agent_type is AgentType.LLM:
        if isinstance(error, LLMConnectionError):
            detail = "Live Ollama is unavailable (the local server could not be reached, or it timed out)."
        elif isinstance(error, LLMError):
            detail = redact_sensitive_text(str(error))[:240]
        else:
            detail = f"({type(error).__name__})"
        return f"{LLM_UNAVAILABLE} {detail}".strip()
    return f"{name} could not complete ({type(error).__name__})."


def special_details_for(agent_type: AgentType, decision: AgentDecision) -> dict[str, Any]:
    """The small subset of metadata an agent's detail panel needs (copied, never recomputed)."""
    return {key: decision.metadata[key] for key in SPECIAL_DETAIL_KEYS[agent_type] if key in decision.metadata}


def live_ollama_status(environ: Mapping[str, str] | None = None) -> tuple[bool, str]:
    """Is live mode explicitly enabled and validly configured? Reads settings only; never contacts Ollama."""
    env = os.environ if environ is None else environ
    if env.get("RUN_OLLAMA_INTEGRATION", "").strip().lower() != "true":
        return False, "Live Ollama is not enabled (RUN_OLLAMA_INTEGRATION is not 'true')."
    try:
        load_ollama_config(env, integration_enabled=True)
    except LLMConfigurationError as exc:
        return False, redact_sensitive_text(str(exc))[:200]
    return True, "Live Ollama is enabled and configured (the server is not contacted until you run)."


class ScenarioModifiedError(RuntimeError):
    """An agent changed the scenario it was given, which would make the comparison unfair."""


class LiveRequestInProgressError(LLMError):
    """A live Ollama request is already running; requests are never run concurrently."""

class SimpleLLMComparisonAgent:
    """Adapter exposing the working Simple LLM Agent as an AgentDecision."""

    def decide(
        self,
        scenario: SecurityScenario,
    ) -> AgentDecision:
        result = run_llm_triage(
            scenario.scenario_id
        )

        evidence = []

        for item in result.get(
            "tool_results",
            [],
        ):
            if not isinstance(item, dict):
                continue

            tool_name = str(
                item.get(
                    "tool_name",
                    "unknown_tool",
                )
            )

            payload = item.get(
                "payload",
                {},
            )

            if not isinstance(payload, dict):
                payload = {}

            status = str(
                payload.get(
                    "status",
                    "UNKNOWN",
                )
            )

            summary = str(
                payload.get(
                    "summary",
                    "",
                )
            ).strip()

            source_ids = (
                payload.get(
                    "source_record_ids",
                    [],
                )
                or []
            )

            evidence_line = (
                f"{tool_name} returned {status}"
            )

            if summary:
                evidence_line += (
                    f": {summary}"
                )

            if source_ids:
                evidence_line += (
                    " Sources: "
                    + ", ".join(
                        str(source_id)
                        for source_id
                        in source_ids
                    )
                )

            evidence.append(
                evidence_line
            )

        reasoning = (
            f"{result['likely_interpretation']} "
            f"Uncertainty: {result['uncertainty']}"
        ).strip()

        metadata = {
            "runtime_mode": RuntimeMode.OLLAMA.value,
            "model_name": result["model"],
            "tools_called": list(
                result["tools_called"]
            ),
            "source_record_ids": list(
                result["source_record_ids"]
            ),
            "tool_results": list(
                result["tool_results"]
            ),
            "grounding_passed": bool(
                result["grounding_passed"]
            ),
            "guardrails_passed": bool(
                result["guardrails_passed"]
            ),
        }

        return AgentDecision.model_validate(
            {
                "agent_type": AgentType.LLM,
                "agent_name": AGENT_DISPLAY_NAMES[AgentType.LLM],
                "priority": result["priority"],
                "decision": result["decision"],
                "reasoning": reasoning,
                "evidence": evidence,
                "uncertainty": result["uncertainty"],
                "likely_interpretation": result[
                    "likely_interpretation"
                ],
                "recommended_actions": result[
                    "recommended_actions"
                ],
                "audit_note": result["audit_note"],
                "metadata": metadata,
            }
        )

# One live model request at a time across the whole process (CPU-only inference is slow and not concurrent-safe).
_LIVE_LLM_LOCK = threading.Lock()


class AgentComparisonService:
    def __init__(
        self,
        llm_mode: RuntimeMode | None,
        *,
        experience_path: Path = DEFAULT_EXPERIENCE_PATH,
        ollama_config: OllamaConfig | None = None,
        agent_factories: Mapping[AgentType, AgentFactory] | None = None,
        llm_unavailable_reason: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """`llm_mode` is explicit: OLLAMA (live), MOCK (the caller must inject a model factory), or None (disabled)."""
        self._llm_mode = llm_mode
        self._llm_unavailable_reason = llm_unavailable_reason
        self._experience_path = Path(experience_path)
        self._ollama_config = ollama_config
        self._clock = clock
        self._factories: dict[AgentType, AgentFactory] = {
            AgentType.SIMPLE_REFLEX: lambda _scenario: SimpleReflexAgent(),
            AgentType.MODEL_BASED_REFLEX: lambda _scenario: ModelBasedReflexAgent(),  # fresh every run
            AgentType.GOAL_BASED: lambda _scenario: GoalBasedAgent(),
            AgentType.UTILITY_BASED: lambda _scenario: UtilityBasedAgent(),
            AgentType.LEARNING: lambda _scenario: LearningAgent(repository=ExperienceRepository(self._experience_path)),
            AgentType.LLM: self._build_simple_llm_agent,
            **(agent_factories or {}),
        }

    @property
    def llm_mode(self) -> RuntimeMode | None:
        return self._llm_mode

    def compare(self, scenario: SecurityScenario) -> AgentComparisonResult:
        fingerprint = scenario_fingerprint(scenario)
        started_at = self._clock()
        entries = tuple(self._run_agent(agent_type, scenario, fingerprint) for agent_type in AGENT_ORDER)
        return self._result(scenario, fingerprint, started_at, entries)

    def rerun_agent(self, result: AgentComparisonResult, agent_type: AgentType) -> AgentComparisonResult:
        """Re-run ONE agent on the same scenario, keeping every other entry exactly as it was."""
        if scenario_fingerprint(result.scenario) != result.scenario_fingerprint:
            raise ValueError("the stored scenario no longer matches its fingerprint")
        if result.llm_runtime_mode is not self._llm_mode:
            raise ValueError("the stored result was produced under a different LLM runtime")
        fresh = self._run_agent(agent_type, result.scenario, result.scenario_fingerprint)
        entries = tuple(fresh if e.agent_type is agent_type else e for e in result.results)
        return self._result(result.scenario, result.scenario_fingerprint, result.started_at, entries)

    def _result(
        self, scenario: SecurityScenario, fingerprint: str, started_at: datetime,
        entries: tuple[AgentComparisonEntry, ...],
    ) -> AgentComparisonResult:
        errors = tuple(
            AgentFailure(agent_type=e.agent_type, agent_name=e.agent_name, message=e.error_message or "")
            for e in entries if e.status is AgentStatus.ERROR
        )
        return AgentComparisonResult(
            scenario=scenario, scenario_fingerprint=fingerprint, started_at=started_at,
            completed_at=self._clock(), results=entries, errors=errors, llm_runtime_mode=self._llm_mode,
        )

    # --- one agent -------------------------------------------------------------------------

    def _run_agent(self, agent_type: AgentType, scenario: SecurityScenario, fingerprint: str) -> AgentComparisonEntry:
        name, mechanism = AGENT_DISPLAY_NAMES[agent_type], AGENT_MECHANISMS[agent_type]
        started = time.perf_counter()
        guard = _LIVE_LLM_LOCK if agent_type is AgentType.LLM and self._llm_mode is RuntimeMode.OLLAMA else None
        acquired = False
        try:
            if guard is not None:
                acquired = guard.acquire(blocking=False)
                if not acquired:
                    raise LiveRequestInProgressError("Another live Ollama request is already running; try again shortly.")
            agent = self._factories[agent_type](scenario)
            # Each agent gets its own equal copy, so a misbehaving agent can never alter what the others see.
            given = scenario.model_copy(deep=True)
            if scenario_fingerprint(given) != fingerprint:
                raise ScenarioModifiedError("the scenario copy differs from the original")
            decision = agent.decide(given)
            if scenario_fingerprint(given) != fingerprint:
                raise ScenarioModifiedError("the agent modified the scenario it was given")
            elapsed_ms = (time.perf_counter() - started) * 1000
            return self._entry_from_decision(agent_type, decision, fingerprint, elapsed_ms)
        except Exception as exc:  # one agent failing must never discard the other agents' results
            logger.warning("%s failed: %s", name, type(exc).__name__)
            logger.debug("Details for the %s failure", name, exc_info=True)
            return AgentComparisonEntry(
                agent_type=agent_type, agent_name=name, mechanism=mechanism, status=AgentStatus.ERROR,
                input_fingerprint=fingerprint, execution_time_ms=(time.perf_counter() - started) * 1000,
                llm_runtime_mode=self._llm_mode if agent_type is AgentType.LLM else None,
                error_message=safe_error_message(agent_type, exc),
            )
        finally:
            if acquired and guard is not None:
                guard.release()

    def _entry_from_decision(
        self, agent_type: AgentType, decision: AgentDecision, fingerprint: str, elapsed_ms: float
    ) -> AgentComparisonEntry:
        if decision.agent_type is not agent_type:
            raise ValueError(f"expected a {agent_type.value} decision but received {decision.agent_type.value}")
        runtime_mode = None
        if agent_type is AgentType.LLM:
            reported = decision.metadata.get("runtime_mode")
            requested = self._llm_mode.value if self._llm_mode else None
            if reported != requested:  # a result may never be presented under the wrong runtime
                raise LLMConfigurationError(
                    f"The LLM result reported runtime mode {reported!r}, not the requested {requested!r}.")
            runtime_mode = RuntimeMode(reported)
        return AgentComparisonEntry(
            agent_type=agent_type, agent_name=AGENT_DISPLAY_NAMES[agent_type], mechanism=AGENT_MECHANISMS[agent_type],
            status=AgentStatus.SUCCESS, input_fingerprint=fingerprint, execution_time_ms=elapsed_ms,
            priority=decision.priority, decision=decision.decision, reasoning_summary=decision.reasoning,
            evidence=tuple(decision.evidence), uncertainty=decision.uncertainty,
            likely_interpretation=decision.likely_interpretation,
            recommended_actions=tuple(decision.recommended_actions), audit_note=decision.audit_note,
            llm_runtime_mode=runtime_mode, special_details=special_details_for(agent_type, decision),
            decision_record=decision,
        )
    def _build_simple_llm_agent(
        self,
        scenario: SecurityScenario,
    ) -> Any:
        """Build the working local-Ollama comparison adapter."""

        del scenario

        if self._llm_mode is None:
            raise LLMConfigurationError(
                self._llm_unavailable_reason
                or "The LLM runtime is disabled."
            )

        if self._llm_mode is RuntimeMode.MOCK:
            raise LLMConfigurationError(
                "The MOCK runtime requires an explicitly injected "
                "development agent factory."
            )

        config = (
            self._ollama_config
            or load_ollama_config()
        )

        if not config.integration_enabled:
            raise LLMConfigurationError(
                "Live Ollama requires "
                "RUN_OLLAMA_INTEGRATION=true."
            )

        return SimpleLLMComparisonAgent()
    # --- LLM runtime -----------------------------------------------------------------------

    def _build_llm_agent(self, scenario: SecurityScenario) -> BaseAgent:
        if self._llm_mode is None:
            raise LLMConfigurationError(self._llm_unavailable_reason or "The LLM runtime is disabled.")
        if self._llm_mode is RuntimeMode.MOCK:
            raise LLMConfigurationError(
                "The MOCK runtime has no model; a development mock must be injected explicitly (tests/development only).")
        # Live mode: no fallback. Any failure surfaces as an error entry for the LLM Agent.
        config = self._ollama_config or load_ollama_config()
        if not config.integration_enabled:
            raise LLMConfigurationError("Live Ollama requires RUN_OLLAMA_INTEGRATION=true and a valid configuration.")
        return LLMAgent(config=config)


# ------------------------------------------------------------------------------------------------
# Demonstrations (explicitly stateful, isolated from the main comparison)
# ------------------------------------------------------------------------------------------------


class ModelBasedMemoryDemo:
    """ONE Model-Based agent reused on purpose, to show how internal state changes the decision."""

    def __init__(self) -> None:
        self._agent = ModelBasedReflexAgent()
        self._observations: list[ModelBasedObservation] = []

    @property
    def observations(self) -> tuple[ModelBasedObservation, ...]:
        return tuple(self._observations)

    @property
    def processed_scenario_count(self) -> int:
        return self._agent.get_state_snapshot().processed_scenario_count

    def observe(self, scenario: SecurityScenario) -> ModelBasedObservation:
        decision = self._agent.decide(scenario)
        summary = decision.metadata["state_summary"]
        observation = ModelBasedObservation(
            observation_number=len(self._observations) + 1, priority=decision.priority, decision=decision.decision,
            matched_rule_id=decision.metadata["matched_rule_id"], account_observations=summary["account_count"],
            previous_decision=summary["previous_decision"],
        )
        self._observations.append(observation)
        return observation

    def reset(self) -> None:
        self._agent = ModelBasedReflexAgent()
        self._observations.clear()


class LearningFeedbackDemo:
    """A Learning agent on a TEMPORARY copy of the seed experience. The committed seed is never written."""

    def __init__(self, seed_path: Path = DEFAULT_EXPERIENCE_PATH) -> None:
        self._seed = Path(seed_path)
        self._directory = tempfile.TemporaryDirectory(prefix="triage-learning-demo-", ignore_cleanup_errors=True)
        store = Path(self._directory.name) / "experience_demo.json"
        if store.resolve() == self._seed.resolve():
            raise RuntimeError("the demonstration store must not be the committed seed")
        self._repository = ExperienceRepository(self._seed).copy_to(store)
        self._agent = LearningAgent(repository=self._repository)
        self._steps: list[FeedbackStep] = []

    @property
    def store_path(self) -> Path:
        return self._repository.path

    @property
    def steps(self) -> tuple[FeedbackStep, ...]:
        return tuple(self._steps)

    @property
    def feedback_count(self) -> int:
        return len(self._steps)

    def decide(self, scenario: SecurityScenario) -> AgentDecision:
        return self._agent.decide(scenario)

    def record_feedback(self, scenario: SecurityScenario, outcome: FeedbackOutcome) -> FeedbackStep:
        """Record the outcome of the action the Learning agent currently recommends, then decide again."""
        before_decision = self._agent.decide(scenario).decision
        update = self._agent.record_feedback(scenario, before_decision, outcome)
        step = FeedbackStep(
            scenario_id=scenario.scenario_id, pattern=update.pattern, action=update.action,
            outcome=outcome.value, decision_before=before_decision,
            decision_after=self._agent.decide(scenario).decision,
            before_success=update.before_success, before_failure=update.before_failure,
            before_smoothed_rate=update.before_smoothed_rate, after_success=update.after_success,
            after_failure=update.after_failure, after_smoothed_rate=update.after_smoothed_rate,
        )
        self._steps.append(step)
        return step

    def reset(self) -> None:
        """Restore the temporary store to the seed's content."""
        self._repository.reset_from(self._seed)
        self._steps.clear()

    def close(self) -> None:
        self._directory.cleanup()
