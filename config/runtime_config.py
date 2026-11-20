"""Which LLM runtime the application uses, and the rules that keep MOCK out of a submission.

    LIVE      the real local Ollama model (the intended submission runtime)
    MOCK      the scripted development model: only with ENABLE_MOCK_LLM=true and never in SUBMISSION_MODE
    DISABLED  no LLM Agent (the five classical agents still run)

Nothing here contacts Ollama or imports a mock; it only validates settings. The runtime is never inferred
from network availability, and a LIVE failure never turns into MOCK.
"""

import os
from collections.abc import Mapping
from enum import StrEnum
from typing import Self

from pydantic import ValidationError, model_validator

from config.llm_config import FALSE_VALUES, TRUE_VALUES, load_ollama_config
from models.goal_models import FrozenModel
from models.llm_models import RuntimeMode
from services.llm_errors import LLMConfigurationError


class LLMRuntimeMode(StrEnum):
    LIVE = "LIVE"
    MOCK = "MOCK"
    DISABLED = "DISABLED"


class RuntimeConfig(FrozenModel):
    # The committed defaults enable nothing. (The environment loader defaults to LIVE, which then needs
    # RUN_OLLAMA_INTEGRATION=true; so an unset environment also starts nothing.)
    mode: LLMRuntimeMode = LLMRuntimeMode.DISABLED
    enable_mock: bool = False
    run_ollama_integration: bool = False
    submission_mode: bool = False

    @model_validator(mode="after")
    def _runtime_rules(self) -> Self:
        if self.submission_mode and self.mode is not LLMRuntimeMode.LIVE:
            raise ValueError("SUBMISSION_MODE=true requires LLM_RUNTIME_MODE=LIVE; MOCK and DISABLED are rejected")
        if self.mode is LLMRuntimeMode.MOCK and not self.enable_mock:
            raise ValueError("LLM_RUNTIME_MODE=MOCK requires ENABLE_MOCK_LLM=true")
        if self.mode is LLMRuntimeMode.LIVE and not self.run_ollama_integration:
            raise ValueError("LLM_RUNTIME_MODE=LIVE requires RUN_OLLAMA_INTEGRATION=true")
        return self

    @property
    def result_mode(self) -> RuntimeMode | None:
        """The label results carry: OLLAMA for LIVE, MOCK for MOCK, None when the LLM is disabled."""
        return {LLMRuntimeMode.LIVE: RuntimeMode.OLLAMA, LLMRuntimeMode.MOCK: RuntimeMode.MOCK}.get(self.mode)

    @property
    def mock_allowed(self) -> bool:
        return self.enable_mock and not self.submission_mode


def _flag(env: Mapping[str, str], name: str) -> bool:
    raw = (env.get(name) or "").strip().lower()
    if not raw or raw in FALSE_VALUES:
        return False
    if raw in TRUE_VALUES:
        return True
    raise LLMConfigurationError(f"{name} must be true or false")


def _mode(env: Mapping[str, str]) -> LLMRuntimeMode:
    raw = (env.get("LLM_RUNTIME_MODE") or "").strip().upper()
    if not raw:
        return LLMRuntimeMode.LIVE
    try:
        return LLMRuntimeMode(raw)
    except ValueError:
        allowed = ", ".join(mode.value for mode in LLMRuntimeMode)
        raise LLMConfigurationError(f"LLM_RUNTIME_MODE must be one of: {allowed}") from None


def load_runtime_config(environ: Mapping[str, str] | None = None, **overrides: object) -> RuntimeConfig:
    """Read LLM_RUNTIME_MODE, ENABLE_MOCK_LLM, RUN_OLLAMA_INTEGRATION and SUBMISSION_MODE, then validate them.

    A LIVE runtime also needs a valid Ollama configuration (including a non-empty OLLAMA_MODEL).
    """
    env = os.environ if environ is None else environ
    values: dict[str, object] = {
        "mode": _mode(env),
        "enable_mock": _flag(env, "ENABLE_MOCK_LLM"),
        "run_ollama_integration": _flag(env, "RUN_OLLAMA_INTEGRATION"),
        "submission_mode": _flag(env, "SUBMISSION_MODE"),
    }
    values.update(overrides)
    try:
        config = RuntimeConfig(**values)
    except ValidationError as exc:
        problems = "; ".join(
            error["msg"].removeprefix("Value error, ") for error in exc.errors(include_input=False, include_url=False))
        raise LLMConfigurationError(problems) from exc
    if config.mode is LLMRuntimeMode.LIVE:
        load_ollama_config(env, integration_enabled=True)  # raises if the model or base URL is invalid
    return config
