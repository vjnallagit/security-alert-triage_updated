"""Decides which LLM runtimes the application may offer, and builds the comparison service for one of them.

This is the ONLY production module allowed to load the development mock (devtools/mock_llm_runtime.py), and
only through `build_comparison_service(RuntimeMode.MOCK, ...)` when ENABLE_MOCK_LLM=true and SUBMISSION_MODE is
not true. Nothing is chosen automatically: a failed or unavailable LIVE runtime is reported as an error and
never replaced by the mock. No network call is made here.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module

from config.llm_config import load_ollama_config
from config.runtime_config import LLMRuntimeMode, RuntimeConfig, load_runtime_config
from models.llm_models import RuntimeMode
from models.schemas import AgentType
from services.comparison_service import AgentComparisonService, live_ollama_status, redact_sensitive_text
from services.llm_errors import LLMConfigurationError

LIVE_UNAVAILABLE_TEMPLATE = (
    "Live Ollama is unavailable. Verify that Ollama is running locally, that {model} is installed, "
    "and that the environment configuration is correct."
)
_MOCK_MODULE = "devtools.mock_llm_runtime"  # loaded by name, only behind the gate below


@dataclass(frozen=True)
class LLMRuntimeStatus:
    """What the UI may offer. `config_error` is a safe message when the environment is invalid."""

    submission_mode: bool
    available: tuple[RuntimeMode, ...]
    default: RuntimeMode | None
    model_name: str | None
    notices: tuple[str, ...] = ()
    config_error: str | None = None


def live_unavailable_message(model_name: str | None) -> str:
    return LIVE_UNAVAILABLE_TEMPLATE.format(model=model_name or "the configured model")


def inspect_runtime(environ: Mapping[str, str] | None = None) -> LLMRuntimeStatus:
    """Read the environment into a safe, displayable status. Never contacts Ollama or loads a mock."""
    env = os.environ if environ is None else environ
    try:
        runtime = load_runtime_config(env)
    except LLMConfigurationError as exc:
        return LLMRuntimeStatus(
            submission_mode=(env.get("SUBMISSION_MODE") or "").strip().lower() in {"1", "true", "yes", "on"},
            available=(), default=None, model_name=None, config_error=redact_sensitive_text(str(exc))[:300])
    model_name = (env.get("OLLAMA_MODEL") or "").strip() or None
    if runtime.mode is LLMRuntimeMode.DISABLED:
        return LLMRuntimeStatus(runtime.submission_mode, (), None, model_name,
                                notices=("The LLM runtime is disabled (LLM_RUNTIME_MODE=DISABLED).",))
    available: list[RuntimeMode] = []
    notices: list[str] = []
    if runtime.mock_allowed:
        available.append(RuntimeMode.MOCK)
    live_ok, live_reason = live_ollama_status(env)
    if live_ok:
        available.append(RuntimeMode.OLLAMA)
    elif not runtime.submission_mode:
        notices.append(live_reason)
    return LLMRuntimeStatus(runtime.submission_mode, tuple(available), runtime.result_mode, model_name, tuple(notices))


def build_comparison_service(
    mode: RuntimeMode | None, environ: Mapping[str, str] | None = None, *, reason: str | None = None,
) -> AgentComparisonService:
    """The comparison service for one runtime. MOCK is built only if the explicit development gate passes."""
    env = os.environ if environ is None else environ
    if mode is None:
        return AgentComparisonService(llm_mode=None, llm_unavailable_reason=reason)
    if mode is RuntimeMode.OLLAMA:
        return AgentComparisonService(llm_mode=RuntimeMode.OLLAMA, ollama_config=load_ollama_config(env))
    runtime: RuntimeConfig = load_runtime_config(env, mode=LLMRuntimeMode.MOCK)
    if not runtime.mock_allowed:
        raise LLMConfigurationError("The MOCK runtime is not allowed (it needs ENABLE_MOCK_LLM=true and not SUBMISSION_MODE).")
    factory = import_module(_MOCK_MODULE).mock_llm_agent_factory
    return AgentComparisonService(llm_mode=RuntimeMode.MOCK, agent_factories={AgentType.LLM: factory})
