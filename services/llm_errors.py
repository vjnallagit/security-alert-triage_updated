"""Exceptions for the LLM layer. Messages are safe to show; original causes are chained for developers."""

from typing import Any


class LLMError(Exception):
    """Base class. `trace` (when set) holds the sanitized execution trace up to the failure."""

    trace: Any = None


class LLMConfigurationError(LLMError):
    """The configuration is invalid or incomplete."""


class LLMDependencyUnavailableError(LLMError):
    """An optional LangChain/Ollama Python package is not installed."""


class LLMConnectionError(LLMError):
    """The local Ollama server could not be reached or timed out."""


class LLMModelError(LLMError):
    """The chat model failed for a reason other than connectivity."""


class UnknownToolError(LLMError):
    """The model asked for a tool that is not in the four-tool registry."""


class ToolArgumentValidationError(LLMError):
    """The model supplied arguments that do not match the tool's input schema."""


class InsufficientToolEvidenceError(LLMError):
    """Fewer than the required number of distinct, relevant, successful evidence tools were used."""


class GroundingValidationError(LLMError):
    """The final recommendation cites evidence that the tool results do not support."""


class LLMOutputValidationError(LLMError):
    """The final answer could not be parsed into the structured recommendation schema."""


class LLMGuardrailViolation(LLMError):
    """The final answer breaks a deterministic safety rule."""


class LLMIterationLimitError(LLMError):
    """The bounded tool loop ended without a valid final answer."""
