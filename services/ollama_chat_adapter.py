"""The only module that talks to a real language model.

`ToolCallingChatModel` is the small contract the orchestrator needs: bind the tools, then invoke with
messages and get an AIMessage back. `OllamaChatModelAdapter` satisfies it with ChatOllama. Tests satisfy it
with a scripted fake, so the orchestrator is identical in both cases.

Nothing here connects to Ollama when imported or constructed; the first network call happens inside
`invoke`. LangChain and the Ollama client are imported lazily for the same reason. Errors are mapped to
safe messages (no paths, no stack traces).
"""

from collections.abc import Sequence
from typing import Any, Protocol

from config.llm_config import OllamaConfig
from models.llm_models import RuntimeMode
from services.llm_errors import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMDependencyUnavailableError,
    LLMModelError,
)


class BoundToolModel(Protocol):
    def invoke(self, messages: list[Any]) -> Any:
        """Return an AIMessage (with `tool_calls` when the model wants tools)."""


class ToolCallingChatModel(Protocol):
    runtime_mode: RuntimeMode
    model_name: str

    def bind_tools(self, tools: Sequence[Any]) -> BoundToolModel:
        """Return a model that may request the given tools."""
    
    def invoke_without_tools(
        self,
        messages: list[Any],
    ) -> Any:
        """Invoke the compact final-decision model without bound tools."""


def map_model_exception(exc: Exception) -> Exception:
    """Translate a client/transport error into a safe LLM error. Never echoes raw messages or paths."""
    try:
        import httpx
        transport_errors: tuple[type[BaseException], ...] = (
            httpx.ConnectError, httpx.TimeoutException, ConnectionError, TimeoutError)
    except ImportError:  # pragma: no cover - httpx ships with the ollama client
        transport_errors = (ConnectionError, TimeoutError)
    if isinstance(exc, transport_errors):
        return LLMConnectionError(
            "Could not reach the local Ollama server (not running, wrong OLLAMA_BASE_URL, or timed out).")
    status = getattr(exc, "status_code", None)
    if status == 404:
        return LLMModelError("Ollama reports the configured model is not available; pull it or fix OLLAMA_MODEL.")
    detail = f" (HTTP {status})" if isinstance(status, int) else ""
    return LLMModelError(f"The Ollama model call failed: {type(exc).__name__}{detail}.")


class _BoundOllama:
    def __init__(self, bound: Any) -> None:
        self._bound = bound

    def invoke(self, messages: list[Any]) -> Any:
        try:
            return self._bound.invoke(messages)
        except Exception as exc:
            raise map_model_exception(exc) from exc


class OllamaChatModelAdapter:
    """ChatOllama behind the ToolCallingChatModel contract. Constructing it performs no network I/O."""

    runtime_mode = RuntimeMode.OLLAMA

    def __init__(
        self,
        config: OllamaConfig,
        chat_model_factory: Any = None,
    ) -> None:
        self.config = config
        self.model_name = config.require_model()

        self._tool_factory = (
            chat_model_factory
            or _default_tool_chat_ollama_factory
        )

        self._decision_factory = (
            chat_model_factory
            or _default_decision_chat_ollama_factory
        )

        self._tool_chat: Any = None
        self._decision_chat: Any = None


    def _tool_chat_model(self) -> Any:
        if self._tool_chat is None:
            self._tool_chat = self._tool_factory(
                self.config
            )

        return self._tool_chat

    def _chat_model(self) -> Any:
        """Backward-compatible alias for the tool-selection chat model."""
        return self._tool_chat_model()

    def _decision_chat_model(self) -> Any:
        if self._decision_chat is None:
            self._decision_chat = self._decision_factory(
                self.config
            )

        return self._decision_chat

    def bind_tools(
        self,
        tools: Sequence[Any],
    ) -> BoundToolModel:
        try:
            return _BoundOllama(
                self._tool_chat_model().bind_tools(
                    list(tools)
                )
            )
        except LLMDependencyUnavailableError:
            raise
        except Exception as exc:
            raise map_model_exception(exc) from exc
    
    def invoke_without_tools(
        self,
        messages: list[Any],
    ) -> Any:
        """Invoke the compact final-decision model without tool schemas."""

        try:
            return self._decision_chat_model().invoke(
                messages
            )
        except LLMDependencyUnavailableError:
            raise
        except Exception as exc:
            raise map_model_exception(exc) from exc

    def ask(self, prompt: str) -> str:
        """One plain prompt without tools, used by the connection check. Contacts the server."""
        try:
            reply = self._tool_chat_model().invoke(prompt)
        except LLMDependencyUnavailableError:
            raise
        except Exception as exc:
            raise map_model_exception(exc) from exc
        content = reply.content
        return content if isinstance(content, str) else str(content)


def _default_tool_chat_ollama_factory(
    config: OllamaConfig,
) -> Any:
    """Create the lazy tool-selection ChatOllama instance."""

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise LLMDependencyUnavailableError(
            "langchain-ollama is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    return ChatOllama(
        model=config.require_model(),
        base_url=config.base_url,
        temperature=config.temperature,
        reasoning=False,
        num_predict=config.tool_num_predict,
        keep_alive=config.keep_alive,
        client_kwargs={
            "timeout": config.request_timeout_seconds,
        },
        validate_model_on_init=(
            config.validate_model_on_init
        ),
    )

def _default_decision_chat_ollama_factory(
    config: OllamaConfig,
) -> Any:
    """Create the lazy compact-decision ChatOllama instance."""

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise LLMDependencyUnavailableError(
            "langchain-ollama is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    return ChatOllama(
        model=config.require_model(),
        base_url=config.base_url,
        temperature=config.temperature,
        reasoning=False,
        num_predict=config.decision_num_predict,
        keep_alive=config.keep_alive,
        client_kwargs={
            "timeout": config.request_timeout_seconds,
        },
        validate_model_on_init=(
            config.validate_model_on_init
        ),
    )
def _default_chat_ollama_factory(
    config: OllamaConfig,
) -> Any:
    """Backward-compatible alias for the tool-selection factory."""
    return _default_tool_chat_ollama_factory(
        config
    )

def list_available_models(config: OllamaConfig) -> tuple[str, ...]:
    """Names of models the local Ollama server has pulled. This DOES contact the server."""
    try:
        from ollama import Client
    except ImportError as exc:
        raise LLMDependencyUnavailableError("The ollama package is not installed.") from exc
    try:
        response = Client(host=config.base_url, timeout=config.request_timeout_seconds).list()
    except Exception as exc:
        raise map_model_exception(exc) from exc
    return tuple(sorted(str(m.model) for m in response.models if getattr(m, "model", None)))


def require_model_available(config: OllamaConfig) -> tuple[str, ...]:
    """Raise a clear error unless the configured model has been pulled. Contacts the server."""
    wanted = config.require_model()
    available = list_available_models(config)
    if wanted not in available and f"{wanted}:latest" not in available:
        raise LLMConfigurationError(
            f"Model '{wanted}' is not pulled. Available models: {', '.join(available) or '(none)'}.")
    return available
