import socket

import pytest
from config.llm_config import DEFAULT_BASE_URL, OllamaConfig, load_ollama_config
from pydantic import ValidationError
from services.llm_errors import LLMConfigurationError


def test_lc01_default_base_url_is_localhost():
    config = OllamaConfig()
    assert config.base_url == "http://127.0.0.1:11434" == DEFAULT_BASE_URL
    assert load_ollama_config({}).base_url == DEFAULT_BASE_URL


@pytest.mark.parametrize("url", ["ftp://127.0.0.1:11434", "127.0.0.1:11434", "file:///etc/hosts", "ws://localhost", "", "http://"])
def test_lc02_invalid_base_url_scheme_is_rejected(url):
    with pytest.raises(ValidationError):
        OllamaConfig(base_url=url)
    with pytest.raises(LLMConfigurationError):
        load_ollama_config({"OLLAMA_BASE_URL": url or " http:// "})


@pytest.mark.parametrize("url", ["http://user:secret@127.0.0.1:11434", "http://:pw@localhost:11434"])
def test_base_url_with_embedded_credentials_is_rejected(url):
    with pytest.raises(ValidationError, match="credentials"):
        OllamaConfig(base_url=url)


@pytest.mark.parametrize("url", ["http://192.168.1.50:11434", "https://example.com", "http://10.0.0.5:11434"])
def test_a_remote_host_is_rejected_unless_explicitly_allowed(url):
    with pytest.raises(ValidationError, match="localhost"):
        OllamaConfig(base_url=url)
    assert OllamaConfig(base_url=url, allow_remote_host=True).base_url == url


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434/", "https://localhost", "http://[::1]:11434"])
def test_loopback_hosts_are_accepted(url):
    assert OllamaConfig(base_url=url).base_url == url.rstrip("/")


def test_lc03_live_integration_requires_a_model_name():
    with pytest.raises(ValidationError, match="model name is required"):
        OllamaConfig(integration_enabled=True)
    with pytest.raises(LLMConfigurationError, match="model name is required"):
        load_ollama_config({"RUN_OLLAMA_INTEGRATION": "true"})
    assert OllamaConfig(integration_enabled=True, model="some-model:tag").model == "some-model:tag"


def test_lc04_mock_operation_needs_no_live_ollama_or_model():
    config = load_ollama_config({})
    assert config.model == "" and config.integration_enabled is False
    with pytest.raises(LLMConfigurationError, match="OLLAMA_MODEL is not set"):
        config.require_model()
    assert load_ollama_config({}, model="injected-for-mock-tests").require_model() == "injected-for-mock-tests"


@pytest.mark.parametrize("temperature, valid", [(0.0, True), (1.0, True), (2.0, True), (-0.1, False), (2.1, False),
                                                  (float("nan"), False), (float("inf"), False)])
def test_lc05_temperature_boundaries(temperature, valid):
    if valid:
        assert OllamaConfig(temperature=temperature).temperature == temperature
    else:
        with pytest.raises(ValidationError):
            OllamaConfig(temperature=temperature)


@pytest.mark.parametrize("timeout", [0, -5, 0.0, float("nan")])
def test_lc06_timeout_must_be_positive(timeout):
    with pytest.raises(ValidationError):
        OllamaConfig(request_timeout_seconds=timeout)
    assert OllamaConfig(request_timeout_seconds=0.5).request_timeout_seconds == 0.5


@pytest.mark.parametrize("iterations, valid", [(1, False), (2, True), (8, True), (20, True), (21, False), (0, False)])
def test_lc07_maximum_iterations_are_bounded(iterations, valid):
    if valid:
        assert OllamaConfig(max_tool_iterations=iterations).max_tool_iterations == iterations
    else:
        with pytest.raises(ValidationError):
            OllamaConfig(max_tool_iterations=iterations)


def test_lc08_minimum_tool_calls_is_at_least_two_and_within_the_iteration_limit():
    for bad in (0, 1, -1):
        with pytest.raises(ValidationError):
            OllamaConfig(minimum_tool_calls=bad)
    with pytest.raises(ValidationError, match="must not exceed"):
        OllamaConfig(minimum_tool_calls=5, max_tool_iterations=4)
    assert OllamaConfig(minimum_tool_calls=4, max_tool_iterations=4)


def test_lc09_integration_defaults_to_false_and_other_defaults_match_the_documented_ones():
    config = OllamaConfig()
    assert config.integration_enabled is False and config.validate_model_on_init is False
    assert (config.temperature, config.request_timeout_seconds, config.max_tool_iterations,
            config.minimum_tool_calls) == (0.0, 300.0, 8, 2)  # 300 s: CPU-only live inference is slow
    assert config.allow_remote_host is False


def test_lc10_configuration_exposes_no_secrets_in_repr_or_summary():
    env = {"OLLAMA_MODEL": "m:1", "OLLAMA_API_KEY": "super-secret", "HTTP_PROXY": "http://user:pw@proxy:8080",
           "PATH": "C:\\private\\path"}
    config = load_ollama_config(env)
    text = repr(config) + str(config.safe_summary()) + config.model_dump_json()
    for secret in ("super-secret", "user:pw", "proxy", "C:\\private"):
        assert secret not in text
    assert "OLLAMA_API_KEY" not in text

    assert set(config.safe_summary()) == {
        "base_url",
        "model",
        "temperature",
        "request_timeout_seconds",
        "tool_num_predict",
        "decision_num_predict",
        "keep_alive",
        "max_tool_iterations",
        "minimum_tool_calls",
        "integration_enabled",
    }


def test_environment_variables_are_read_with_documented_names():
    config = load_ollama_config({
        "OLLAMA_BASE_URL": "http://localhost:11500", "OLLAMA_MODEL": "m:tag", "OLLAMA_TEMPERATURE": "0.5",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS": "30", "OLLAMA_MAX_TOOL_ITERATIONS": "6", "OLLAMA_MINIMUM_TOOL_CALLS": "3",
        "RUN_OLLAMA_INTEGRATION": "TRUE", "OLLAMA_VALIDATE_MODEL_ON_INIT": "yes",
    })
    assert (config.base_url, config.model, config.temperature, config.request_timeout_seconds,
            config.max_tool_iterations, config.minimum_tool_calls) == ("http://localhost:11500", "m:tag", 0.5, 30.0, 6, 3)
    assert config.integration_enabled is True and config.validate_model_on_init is True


def test_blank_environment_values_fall_back_to_defaults():
    assert load_ollama_config({"OLLAMA_BASE_URL": "  ", "OLLAMA_TEMPERATURE": "", "RUN_OLLAMA_INTEGRATION": " "}) == OllamaConfig()


@pytest.mark.parametrize("name, value", [("OLLAMA_TEMPERATURE", "warm"), ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "soon"),
                                         ("OLLAMA_MAX_TOOL_ITERATIONS", "many"), ("OLLAMA_MINIMUM_TOOL_CALLS", "2.5"),
                                         ("RUN_OLLAMA_INTEGRATION", "maybe")])
def test_malformed_environment_values_raise_a_clear_configuration_error(name, value):
    with pytest.raises(LLMConfigurationError):
        load_ollama_config({name: value})


def test_invalid_values_from_the_environment_are_reported_as_configuration_errors():
    with pytest.raises(LLMConfigurationError, match="max_tool_iterations"):
        load_ollama_config({"OLLAMA_MAX_TOOL_ITERATIONS": "99"})
    with pytest.raises(LLMConfigurationError, match="request_timeout_seconds"):
        load_ollama_config({"OLLAMA_REQUEST_TIMEOUT_SECONDS": "-1"})


def test_overrides_take_precedence_over_the_environment():
    assert load_ollama_config({"OLLAMA_MODEL": "from-env"}, model="from-override").model == "from-override"


def test_configuration_is_immutable():
    with pytest.raises(ValidationError):
        OllamaConfig().model = "changed"
    with pytest.raises(ValidationError):
        OllamaConfig(unexpected_option=True)


def test_loading_and_validating_configuration_never_touches_the_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    config = load_ollama_config({"OLLAMA_MODEL": "m:1", "RUN_OLLAMA_INTEGRATION": "true"})
    assert config.integration_enabled is True
