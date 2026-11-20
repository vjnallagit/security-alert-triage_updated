"""Shared test infrastructure.

Learning tests must never change the committed data/experience.json, so a session guard
records its hash before the first test and verifies it after the last one.
"""

import hashlib
from pathlib import Path

import pytest
from services.data_loader import DATA_DIR
from services.experience_repository import ExperienceRepository

COMMITTED_EXPERIENCE = DATA_DIR / "experience.json"


def _digest() -> str:
    return hashlib.sha256(COMMITTED_EXPERIENCE.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def committed_experience_is_never_modified():
    before = _digest()
    yield
    assert _digest() == before, "a test modified the committed data/experience.json"


@pytest.fixture
def seed_repository(tmp_path: Path) -> ExperienceRepository:
    """A temporary, writable copy of the committed seed experience."""
    return ExperienceRepository(COMMITTED_EXPERIENCE).copy_to(tmp_path / "experience.json")


# Every runtime-related variable, so a test never inherits real local settings.
RUNTIME_ENVIRONMENT = (
    "LLM_RUNTIME_MODE", "ENABLE_MOCK_LLM", "RUN_OLLAMA_INTEGRATION", "SUBMISSION_MODE", "OLLAMA_BASE_URL",
    "OLLAMA_MODEL", "OLLAMA_TEMPERATURE", "OLLAMA_REQUEST_TIMEOUT_SECONDS", "OLLAMA_MAX_TOOL_ITERATIONS",
    "OLLAMA_MINIMUM_TOOL_CALLS", "OLLAMA_VALIDATE_MODEL_ON_INIT", "OLLAMA_ALLOW_REMOTE_HOST", "OLLAMA_DEBUG",
)


@pytest.fixture
def clean_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in RUNTIME_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def dev_mock_environment(clean_runtime_environment: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """The offline development settings: MOCK runtime, explicitly enabled, no integration, no submission."""
    clean_runtime_environment.setenv("LLM_RUNTIME_MODE", "MOCK")
    clean_runtime_environment.setenv("ENABLE_MOCK_LLM", "true")
    return clean_runtime_environment
