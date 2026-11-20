import hashlib
import json
import os
from pathlib import Path

import pytest
from models.learning_models import ExperienceStats, FeedbackOutcome
from services import experience_repository as repo_module
from services.data_loader import DATA_DIR, DataFileNotFoundError, MalformedDataError
from services.experience_repository import (
    DEFAULT_EXPERIENCE_PATH,
    ExperienceRepository,
    ExperienceWriteError,
    InvalidExperienceDataError,
)

PATTERN = "privileged_login_anomaly"
INVESTIGATE = "INVESTIGATE"
ESCALATE = "ESCALATE_TO_SECURITY_ANALYST"


@pytest.fixture
def repo(tmp_path) -> ExperienceRepository:
    return ExperienceRepository.create(
        tmp_path / "experience.json",
        {
            PATTERN: {INVESTIGATE: (4, 1), ESCALATE: (3, 2)},
            "other_pattern": {INVESTIGATE: (7, 7)},
        },
    )


def records_of(repository: ExperienceRepository) -> dict:
    return json.loads(repository.path.read_text(encoding="utf-8"))["records"]


def test_er01_repository_loads_valid_data(repo):
    data = repo.load()
    assert data.simulated is True and set(data.records) == {PATTERN, "other_pattern"}
    assert repo.patterns() == (PATTERN, "other_pattern")


def test_er01_committed_seed_loads_and_contains_the_fact_derived_patterns():
    committed = ExperienceRepository(DEFAULT_EXPERIENCE_PATH)
    assert {"privileged_login_anomaly", "low_severity_limited_evidence",
            "critical_service_correlated_access", "general_security_alert"} <= set(committed.patterns())
    assert "repeated_failed_login" in committed.patterns()  # Phase 1 records are preserved


def test_er02_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(DataFileNotFoundError, match="missing.json"):
        ExperienceRepository(tmp_path / "missing.json").load()


def test_er03_malformed_json_raises_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(MalformedDataError, match="bad.json"):
        ExperienceRepository(path).load()


@pytest.mark.parametrize("document", [
    {"description": "d", "simulated": True, "records": {PATTERN: {INVESTIGATE: {"success": -1, "failure": 0}}}},
    {"description": "d", "simulated": True, "records": {PATTERN: {INVESTIGATE: {"success": 1, "failure": -3}}}},
    {"description": "d", "simulated": False, "records": {PATTERN: {INVESTIGATE: {"success": 1, "failure": 1}}}},
    {"description": "d", "simulated": True, "records": {}},
    {"description": "d", "simulated": True, "records": {PATTERN: {INVESTIGATE: {"success": 1, "failure": 1, "rate": 0.5}}}},
])
def test_er04_invalid_stored_data_is_rejected(tmp_path, document):
    path = tmp_path / "experience.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(InvalidExperienceDataError, match="experience.json"):
        ExperienceRepository(path).load()


def test_er05_lookup_returns_the_correct_pattern_and_action(repo):
    assert repo.get_stats(PATTERN, INVESTIGATE) == ExperienceStats(success=4, failure=1)
    assert repo.get_stats(PATTERN, ESCALATE) == ExperienceStats(success=3, failure=2)
    assert repo.get_pattern_history(PATTERN) == {
        INVESTIGATE: ExperienceStats(success=4, failure=1),
        ESCALATE: ExperienceStats(success=3, failure=2),
    }


def test_lookup_of_unknown_pattern_or_action_is_zero_and_does_not_create_it(repo):
    assert repo.get_stats("never_seen", INVESTIGATE) == ExperienceStats(success=0, failure=0)
    assert repo.get_stats(PATTERN, "MONITOR") == ExperienceStats(success=0, failure=0)
    assert repo.get_pattern_history("never_seen") == {}
    assert "never_seen" not in records_of(repo)


def test_er06_success_increments_only_success(repo):
    before, after = repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.SUCCESS)
    assert (before.success, before.failure, after.success, after.failure) == (4, 1, 5, 1)
    assert records_of(repo)[PATTERN][INVESTIGATE] == {"success": 5, "failure": 1}


def test_er07_failure_increments_only_failure(repo):
    before, after = repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    assert (before.success, before.failure, after.success, after.failure) == (4, 1, 4, 2)
    assert records_of(repo)[PATTERN][INVESTIGATE] == {"success": 4, "failure": 2}


def test_er08_unknown_changes_nothing_and_does_not_write(repo, monkeypatch):
    original = repo.path.read_bytes()
    monkeypatch.setattr(repo_module, "_atomic_write", lambda *a, **k: pytest.fail("UNKNOWN must not write"))
    before, after = repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.UNKNOWN)
    assert before == after
    assert repo.path.read_bytes() == original


def test_unknown_for_a_new_pattern_does_not_create_it(repo):
    repo.record_outcome("brand_new", INVESTIGATE, FeedbackOutcome.UNKNOWN)
    assert "brand_new" not in records_of(repo)


def test_er09_unrelated_records_are_logically_unchanged(repo):
    before = records_of(repo)
    repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    after = records_of(repo)
    assert after[PATTERN][ESCALATE] == before[PATTERN][ESCALATE]
    assert after["other_pattern"] == before["other_pattern"]
    changed = {(p, a) for p in after for a in after[p] if after[p][a] != before.get(p, {}).get(a)}
    assert changed == {(PATTERN, INVESTIGATE)}


def test_er09_committed_seed_records_survive_an_update_on_a_copy(seed_repository):
    committed = ExperienceRepository(DEFAULT_EXPERIENCE_PATH).load().records
    seed_repository.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    after = seed_repository.load().records
    for pattern, actions in committed.items():
        for action, counts in actions.items():
            if (pattern, action) != (PATTERN, INVESTIGATE):
                assert after[pattern][action] == counts


def test_er10_er11_update_is_persisted_and_seen_after_reloading(repo):
    repo.record_outcome(PATTERN, ESCALATE, FeedbackOutcome.SUCCESS)
    assert ExperienceRepository(repo.path).get_stats(PATTERN, ESCALATE) == ExperienceStats(success=4, failure=2)
    assert repo.get_stats(PATTERN, ESCALATE) == ExperienceStats(success=4, failure=2)


def test_new_pattern_action_is_initialised_at_zero_then_counted(repo):
    repo.record_outcome("brand_new", INVESTIGATE, FeedbackOutcome.FAILURE)
    assert records_of(repo)["brand_new"] == {INVESTIGATE: {"success": 0, "failure": 1}}


def test_er12_atomic_write_leaves_complete_json_and_no_temporary_files(repo):
    for _ in range(3):
        repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
        json.loads(repo.path.read_text(encoding="utf-8"))  # always complete, valid JSON
    assert [p.name for p in repo.path.parent.iterdir()] == ["experience.json"]


def test_er13_failure_during_write_does_not_corrupt_the_last_valid_file(repo, monkeypatch):
    original = repo.path.read_bytes()

    def failing_replace(source, destination):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(ExperienceWriteError, match="simulated disk failure"):
        repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    monkeypatch.undo()
    assert repo.path.read_bytes() == original
    assert [p.name for p in repo.path.parent.iterdir()] == ["experience.json"]  # temp file cleaned up
    assert repo.get_stats(PATTERN, INVESTIGATE) == ExperienceStats(success=4, failure=1)


def test_failure_while_writing_the_temporary_file_does_not_corrupt_the_store(repo, monkeypatch):
    original = repo.path.read_bytes()
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(ExperienceWriteError):
        repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.SUCCESS)
    monkeypatch.undo()
    assert repo.path.read_bytes() == original
    assert [p.name for p in repo.path.parent.iterdir()] == ["experience.json"]


@pytest.mark.parametrize("action", ["investigate", "BLOCK_ACCOUNT", "", "ESCALATE"])
def test_invalid_action_is_rejected_without_writing(repo, action):
    original = repo.path.read_bytes()
    with pytest.raises(ValueError, match="action"):
        repo.record_outcome(PATTERN, action, FeedbackOutcome.SUCCESS)
    assert repo.path.read_bytes() == original


@pytest.mark.parametrize("pattern", ["", "   "])
def test_empty_pattern_is_rejected_without_writing(repo, pattern):
    original = repo.path.read_bytes()
    with pytest.raises(ValueError, match="pattern"):
        repo.record_outcome(pattern, INVESTIGATE, FeedbackOutcome.SUCCESS)
    assert repo.path.read_bytes() == original


def test_transient_permission_error_on_replace_is_retried_and_the_update_succeeds(repo, monkeypatch):
    real_replace, calls = os.replace, {"n": 0}

    def flaky_replace(source, destination):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("file briefly held open by a scanner")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr(repo_module, "REPLACE_RETRY_DELAY_SECONDS", 0)
    repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    monkeypatch.undo()
    assert calls["n"] == 3 and repo.get_stats(PATTERN, INVESTIGATE) == ExperienceStats(success=4, failure=2)
    assert [p.name for p in repo.path.parent.iterdir()] == ["experience.json"]


def test_persistent_permission_error_fails_clearly_and_keeps_the_last_valid_file(repo, monkeypatch):
    original = repo.path.read_bytes()
    attempts = {"n": 0}

    def locked(source, destination):
        attempts["n"] += 1
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", locked)
    monkeypatch.setattr(repo_module, "REPLACE_RETRY_DELAY_SECONDS", 0)
    with pytest.raises(ExperienceWriteError, match="locked"):
        repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    monkeypatch.undo()
    assert attempts["n"] == repo_module.REPLACE_ATTEMPTS  # bounded, not infinite
    assert repo.path.read_bytes() == original
    assert [p.name for p in repo.path.parent.iterdir()] == ["experience.json"]


def test_er14_repository_works_entirely_on_temporary_paths(tmp_path):
    store = ExperienceRepository.create(tmp_path / "nested.json", {PATTERN: {INVESTIGATE: (0, 0)}})
    store.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.SUCCESS)
    assert store.path.parent == tmp_path and store.path != DEFAULT_EXPERIENCE_PATH


def test_er15_committed_experience_json_is_never_mutated_by_repository_work(seed_repository):
    committed = DATA_DIR / "experience.json"
    before = hashlib.sha256(committed.read_bytes()).hexdigest()
    for _ in range(5):
        seed_repository.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    assert hashlib.sha256(committed.read_bytes()).hexdigest() == before
    assert seed_repository.path != committed


def test_copy_to_and_reset_from_support_resettable_test_stores(tmp_path):
    seed = ExperienceRepository.create(tmp_path / "seed.json", {PATTERN: {INVESTIGATE: (4, 1)}})
    working = seed.copy_to(tmp_path / "working.json")
    working.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    assert working.get_stats(PATTERN, INVESTIGATE).failure == 2 and seed.get_stats(PATTERN, INVESTIGATE).failure == 1
    working.reset_from(seed.path)
    assert working.get_stats(PATTERN, INVESTIGATE) == ExperienceStats(success=4, failure=1)


def test_repository_keeps_no_cached_state(repo):
    other = ExperienceRepository(repo.path)
    repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.FAILURE)
    assert other.get_stats(PATTERN, INVESTIGATE).failure == 2


def test_write_preserves_the_document_schema(repo):
    repo.record_outcome(PATTERN, INVESTIGATE, FeedbackOutcome.SUCCESS)
    document = json.loads(repo.path.read_text(encoding="utf-8"))
    assert set(document) == {"description", "simulated", "records"} and document["simulated"] is True
    assert not any("rate" in key for actions in document["records"].values() for a in actions.values() for key in a)
