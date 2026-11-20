"""Persistent store for the Learning Agent's success/failure experience.

All JSON reading goes through services.data_loader (no duplicated parsing) and every
write is atomic: the new document is validated, written to a temporary file in the same
directory, flushed to disk, then swapped in with os.replace. A failure at any step leaves
the previous valid file untouched.

Limitation: this JSON store suits a single-process academic demonstration. A production
multi-process system would need a transactional datastore or explicit concurrency control.

The repository holds no cached state: every call reads the file, so separate repository
or agent instances pointing at the same path always see each other's updates.
"""

import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import get_args

from models.goal_models import DecisionName
from models.learning_models import ExperienceStats, FeedbackOutcome
from models.schemas import ExperienceData, ExperienceOutcome
from pydantic import ValidationError
from services.data_loader import DATA_DIR, DataLoadError, load_json_object

DEFAULT_EXPERIENCE_PATH = DATA_DIR / "experience.json"
VALID_ACTIONS = frozenset(get_args(DecisionName))
# On Windows, os.replace can fail briefly if a scanner or indexer has the target open.
REPLACE_ATTEMPTS = 6
REPLACE_RETRY_DELAY_SECONDS = 0.05
SIMULATION_DESCRIPTION = (
    "SIMULATED academic experience data for the Learning Agent. These counts are invented for "
    "coursework and are NOT real security statistics. Success rate is derived as "
    "success / (success + failure)."
)


class InvalidExperienceDataError(DataLoadError):
    """The experience document does not match the expected structure."""


class ExperienceWriteError(DataLoadError):
    """The experience file could not be updated; the previous file is unchanged."""


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Atomic swap; retries only the transient PermissionError, then fails clearly."""
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS:
                raise
            time.sleep(REPLACE_RETRY_DELAY_SECONDS)


def _atomic_write(path: Path, document: ExperienceData) -> None:
    text = json.dumps(document.model_dump(mode="json"), indent=2) + "\n"
    temporary: Path | None = None
    try:
        handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
        temporary = Path(name)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
        temporary = None
    except OSError as exc:
        raise ExperienceWriteError(f"Could not update experience store {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ExperienceRepository:
    """Reads and updates experience counts for (pattern, action) pairs."""

    def __init__(self, path: Path = DEFAULT_EXPERIENCE_PATH) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    # --- reading -----------------------------------------------------------

    def load(self) -> ExperienceData:
        raw = load_json_object(self._path.name, self._path.parent)
        try:
            return ExperienceData.model_validate(raw)
        except ValidationError as exc:
            raise InvalidExperienceDataError(f"Invalid experience data in {self._path}: {exc}") from exc

    def patterns(self) -> tuple[str, ...]:
        return tuple(self.load().records)

    def get_pattern_history(self, pattern: str) -> dict[str, ExperienceStats]:
        """Every action recorded for a pattern (empty if the pattern is unknown)."""
        actions = self.load().records.get(pattern, {})
        return {action: self._stats(outcome) for action, outcome in actions.items()}

    def get_stats(self, pattern: str, action: str) -> ExperienceStats:
        """Counts for one pattern/action; 0/0 when nothing has been recorded yet."""
        outcome = self.load().records.get(pattern, {}).get(action)
        return self._stats(outcome) if outcome else ExperienceStats(success=0, failure=0)

    # --- updating ----------------------------------------------------------

    def record_outcome(
        self, pattern: str, action: str, outcome: FeedbackOutcome
    ) -> tuple[ExperienceStats, ExperienceStats]:
        """Count one outcome for one pattern/action and persist it. Returns (before, after).

        Only that single entry changes; a missing entry is initialised at 0/0 first.
        UNKNOWN is never counted and does not touch the file.
        """
        if not pattern.strip():
            raise ValueError("pattern must not be empty")
        if action not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {sorted(VALID_ACTIONS)}, got {action!r}")

        document = self.load()
        before = self.get_stats(pattern, action)
        after = before.after(outcome)
        if after == before:
            return before, before

        updated = document.model_copy(deep=True)
        updated.records.setdefault(pattern, {})[action] = ExperienceOutcome(
            success=after.success, failure=after.failure
        )
        # Re-validate the whole document so an invalid structure is never written.
        _atomic_write(self._path, ExperienceData.model_validate(updated.model_dump()))
        return before, after

    # --- test / demonstration support ------------------------------------------

    @classmethod
    def create(
        cls, path: Path, records: Mapping[str, Mapping[str, tuple[int, int]]]
    ) -> "ExperienceRepository":
        """Write a new store from {pattern: {action: (success, failure)}} (temporary/test stores)."""
        document = ExperienceData(
            description=SIMULATION_DESCRIPTION,
            simulated=True,
            records={
                pattern: {
                    action: ExperienceOutcome(success=counts[0], failure=counts[1])
                    for action, counts in actions.items()
                }
                for pattern, actions in records.items()
            },
        )
        _atomic_write(Path(path), document)
        return cls(path)

    def copy_to(self, destination: Path) -> "ExperienceRepository":
        """Copy this (validated) store to another path and return a repository for the copy."""
        _atomic_write(Path(destination), self.load())
        return ExperienceRepository(destination)

    def reset_from(self, seed: Path) -> None:
        """Replace this store's content with a validated copy of the seed file."""
        _atomic_write(self._path, ExperienceRepository(seed).load())

    @staticmethod
    def _stats(outcome: ExperienceOutcome) -> ExperienceStats:
        return ExperienceStats(success=outcome.success, failure=outcome.failure)
