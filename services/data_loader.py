"""Single place where JSON data files are located and read.

Files are resolved relative to this package (not the shell's working directory)
and every failure raises an explicit exception instead of being swallowed.
"""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class DataLoadError(Exception):
    """Base class for all data loading problems."""


class DataFileNotFoundError(DataLoadError):
    """The requested data file does not exist."""


class MalformedDataError(DataLoadError):
    """The file is not valid JSON."""


class UnexpectedStructureError(DataLoadError):
    """The JSON is valid but its top-level type is not what the caller expects."""


def _read_json(filename: str, data_dir: Path) -> Any:
    base = data_dir.resolve()
    path = (base / filename).resolve()
    # Guards against filenames such as "../secret.json" escaping the data directory.
    if not path.is_relative_to(base):
        raise DataLoadError(f"Data file '{filename}' is outside the data directory {base}")
    if not path.is_file():
        raise DataFileNotFoundError(f"Data file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MalformedDataError(f"Malformed JSON in {path}: {exc}") from exc


def _require_type(filename: str, data: Any, expected: type) -> None:
    if not isinstance(data, expected):
        raise UnexpectedStructureError(
            f"{filename} must contain a JSON {_json_name(expected)} at the top level, "
            f"found {type(data).__name__}"
        )


def _json_name(expected: type) -> str:
    return "array" if expected is list else "object"


def load_json_list(filename: str, data_dir: Path = DATA_DIR) -> list[Any]:
    """Load a data file whose top level must be a JSON array."""
    data = _read_json(filename, data_dir)
    _require_type(filename, data, list)
    return data


def load_json_object(filename: str, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Load a data file whose top level must be a JSON object."""
    data = _read_json(filename, data_dir)
    _require_type(filename, data, dict)
    return data
