"""Cross-platform readiness tests: the code must behave the same on Windows and Linux (Python 3.11+, 3.12 included).

Static and behavioural checks that no runtime code is Windows- or PowerShell-specific, that paths use pathlib,
and that the canonical scenario fingerprint is identical on every platform.
"""

import ast
import re
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

from models.comparison_models import scenario_fingerprint
from services.data_loader import DATA_DIR
from services.scenario_service import get_all_scenarios, get_scenario
from tests.app_helpers import ROOT

PRODUCTION_DIRS = ("agents", "config", "models", "services", "tools", "ui", "devtools")
RUNTIME_FILES = [ROOT / "app.py", *(p for d in PRODUCTION_DIRS for p in sorted((ROOT / d).glob("*.py")))]
SCRIPT_FILES = sorted((ROOT / "scripts").glob("*.py"))

# Computed on Windows. A Linux run must reproduce them exactly (canonical JSON: sorted keys, compact
# separators, UTF-8, no dependence on line endings, paths or locale).
PINNED_FINGERPRINTS = {
    "A": "008c9840b06cfdc1427439c4ed0b3879711e519e52d567a13c2abc201785eca4",
    "B": "4276b656d2177a2a47de90ec8b1614217fdfa4b8fe6f76ecdb44dd66878fa076",
    "C": "41130af1dd0990536ca6f3a8c8dbd0059aa8f9c38bfdbc2511854d583f3e3e18",
}
WINDOWS_PATH_LITERAL = re.compile(r"^[A-Za-z]:[\\/]|\\Users\\|\\AppData\\|\\Windows\\")
POWERSHELL_TOKENS = re.compile(r"powershell|Get-Content|Set-Location|Remove-Item|Invoke-|\$env:|\.ps1\b", re.IGNORECASE)
WINDOWS_ONLY_MODULES = {"msvcrt", "winreg", "_winapi", "winsound", "pywintypes", "win32api", "win32con"}


def string_constants(path: Path) -> list[str]:
    return [n.value for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_windows_path_literals_in_runtime_code(path: Path) -> None:
    offenders = [s for s in string_constants(path) if WINDOWS_PATH_LITERAL.search(s)]
    assert offenders == [], offenders


@pytest.mark.parametrize("path", [*RUNTIME_FILES, *SCRIPT_FILES], ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_powershell_commands_or_windows_only_modules_in_code(path: Path) -> None:
    code = path.read_text(encoding="utf-8")
    assert not POWERSHELL_TOKENS.search(code), POWERSHELL_TOKENS.search(code)
    roots = {a.name.split(".")[0] for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Import) for a in n.names}
    roots |= {n.module.split(".")[0] for n in ast.walk(ast.parse(code)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not roots & WINDOWS_ONLY_MODULES
    assert "os.startfile" not in code and "ctypes.windll" not in code


@pytest.mark.parametrize("path", [*RUNTIME_FILES, *SCRIPT_FILES], ids=lambda p: f"{p.parent.name}/{p.name}")
def test_paths_are_built_with_pathlib_not_string_separators(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert "path" not in {a.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                          for a in [n] if isinstance(n.value, ast.Name) and n.value.id == "os"}, "use pathlib, not os.path"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            assert node.args and not isinstance(node.args[0], ast.Constant), "open() needs a pathlib path"


def test_data_and_project_roots_are_derived_from_the_package_location() -> None:
    assert DATA_DIR.is_absolute() and DATA_DIR.name == "data" and DATA_DIR == ROOT / "data"
    for script in SCRIPT_FILES:
        text = script.read_text(encoding="utf-8")
        assert "Path(__file__)" in text or script.name == "live_common.py" or "live_common" in text, script.name


def test_the_repository_ships_no_windows_only_launchers() -> None:
    launchers = [p for p in ROOT.rglob("*") if p.suffix.lower() in {".bat", ".cmd", ".ps1"} and ".venv" not in p.parts]
    assert launchers == []
    assert all(p.suffix == ".py" for p in (ROOT / "scripts").iterdir() if p.is_file() and p.name != "__pycache__")


def test_linux_virtual_environment_paths_are_supported_and_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".venv/" in ignore and "venv/" in ignore and "__pycache__/" in ignore and ".env" in ignore
    guide = (ROOT / "docs" / "local_ollama_setup.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "source .venv/bin/activate" in guide and "source .venv/bin/activate" in readme
    assert "python3 -m venv .venv" in guide and "source .venv/bin/activate" in guide
    assert ".venv\\Scripts\\Activate.ps1" in readme  # the Windows variant is documented too, but only as an alternative


def test_streamlit_startup_configuration_is_cross_platform() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["address"] == "localhost" and config["browser"]["gatherUsageStats"] is False
    for section in config.values():
        for value in section.values():
            assert not (isinstance(value, str) and ("\\" in value or re.match(r"^[A-Za-z]:", value)))
    guide = (ROOT / "docs" / "local_ollama_setup.md").read_text(encoding="utf-8")
    assert "--server.address 127.0.0.1" in guide and "--server.port 8501" in guide


def test_requirements_are_python_312_compatible_and_not_windows_only() -> None:
    lines = [ln.strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    names = {re.split(r"[<>=!~ ;\[]", ln, maxsplit=1)[0].lower() for ln in lines}
    assert names == {"pydantic", "pytest", "langchain-core", "langchain-ollama", "streamlit"}
    assert not names & {"pywin32", "pywinpty", "colorama", "wmi", "comtypes", "pypiwin32"}
    assert all("sys_platform" not in ln and "win32" not in ln.lower() for ln in lines)
    assert sys.version_info >= (3, 11)  # the code base needs 3.11+ (StrEnum, datetime.UTC); 3.12 is a superset


@pytest.mark.parametrize("scenario_id", ["A", "B", "C"])
def test_the_scenario_fingerprint_is_pinned_and_platform_independent(scenario_id: str) -> None:
    assert scenario_fingerprint(get_scenario(scenario_id)) == PINNED_FINGERPRINTS[scenario_id]


def test_line_endings_and_directory_location_do_not_change_fingerprints(tmp_path: Path) -> None:
    crlf_dir = tmp_path / "copied data with spaces"
    crlf_dir.mkdir()
    for source in DATA_DIR.glob("*.json"):
        content = source.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")  # a Windows-style checkout
        (crlf_dir / source.name).write_bytes(content)
    copied = {s.scenario_id: scenario_fingerprint(s) for s in get_all_scenarios(crlf_dir)}
    assert copied == PINNED_FINGERPRINTS
    lf_dir = tmp_path / "lf"
    shutil.copytree(DATA_DIR, lf_dir)
    for source in lf_dir.glob("*.json"):
        source.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))  # a Linux-style checkout
    assert {s.scenario_id: scenario_fingerprint(s) for s in get_all_scenarios(lf_dir)} == PINNED_FINGERPRINTS


def test_the_canonical_form_is_explicit() -> None:
    import json
    import hashlib

    scenario = get_scenario("A")
    canonical = json.dumps(scenario.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert "\n" not in canonical and "\r" not in canonical and ": " not in canonical
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == PINNED_FINGERPRINTS["A"]
