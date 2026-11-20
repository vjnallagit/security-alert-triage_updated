import ast
import hashlib
import inspect
import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import tools
from models.tool_models import (
    TOOL_CATALOGUE,
    GetAlertDetailsResult,
    GetAssetCriticalityResult,
    GetResponsePolicyResult,
    SearchAuthenticationEventsResult,
    ToolStatus,
)
from services.data_loader import DATA_DIR
from services.security_tool_service import SecurityToolService
from tools import (
    get_alert_details,
    get_asset_criticality,
    get_response_policy,
    search_authentication_events,
)
from tools import security_tools as tools_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_DATA_FILES = ("scenarios.json", "alert_details.json", "authentication_events.json", "assets.json",
                  "policies.json", "experience.json")
TOOL_SOURCES = ("tools/security_tools.py", "services/security_tool_service.py", "models/tool_models.py")
FORBIDDEN_MODULES = {"langchain", "langchain_core", "langchain_ollama", "ollama", "requests", "httpx", "urllib",
                     "urllib3", "http", "socket", "aiohttp", "fastapi", "streamlit", "subprocess", "smtplib", "ftplib"}


def data_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() for name in ALL_DATA_FILES}


def imported_modules(relative_path: str) -> set[str]:
    tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def call_all_tools(service: SecurityToolService | None = None):
    kwargs = {"service": service} if service else {}
    return (
        get_alert_details("ALERT-A-001", **kwargs),
        search_authentication_events(account="admin.synthetic01", success=False, **kwargs),
        get_asset_criticality("ASSET-IDP-01", **kwargs),
        get_response_policy(policy_id="POL-PRIV-LOGIN-001", **kwargs),
    )


def test_pt01_all_four_functions_are_importable_from_tools():
    for name in ("get_alert_details", "search_authentication_events", "get_asset_criticality",
                 "get_response_policy", "get_tool_catalogue"):
        assert callable(getattr(tools, name))
    assert set(tools.__all__) == {"get_alert_details", "search_authentication_events", "get_asset_criticality",
                                  "get_response_policy", "get_tool_catalogue"}


def test_pt02_each_function_delegates_to_the_security_tool_service():
    service = MagicMock(spec=SecurityToolService)
    get_alert_details("ALERT-A-001", include_raw_evidence=False, service=service)
    service.get_alert_details.assert_called_once_with("ALERT-A-001", False)
    start = datetime(2026, 3, 10, tzinfo=UTC)
    search_authentication_events(account="a", source="s", asset_id="x", alert_id="al", scenario_id="A", success=False,
                                 start_time=start, end_time=start, limit=7, service=service)
    service.search_authentication_events.assert_called_once_with(
        account="a", source="s", asset_id="x", alert_id="al", scenario_id="A", success=False,
        start_time=start, end_time=start, limit=7)
    get_asset_criticality("ASSET-X", service=service)
    service.get_asset_criticality.assert_called_once_with("ASSET-X")
    get_response_policy(policy_id="P", alert_type="t", severity="HIGH", asset_criticality="CRITICAL", service=service)
    service.get_response_policy.assert_called_once_with(
        policy_id="P", alert_type="t", severity="HIGH", asset_criticality="CRITICAL")


def test_pt03_an_injected_service_is_used(tmp_path):
    import shutil

    copy = tmp_path / "data"
    copy.mkdir()
    for name in ("alert_details.json", "authentication_events.json", "assets.json", "policies.json", "scenarios.json"):
        shutil.copy(DATA_DIR / name, copy / name)
    assets = json.loads((copy / "assets.json").read_text(encoding="utf-8"))
    assets[2]["criticality"] = "LOW"  # differs from the committed data
    (copy / "assets.json").write_text(json.dumps(assets), encoding="utf-8")
    service = SecurityToolService(
        alert_details_path=copy / "alert_details.json", authentication_events_path=copy / "authentication_events.json",
        assets_path=copy / "assets.json", policies_path=copy / "policies.json", scenarios_path=copy / "scenarios.json")
    assert get_asset_criticality("ASSET-PAY-01", service=service).evidence.criticality.value == "LOW"
    assert get_asset_criticality("ASSET-PAY-01").evidence.criticality.value == "CRITICAL"
    assert [r.tool_name for r in service.get_invocation_log()] == ["get_asset_criticality"]  # the log is the injected service's


def test_pt04_the_default_service_works_against_the_committed_synthetic_data():
    alert, events, asset, policy = call_all_tools()
    assert [r.status for r in (alert, events, asset, policy)] == [ToolStatus.SUCCESS] * 4
    assert events.evidence.matched_count == 12 and asset.evidence.criticality.value == "HIGH"


def test_pt05_a_public_call_returns_the_documented_typed_result():
    alert, events, asset, policy = call_all_tools()
    assert isinstance(alert, GetAlertDetailsResult) and isinstance(events, SearchAuthenticationEventsResult)
    assert isinstance(asset, GetAssetCriticalityResult) and isinstance(policy, GetResponsePolicyResult)
    for result in (alert, events, asset, policy):
        assert json.dumps(result.model_dump(mode="json")) and json.dumps(result.to_llm_evidence())


def test_default_calls_create_no_shared_module_level_state():
    first, second = get_alert_details("ALERT-A-001"), get_alert_details("ALERT-A-001")
    assert first.call_id != second.call_id  # independent calls on independent default services
    module_state = [n for n, v in vars(tools_module).items()
                    if not n.startswith("__") and isinstance(v, (list, dict, set, SecurityToolService))]
    assert module_state == []


def test_pt06_public_functions_do_not_mutate_source_data():
    before = data_hashes()
    for _ in range(5):
        call_all_tools()
    assert data_hashes() == before


@pytest.mark.parametrize("path", TOOL_SOURCES)
def test_pt07_pt08_pt09_tool_layer_imports_no_llm_or_network_library(path):
    assert not imported_modules(path) & FORBIDDEN_MODULES, imported_modules(path) & FORBIDDEN_MODULES


def test_pt09_no_http_or_socket_calls_appear_in_the_tool_source():
    for path in TOOL_SOURCES:
        source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not called & {"urlopen", "connect", "getaddrinfo", "create_connection", "request", "post", "get_connection"}


def test_pt10_public_function_arguments_are_representable_by_the_future_input_schemas():
    for definition in TOOL_CATALOGUE:
        function = getattr(tools, definition.name)
        parameters = set(inspect.signature(function).parameters) - {"service"}
        assert parameters == set(definition.input_model.model_fields), definition.name
        assert definition.input_schema()["properties"].keys() == set(definition.input_model.model_fields)
        # Every function can be called with exactly the validated input model's fields.
        defaults = {name: field.default for name, field in definition.input_model.model_fields.items() if not field.is_required()}
        assert all(name in parameters for name in defaults)


def test_all_tool_modules_have_no_tool_function_with_a_hidden_mutable_default():
    for definition in TOOL_CATALOGUE:
        for parameter in inspect.signature(getattr(tools, definition.name)).parameters.values():
            assert not isinstance(parameter.default, (list, dict, set)), (definition.name, parameter.name)


def test_filesystem_immutability_all_six_data_files_stay_byte_identical_across_repeated_calls():
    before = data_hashes()
    service = SecurityToolService()
    for _ in range(25):
        call_all_tools(service)
        service.search_authentication_events(scenario_id="C")
        service.get_alert_details("NOPE")
        service.get_response_policy(severity="HIGH")  # ambiguous
        service.get_asset_criticality(" ")  # invalid
    assert data_hashes() == before


def test_network_isolation_all_tools_succeed_with_every_network_call_blocked(monkeypatch):
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "gethostbyname", blocked)
    results = call_all_tools()
    assert [r.status for r in results] == [ToolStatus.SUCCESS] * 4 and attempts == []


def test_no_llm_or_http_library_is_loaded_by_the_tool_layer():
    import sys

    # LangChain and its HTTP stack may legitimately be loaded by the separate LLM layer's tests, so check what
    # calling THESE tools newly loads. The import statements are covered by the AST test above.
    before = set(sys.modules)
    call_all_tools()
    assert not {"langchain", "langchain_core", "langchain_ollama", "ollama", "requests", "httpx", "streamlit",
                "fastapi"} & (set(sys.modules) - before)


def test_requirements_stay_minimal():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    # Streamlit is now the declared UI dependency; the tool layer itself still never imports it (AST test above).
    assert not any(name in requirements for name in ("fastapi", "requests", "httpx", "pandas", "numpy"))
