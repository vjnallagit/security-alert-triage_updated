import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from models.llm_models import CallStatus
from models.tool_models import TOOL_CATALOGUE
from services.data_loader import DATA_DIR
from services.llm_errors import LLMDependencyUnavailableError
from services.security_tool_service import SecurityToolService
from tools.langchain_security_tools import (
    TOOL_DESCRIPTIONS,
    SecurityToolbox,
    build_langchain_tools,
    tool_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAMES = ["get_alert_details", "search_authentication_events", "get_asset_criticality", "get_response_policy"]
DATA_FILES = ("scenarios.json", "alert_details.json", "authentication_events.json", "assets.json",
              "policies.json", "experience.json")
PAYLOAD_KEYS = {"tool_name", "status", "summary", "evidence", "source_record_ids", "warnings", "limitations",
                "error", "is_synthetic", "is_read_only", "executed_actions", "audit_call_id"}


class SpyService(SecurityToolService):
    """A real Phase 6 service that also records how each method was called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[str, dict]] = []

    def get_alert_details(self, alert_id, include_raw_evidence=True):
        self.calls.append(("get_alert_details", {"alert_id": alert_id, "include_raw_evidence": include_raw_evidence}))
        return super().get_alert_details(alert_id, include_raw_evidence)

    def search_authentication_events(self, **kwargs):
        self.calls.append(("search_authentication_events", kwargs))
        return super().search_authentication_events(**kwargs)

    def get_asset_criticality(self, asset_id):
        self.calls.append(("get_asset_criticality", {"asset_id": asset_id}))
        return super().get_asset_criticality(asset_id)

    def get_response_policy(self, **kwargs):
        self.calls.append(("get_response_policy", kwargs))
        return super().get_response_policy(**kwargs)


@pytest.fixture
def spy() -> SpyService:
    return SpyService()


@pytest.fixture
def toolbox(spy) -> SecurityToolbox:
    return SecurityToolbox(spy)


def data_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() for name in DATA_FILES}


def test_lt01_exactly_four_langchain_tools_are_registered(toolbox):
    assert len(toolbox.tools) == 4 and len(toolbox.tool_names) == 4


def test_lt02_names_exactly_match_the_phase_6_public_names(toolbox):
    assert [t.name for t in toolbox.tools] == PUBLIC_NAMES == [d.name for d in TOOL_CATALOGUE]
    assert list(TOOL_DESCRIPTIONS) == PUBLIC_NAMES


def test_lt03_each_wrapper_has_a_non_empty_description(toolbox):
    for tool in toolbox.tools:
        assert len(tool.description) > 80 and tool.description == TOOL_DESCRIPTIONS[tool.name]
    assert "does not determine final priority" in TOOL_DESCRIPTIONS["get_alert_details"]
    assert "does not confirm compromise" in TOOL_DESCRIPTIONS["search_authentication_events"]
    assert "not a final incident priority" in TOOL_DESCRIPTIONS["get_asset_criticality"]
    assert "never executes the policy" in TOOL_DESCRIPTIONS["get_response_policy"]


def test_lt04_each_wrapper_uses_the_correct_pydantic_argument_schema(toolbox):
    by_name = {d.name: d.input_model for d in TOOL_CATALOGUE}
    for tool in toolbox.tools:
        assert tool.args_schema is by_name[tool.name]
        assert set(tool.args) == set(by_name[tool.name].model_fields)


def test_lt05_get_alert_details_delegates_to_phase_6(toolbox, spy):
    execution = toolbox.execute("get_alert_details", {"alert_id": "ALERT-A-001"})
    assert spy.calls == [("get_alert_details", {"alert_id": "ALERT-A-001", "include_raw_evidence": True})]
    assert execution.payload["source_record_ids"] == ["ALERT-A-001"]


def test_lt06_authentication_wrapper_delegates_to_phase_6(toolbox, spy):
    execution = toolbox.execute("search_authentication_events", {"account": "admin.synthetic01", "success": False})
    assert spy.calls[0][0] == "search_authentication_events"
    assert spy.calls[0][1]["account"] == "admin.synthetic01" and spy.calls[0][1]["success"] is False
    assert execution.payload["evidence"]["matched_count"] == 12


def test_lt07_asset_wrapper_delegates_to_phase_6(toolbox, spy):
    execution = toolbox.execute("get_asset_criticality", {"asset_id": "ASSET-PAY-01"})
    assert spy.calls == [("get_asset_criticality", {"asset_id": "ASSET-PAY-01"})]
    assert execution.payload["evidence"]["criticality"] == "CRITICAL"


def test_lt08_policy_wrapper_delegates_to_phase_6(toolbox, spy):
    execution = toolbox.execute("get_response_policy", {"policy_id": "POL-PRIV-LOGIN-001"})
    assert spy.calls[0][0] == "get_response_policy" and spy.calls[0][1]["policy_id"] == "POL-PRIV-LOGIN-001"
    assert execution.payload["evidence"]["automatic_disruptive_action_permitted"] is False


def test_lt09_wrapper_returns_compact_json(toolbox):
    langchain_tool = toolbox.tools[0]
    content = langchain_tool.invoke({"alert_id": "ALERT-A-001"})
    assert isinstance(content, str) and set(json.loads(content)) == PAYLOAD_KEYS
    assert len(content) < 2500  # compact: no raw model dumps or unrelated records
    assert "ALERT-B-001" not in content and "admin.synthetic01" in content


def test_lt10_wrapper_preserves_source_ids(toolbox):
    events = toolbox.execute("search_authentication_events", {"scenario_id": "A"})
    assert events.payload["source_record_ids"] == [f"AUTH-A-{n:03d}" for n in range(12, 0, -1)]  # newest first
    assert toolbox.execute("get_asset_criticality", {"asset_id": "ASSET-IDP-01"}).payload["source_record_ids"] == ["ASSET-IDP-01"]


def test_lt11_wrapper_preserves_warnings_and_limitations(toolbox):
    truncated = toolbox.execute("search_authentication_events", {"account": "admin.synthetic01", "limit": 5})
    assert truncated.payload["warnings"] and "truncated" in truncated.payload["warnings"][0].lower()
    for name, args in (("get_alert_details", {"alert_id": "ALERT-A-001"}), ("get_response_policy", {"policy_id": "POL-PRIV-LOGIN-001"})):
        assert toolbox.execute(name, args).payload["limitations"]
    assert "synthetic" in " ".join(toolbox.execute("get_asset_criticality", {"asset_id": "ASSET-IDP-01"}).payload["limitations"]).lower()


def test_lt12_wrapper_performs_no_final_decision(toolbox):
    for name, args in (("get_alert_details", {"alert_id": "ALERT-C-001"}), ("get_asset_criticality", {"asset_id": "ASSET-PAY-01"}),
                       ("get_response_policy", {"policy_id": "POL-CORRELATED-CRITICAL-001"}),
                       ("search_authentication_events", {"scenario_id": "C"})):
        payload = toolbox.execute(name, args).payload
        assert not {"decision", "priority", "agent_name", "reasoning", "recommended_actions"} & set(payload)
        evidence_keys = set((payload["evidence"] or {}))
        assert not {"decision", "priority", "recommended_actions"} & evidence_keys


def test_lt13_wrapper_performs_no_remediation(toolbox):
    for name, args in (("get_alert_details", {"alert_id": "ALERT-A-001"}), ("get_response_policy", {"policy_id": "POL-PRIV-LOGIN-001"})):
        payload = toolbox.execute(name, args).payload
        assert payload["executed_actions"] == [] and payload["is_read_only"] is True and payload["is_synthetic"] is True


def test_lt14_an_injected_service_is_used(tmp_path):
    import shutil

    copy = tmp_path / "data"
    copy.mkdir()
    for name in ("alert_details.json", "authentication_events.json", "assets.json", "policies.json", "scenarios.json"):
        shutil.copy(DATA_DIR / name, copy / name)
    assets = json.loads((copy / "assets.json").read_text(encoding="utf-8"))
    assets[2]["criticality"] = "LOW"
    (copy / "assets.json").write_text(json.dumps(assets), encoding="utf-8")
    service = SecurityToolService(
        alert_details_path=copy / "alert_details.json", authentication_events_path=copy / "authentication_events.json",
        assets_path=copy / "assets.json", policies_path=copy / "policies.json", scenarios_path=copy / "scenarios.json")
    box = SecurityToolbox(service)
    assert box.service is service
    assert box.execute("get_asset_criticality", {"asset_id": "ASSET-PAY-01"}).payload["evidence"]["criticality"] == "LOW"
    assert [r.tool_name for r in service.get_invocation_log()] == ["get_asset_criticality"]


def test_lt15_no_source_data_is_modified(toolbox):
    before = data_hashes()
    for _ in range(5):
        for name, args in (("get_alert_details", {"alert_id": "ALERT-A-001"}), ("search_authentication_events", {"scenario_id": "B"}),
                           ("get_asset_criticality", {"asset_id": "ASSET-WIKI-01"}), ("get_response_policy", {"severity": "HIGH"}),
                           ("get_weather", {}), ("get_alert_details", {"alert_id": ""})):
            toolbox.execute(name, args)
    assert data_hashes() == before


def test_unknown_tool_is_never_executed_and_returns_a_safe_error(toolbox, spy):
    execution = toolbox.execute("delete_everything", {"path": "C:\\"})
    assert execution.status is CallStatus.UNKNOWN_TOOL and spy.calls == []
    assert execution.payload["evidence"] is None and execution.payload["available_tools"] == PUBLIC_NAMES
    assert execution.payload["executed_actions"] == []


@pytest.mark.parametrize("arguments", [{"alert_id": ""}, {"alert_id": "A", "run_remediation": True}, {}, {"alert_id": 5},
                                       "not-a-dict", None, ["alert_id"]])
def test_malformed_or_unexpected_arguments_return_invalid_input_without_calling_the_tool(toolbox, spy, arguments):
    execution = toolbox.execute("get_alert_details", arguments)
    assert execution.status is CallStatus.INVALID_INPUT and spy.calls == []
    assert execution.payload["status"] == "INVALID_INPUT" and execution.payload["error"]
    assert "Traceback" not in execution.content


def test_search_without_a_criterion_is_reported_by_phase_6_as_invalid_input(toolbox):
    execution = toolbox.execute("search_authentication_events", {"limit": 5})
    assert execution.status is CallStatus.INVALID_INPUT and "at least one search criterion" in execution.payload["error"]


def test_statuses_not_found_and_ambiguous_flow_through_unchanged(toolbox):
    assert toolbox.execute("get_alert_details", {"alert_id": "NOPE"}).status is CallStatus.NOT_FOUND
    ambiguous = toolbox.execute("get_response_policy", {"severity": "HIGH"})
    assert ambiguous.status is CallStatus.AMBIGUOUS and ambiguous.payload["evidence"] is None


def test_tool_payload_adds_only_safety_declarations_to_the_phase_6_view(spy):
    result = spy.get_alert_details("ALERT-A-001")
    payload = tool_payload(result)
    assert set(payload) == set(result.to_llm_evidence()) | {"is_synthetic", "is_read_only", "executed_actions", "audit_call_id"}
    assert payload["audit_call_id"] == result.call_id


def test_execution_arguments_are_sanitized_json(toolbox):
    execution = toolbox.execute("get_response_policy", {"severity": "HIGH", "alert_type": None})
    assert execution.arguments == {"severity": "HIGH"} and json.dumps(execution.arguments)


def test_missing_langchain_raises_a_clear_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", None)
    with pytest.raises(LLMDependencyUnavailableError, match="pip install -r requirements.txt") as info:
        build_langchain_tools(SecurityToolService())
    assert isinstance(info.value.__cause__, ImportError)


def run_clean_python(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_the_tool_wrappers_loads_no_langchain_or_network_package():
    out = run_clean_python(
        "import sys, tools.langchain_security_tools\n"
        "bad = sorted(m for m in ('langchain_core','langchain_ollama','ollama','httpx','requests') if m in sys.modules)\n"
        "print(bad)")
    assert out == "[]"
