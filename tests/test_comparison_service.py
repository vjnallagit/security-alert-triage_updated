"""Six-agent comparison service tests: CS-01..15, EO-01..06, LD-01..08, MD-01..05 (plus extras).

All offline: the LLM Agent runs on the scripted MOCK model, and no Ollama server is contacted.
"""

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import services.comparison_service as comparison_module
from agents import (
    GoalBasedAgent,
    LearningAgent,
    LLMAgent,
    ModelBasedReflexAgent,
    SimpleReflexAgent,
    UtilityBasedAgent,
)
from agents.base import BaseAgent
from config.llm_config import OllamaConfig
from models.comparison_models import (
    AGENT_ORDER,
    AgentComparisonEntry,
    AgentComparisonResult,
    AgentStatus,
    scenario_fingerprint,
)
from models.learning_models import FeedbackOutcome
from models.llm_models import RuntimeMode
from models.schemas import AgentDecision, AgentType
from services.comparison_service import (
    LearningFeedbackDemo,
    ModelBasedMemoryDemo,
    live_ollama_status,
    redact_sensitive_text,
    safe_error_message,
)
from services.data_loader import DATA_DIR
from services.experience_repository import ExperienceRepository
from services.llm_errors import LLMConnectionError
from services.scenario_service import get_all_scenarios, get_scenario
from tests.fakes.fake_tool_calling_model import happy_path_model
from tests.fakes.service import mock_service

SCENARIO_IDS = ["A", "B", "C"]
pytestmark = pytest.mark.usefixtures("clean_runtime_environment")  # never inherit real local settings
EXPECTED = {
    "A": ("HIGH", "INVESTIGATE"),
    "B": ("LOW", "COLLECT_MORE_EVIDENCE"),
    "C": ("CRITICAL", "ESCALATE_TO_SECURITY_ANALYST"),
}
DATA_FILES = sorted(DATA_DIR.glob("*.json"))
EXPERIENCE = DATA_DIR / "experience.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_hashes() -> dict[str, str]:
    return {p.name: sha(p) for p in DATA_FILES}


def compare(scenario_id: str = "A", **kwargs: Any) -> AgentComparisonResult:
    return mock_service(**kwargs).compare(get_scenario(scenario_id))


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a network connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# ------------------------------------------------------------------------------ CS: comparison


def test_cs_01_returns_six_entries() -> None:
    result = compare()
    assert len(result.results) == 6 and result.all_succeeded and result.errors == ()


def test_cs_02_exactly_the_expected_six_agent_types_in_order() -> None:
    assert tuple(e.agent_type for e in compare().results) == AGENT_ORDER
    assert set(AGENT_ORDER) == {
        AgentType.SIMPLE_REFLEX, AgentType.MODEL_BASED_REFLEX, AgentType.GOAL_BASED,
        AgentType.UTILITY_BASED, AgentType.LEARNING, AgentType.LLM}


class SpyAgent(BaseAgent):
    name = "spy"

    def __init__(self, inner: BaseAgent, log: list[Any]) -> None:
        self._inner, self._log = inner, log

    def decide(self, scenario: Any) -> AgentDecision:
        self._log.append(scenario)
        return self._inner.decide(scenario)


def spy_factories(log: list[Any]) -> dict[AgentType, Any]:
    real = {
        AgentType.SIMPLE_REFLEX: lambda s: SimpleReflexAgent(),
        AgentType.MODEL_BASED_REFLEX: lambda s: ModelBasedReflexAgent(),
        AgentType.GOAL_BASED: lambda s: GoalBasedAgent(),
        AgentType.UTILITY_BASED: lambda s: UtilityBasedAgent(),
        AgentType.LEARNING: lambda s: LearningAgent(ExperienceRepository(EXPERIENCE)),
        AgentType.LLM: lambda s: LLMAgent(model=happy_path_model(s, parallel=False), config=OllamaConfig()),
    }
    return {t: (lambda s, build=build: SpyAgent(build(s), log)) for t, build in real.items()}


def test_cs_03_all_six_agents_receive_the_same_scenario() -> None:
    received: list[Any] = []
    scenario = get_scenario("A")
    result = mock_service(agent_factories=spy_factories(received)).compare(scenario)
    assert result.all_succeeded and len(received) == 6
    # Equivalent data for every agent, each in its own isolated copy, and the caller's scenario untouched.
    assert len({json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in received}) == 1
    assert all(item == scenario for item in received) and len({id(item) for item in received}) == 6
    assert {scenario_fingerprint(item) for item in received} == {result.scenario_fingerprint}
    assert scenario_fingerprint(scenario) == result.scenario_fingerprint


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_cs_04_scenario_fingerprint_is_identical_across_agent_executions(scenario_id: str) -> None:
    result = compare(scenario_id)
    assert {e.input_fingerprint for e in result.results} == {result.scenario_fingerprint}
    assert result.scenario_fingerprint == scenario_fingerprint(get_scenario(scenario_id))
    assert len(result.scenario_fingerprint) == 64


def test_fingerprint_is_canonical_and_excludes_agent_output() -> None:
    first, second = compare("A"), compare("A")
    assert first.scenario_fingerprint == second.scenario_fingerprint  # stable across runs
    fingerprints = {sid: compare(sid).scenario_fingerprint for sid in SCENARIO_IDS}
    assert len(set(fingerprints.values())) == 3  # different scenarios differ
    scenario = get_scenario("A")
    changed = scenario.model_copy(update={"failed_login_count": scenario.failed_login_count + 1})
    assert scenario_fingerprint(changed) != scenario_fingerprint(scenario)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_cs_05_to_07_each_scenario_produces_the_validated_outputs(scenario_id: str) -> None:
    result = compare(scenario_id)
    priority, decision = EXPECTED[scenario_id]
    for entry in result.results:
        assert (entry.priority.value, entry.decision) == (priority, decision), entry.agent_name


class RecordingModelBased(ModelBasedReflexAgent):
    instances: list["RecordingModelBased"] = []

    def __init__(self) -> None:
        super().__init__()
        type(self).instances.append(self)


def test_cs_08_main_comparison_always_uses_a_fresh_model_based_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingModelBased.instances = []
    monkeypatch.setattr(comparison_module, "ModelBasedReflexAgent", RecordingModelBased)
    service = mock_service()
    scenario = get_scenario("A")
    service.compare(scenario)
    service.compare(scenario)
    assert len(RecordingModelBased.instances) == 2
    assert RecordingModelBased.instances[0] is not RecordingModelBased.instances[1]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_cs_09_repeated_comparisons_do_not_accumulate_model_based_state(scenario_id: str) -> None:
    service = mock_service()
    scenario = get_scenario(scenario_id)
    for _ in range(3):
        entry = service.compare(scenario).entry(AgentType.MODEL_BASED_REFLEX)
        state = entry.special_details["state_summary"]
        assert state["processed_scenario_count"] == 1 and state["account_count"] == 1
        assert state["pattern_seen_previously"] is False and state["previous_decision"] is None
        assert (entry.priority.value, entry.decision) == EXPECTED[scenario_id]


def test_cs_10_main_comparison_never_changes_committed_experience() -> None:
    before = data_hashes()
    service = mock_service()
    for scenario in get_all_scenarios():
        for _ in range(2):
            service.compare(scenario)
    assert data_hashes() == before


def test_cs_10_learning_reads_a_stable_seed_each_time() -> None:
    first = compare("A").entry(AgentType.LEARNING).special_details["candidate_experience"]
    second = compare("A").entry(AgentType.LEARNING).special_details["candidate_experience"]
    assert first == second


def test_cs_11_mock_llm_entry_is_explicitly_mock() -> None:
    result = compare("A")
    llm = result.entry(AgentType.LLM)
    assert result.llm_runtime_mode is RuntimeMode.MOCK and llm.llm_runtime_mode is RuntimeMode.MOCK
    assert llm.decision_record.metadata["runtime_mode"] == "MOCK"
    assert all(e.llm_runtime_mode is None for e in result.results if e.agent_type is not AgentType.LLM)


def test_cs_12_a_mock_result_cannot_be_labelled_ollama() -> None:
    # Live mode requested, but the factory yields the scripted mock: the result must be rejected, not relabelled.
    def mock_factory(scenario: Any) -> BaseAgent:
        return LLMAgent(model=happy_path_model(scenario), config=OllamaConfig())

    result = compare("A", llm_mode=RuntimeMode.OLLAMA, agent_factories={AgentType.LLM: mock_factory})
    llm = result.entry(AgentType.LLM)
    assert llm.status is AgentStatus.ERROR and llm.decision_record is None and llm.priority is None
    assert "MOCK" in (llm.error_message or "") or "runtime mode" in (llm.error_message or "")
    assert len(result.successful) == 5


def test_cs_13_live_failure_never_falls_back_silently_to_the_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    import devtools.mock_llm_runtime as fakes  # the single implementation the mock factory actually calls

    mock_calls: list[Any] = []
    monkeypatch.setattr(fakes, "happy_path_model", lambda *a, **k: mock_calls.append(a))
    monkeypatch.delenv("RUN_OLLAMA_INTEGRATION", raising=False)
    result = compare("A", llm_mode=RuntimeMode.OLLAMA)  # live requested but not enabled
    llm = result.entry(AgentType.LLM)
    assert mock_calls == []  # the mock model was never even built
    assert llm.status is AgentStatus.ERROR and "unavailable" in llm.error_message.lower()
    assert llm.llm_runtime_mode is RuntimeMode.OLLAMA  # records what was requested; no result was produced
    assert len(result.successful) == 5 and result.errors[0].agent_type is AgentType.LLM


def test_cs_13_live_connection_failure_is_an_error_not_a_mock_result(monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.llm_agent as llm_agent_module

    class Unreachable:
        def __init__(self, config: Any) -> None:
            self.runtime_mode, self.model_name = RuntimeMode.OLLAMA, config.model

        def bind_tools(self, _tools: Any) -> Any:
            raise LLMConnectionError("Could not reach the local Ollama server.")

    monkeypatch.setattr(llm_agent_module, "OllamaChatModelAdapter", Unreachable)
    config = OllamaConfig(model="example-model", integration_enabled=True)
    result = compare("A", llm_mode=RuntimeMode.OLLAMA, ollama_config=config)
    llm = result.entry(AgentType.LLM)
    assert llm.status is AgentStatus.ERROR and "Live Ollama is unavailable" in llm.error_message
    assert llm.llm_runtime_mode is RuntimeMode.OLLAMA and llm.decision_record is None
    assert len(result.successful) == 5


def test_cs_14_one_agent_failure_does_not_discard_other_results() -> None:
    def boom(_scenario: Any) -> BaseAgent:
        raise RuntimeError('File "C:\\Users\\someone\\secret.py", line 3 Traceback token=abc123')

    result = compare("A", agent_factories={AgentType.SIMPLE_REFLEX: boom, AgentType.LLM: boom})
    assert [e.status for e in result.results].count(AgentStatus.SUCCESS) == 4
    assert {f.agent_type for f in result.errors} == {AgentType.SIMPLE_REFLEX, AgentType.LLM}
    for failure in result.errors:
        for leaked in ("Traceback", "secret.py", "C:\\", "abc123"):
            assert leaked not in failure.message
    assert result.entry(AgentType.LLM).error_message.startswith("LLM Agent unavailable in the selected runtime mode.")


def test_an_agent_that_modifies_the_scenario_is_reported_without_hiding_the_others() -> None:
    class Mutator(BaseAgent):
        name = "mutator"

        def decide(self, scenario: Any) -> AgentDecision:
            scenario.failed_login_count += 1
            return SimpleReflexAgent().decide(scenario)

    scenario = get_scenario("A")
    before = scenario_fingerprint(scenario)
    result = mock_service(
        agent_factories={AgentType.GOAL_BASED: lambda s: Mutator()}).compare(scenario)
    assert result.entry(AgentType.GOAL_BASED).status is AgentStatus.ERROR
    assert "ScenarioModifiedError" in result.entry(AgentType.GOAL_BASED).error_message
    assert len(result.successful) == 5  # the mutation never reached the other agents
    assert scenario_fingerprint(scenario) == before == result.scenario_fingerprint
    assert {e.input_fingerprint for e in result.results} == {before}


def test_an_agent_returning_the_wrong_agent_type_is_rejected() -> None:
    result = compare("A", agent_factories={AgentType.GOAL_BASED: lambda s: SimpleReflexAgent()})
    assert result.entry(AgentType.GOAL_BASED).status is AgentStatus.ERROR


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_cs_15_every_successful_result_preserves_the_original_agent_decision(scenario_id: str) -> None:
    scenario = get_scenario(scenario_id)
    result = mock_service().compare(scenario)
    direct = {
        AgentType.SIMPLE_REFLEX: SimpleReflexAgent().decide(scenario),
        AgentType.MODEL_BASED_REFLEX: ModelBasedReflexAgent().decide(scenario),
        AgentType.GOAL_BASED: GoalBasedAgent().decide(scenario),
        AgentType.UTILITY_BASED: UtilityBasedAgent().decide(scenario),
        AgentType.LEARNING: LearningAgent(ExperienceRepository(EXPERIENCE)).decide(scenario),
    }
    for agent_type, expected in direct.items():
        assert result.entry(agent_type).decision_record == expected
    llm_record = result.entry(AgentType.LLM).decision_record
    assert isinstance(llm_record, AgentDecision) and llm_record.agent_type is AgentType.LLM
    for entry in result.successful:  # the displayed fields mirror the preserved record exactly
        record = entry.decision_record
        assert (entry.priority, entry.decision, entry.uncertainty, entry.audit_note) == (
            record.priority, record.decision, record.uncertainty, record.audit_note)
        assert list(entry.evidence) == record.evidence and list(entry.recommended_actions) == record.recommended_actions


def test_outcomes_come_from_agent_execution_not_from_hard_coded_values() -> None:
    changed = get_scenario("B").model_copy(update={
        "severity": get_scenario("C").severity, "asset_criticality": get_scenario("C").asset_criticality})
    results = mock_service(agent_factories={
        AgentType.LLM: lambda s: SimpleReflexAgent()}).compare(changed)
    assert results.entry(AgentType.SIMPLE_REFLEX).decision_record == SimpleReflexAgent().decide(changed)
    source = Path(comparison_module.__file__).read_text(encoding="utf-8")
    for outcome in ("ESCALATE_TO_SECURITY_ANALYST", "COLLECT_MORE_EVIDENCE", "INVESTIGATE"):
        assert outcome not in source


def test_execution_time_is_recorded_for_every_agent() -> None:
    assert all(e.execution_time_ms is not None and e.execution_time_ms >= 0 for e in compare().results)


def test_the_comparison_opens_no_network_connection(no_network: None) -> None:
    assert compare("C").all_succeeded


def test_comparison_models_reject_inconsistent_data() -> None:
    result = compare("A")
    entry = result.entry(AgentType.SIMPLE_REFLEX)
    with pytest.raises(ValidationError):
        AgentComparisonEntry(**{**entry.model_dump(), "evidence": ()})  # SUCCESS needs evidence
    with pytest.raises(ValidationError):
        AgentComparisonEntry(**{**entry.model_dump(), "status": AgentStatus.ERROR, "error_message": None})
    with pytest.raises(ValidationError):
        AgentComparisonResult(**{**result.model_dump(), "results": (entry, entry)})


def test_safe_error_message_and_redaction() -> None:
    text = 'Traceback (most recent call last): File "C:\\Users\\me\\x.py" api_key=abc /home/me/data'
    cleaned = redact_sensitive_text(text)
    assert "C:\\" not in cleaned and "abc" not in cleaned and "/home/me" not in cleaned and "Traceback" not in cleaned
    assert "unavailable" in safe_error_message(AgentType.LLM, LLMConnectionError("x")).lower()
    assert "RuntimeError" in safe_error_message(AgentType.GOAL_BASED, RuntimeError("secret C:\\x"))
    assert "secret" not in safe_error_message(AgentType.GOAL_BASED, RuntimeError("secret C:\\x"))


def test_live_status_is_never_claimed_without_explicit_enablement() -> None:
    assert live_ollama_status({})[0] is False
    assert live_ollama_status({"RUN_OLLAMA_INTEGRATION": "false"})[0] is False
    assert live_ollama_status({"RUN_OLLAMA_INTEGRATION": "true"})[0] is False  # no model configured
    assert live_ollama_status({"RUN_OLLAMA_INTEGRATION": "true", "OLLAMA_MODEL": "example"})[0] is True
    bad = {"RUN_OLLAMA_INTEGRATION": "true", "OLLAMA_MODEL": "example", "OLLAMA_BASE_URL": "http://203.0.113.9:11434"}
    assert live_ollama_status(bad)[0] is False  # non-local host rejected


# ------------------------------------------------------------------------------ EO: required output


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
@pytest.mark.parametrize("agent_type", AGENT_ORDER)
def test_eo_01_to_06_every_agent_provides_every_assignment_field(agent_type: AgentType, scenario_id: str) -> None:
    entry = compare(scenario_id).entry(agent_type)
    assert entry.status is AgentStatus.SUCCESS
    assert entry.priority and entry.decision and entry.reasoning_summary.strip()
    assert entry.evidence and all(item.strip() for item in entry.evidence)
    assert entry.uncertainty.strip() and entry.likely_interpretation.strip()
    assert entry.recommended_actions and all(step.strip() for step in entry.recommended_actions)
    assert entry.audit_note.strip()


# ------------------------------------------------------------------------------ LD: learning demo


def test_ld_01_temporary_learning_store_is_created() -> None:
    demo = LearningFeedbackDemo()
    try:
        assert demo.store_path.exists() and demo.store_path.resolve() != EXPERIENCE.resolve()
        assert DATA_DIR.resolve() not in demo.store_path.resolve().parents
        assert demo.feedback_count == 0
    finally:
        demo.close()
    assert not demo.store_path.exists()


def test_ld_02_committed_experience_hash_is_unchanged() -> None:
    before = sha(EXPERIENCE)
    demo = LearningFeedbackDemo()
    try:
        scenario = get_scenario("A")
        for _ in range(5):
            demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
        demo.record_feedback(scenario, FeedbackOutcome.SUCCESS)
        demo.reset()
    finally:
        demo.close()
    assert sha(EXPERIENCE) == before


def test_ld_03_failure_increases_the_failure_count() -> None:
    demo = LearningFeedbackDemo()
    try:
        step = demo.record_feedback(get_scenario("A"), FeedbackOutcome.FAILURE)
        assert (step.after_failure, step.after_success) == (step.before_failure + 1, step.before_success)
        assert step.after_smoothed_rate < step.before_smoothed_rate and step.outcome == "FAILURE"
    finally:
        demo.close()


def test_ld_04_success_increases_the_success_count() -> None:
    demo = LearningFeedbackDemo()
    try:
        step = demo.record_feedback(get_scenario("A"), FeedbackOutcome.SUCCESS)
        assert (step.after_success, step.after_failure) == (step.before_success + 1, step.before_failure)
        assert step.after_smoothed_rate > step.before_smoothed_rate
    finally:
        demo.close()


def test_ld_05_the_same_scenario_is_reused_and_unchanged() -> None:
    scenario = get_scenario("A")
    fingerprint = scenario_fingerprint(scenario)
    demo = LearningFeedbackDemo()
    try:
        for _ in range(4):
            demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
        assert {s.scenario_id for s in demo.steps} == {"A"} and scenario_fingerprint(scenario) == fingerprint
    finally:
        demo.close()


def test_ld_06_decision_eventually_changes_in_scenario_a_after_enough_failures() -> None:
    scenario = get_scenario("A")
    demo = LearningFeedbackDemo()
    try:
        assert demo.decide(scenario).decision == "INVESTIGATE"
        for _ in range(3):
            assert demo.record_feedback(scenario, FeedbackOutcome.FAILURE).decision_after == "INVESTIGATE"
        last = demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
        assert (last.decision_before, last.decision_after) == ("INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST")
        assert demo.decide(scenario).decision == "ESCALATE_TO_SECURITY_ANALYST"
    finally:
        demo.close()


def test_ld_07_reset_restores_the_demonstration_seed() -> None:
    scenario = get_scenario("A")
    demo = LearningFeedbackDemo()
    try:
        original = demo.decide(scenario).metadata["candidate_experience"]
        for _ in range(4):
            demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
        assert demo.decide(scenario).decision != "INVESTIGATE"
        demo.reset()
        assert demo.feedback_count == 0 and demo.steps == ()
        assert demo.decide(scenario).metadata["candidate_experience"] == original
        assert demo.decide(scenario).decision == "INVESTIGATE"
    finally:
        demo.close()


def test_ld_08_closing_and_restarting_never_modifies_the_committed_seed() -> None:
    before = data_hashes()
    for _ in range(2):
        demo = LearningFeedbackDemo()
        demo.record_feedback(get_scenario("A"), FeedbackOutcome.FAILURE)
        demo.close()
    assert data_hashes() == before


def test_the_demo_does_not_change_the_main_comparison() -> None:
    baseline = compare("A").entry(AgentType.LEARNING).special_details["candidate_experience"]
    demo = LearningFeedbackDemo()
    try:
        for _ in range(4):
            demo.record_feedback(get_scenario("A"), FeedbackOutcome.FAILURE)
        assert compare("A").entry(AgentType.LEARNING).decision == "INVESTIGATE"
        assert compare("A").entry(AgentType.LEARNING).special_details["candidate_experience"] == baseline
    finally:
        demo.close()


# ------------------------------------------------------------------------------ MD: model-based demo


def test_md_01_a_fresh_demo_agent_starts_empty() -> None:
    demo = ModelBasedMemoryDemo()
    assert demo.processed_scenario_count == 0 and demo.observations == ()


def test_md_02_first_observation_of_a_is_high_investigate() -> None:
    first = ModelBasedMemoryDemo().observe(get_scenario("A"))
    assert (first.priority.value, first.decision, first.account_observations) == ("HIGH", "INVESTIGATE", 1)
    assert first.previous_decision is None


def test_md_03_second_observation_with_the_same_instance_escalates() -> None:
    demo, scenario = ModelBasedMemoryDemo(), get_scenario("A")
    demo.observe(scenario)
    second = demo.observe(scenario)
    assert (second.priority.value, second.decision) == ("CRITICAL", "ESCALATE_TO_SECURITY_ANALYST")
    assert second.account_observations == 2 and second.previous_decision == "INVESTIGATE"
    assert demo.processed_scenario_count == 2


def test_md_04_the_main_comparison_is_unaffected_by_the_demo() -> None:
    demo, scenario = ModelBasedMemoryDemo(), get_scenario("A")
    demo.observe(scenario)
    demo.observe(scenario)
    entry = mock_service().compare(scenario).entry(AgentType.MODEL_BASED_REFLEX)
    assert (entry.priority.value, entry.decision) == ("HIGH", "INVESTIGATE")
    assert entry.special_details["state_summary"]["processed_scenario_count"] == 1


def test_md_05_reset_creates_fresh_state() -> None:
    demo, scenario = ModelBasedMemoryDemo(), get_scenario("A")
    demo.observe(scenario)
    demo.observe(scenario)
    demo.reset()
    assert demo.processed_scenario_count == 0 and demo.observations == ()
    assert demo.observe(scenario).decision == "INVESTIGATE"
