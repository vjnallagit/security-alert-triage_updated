"""Orchestration (LO), minimum-tool (MT) and grounding (LG) tests. All use the scripted fake model."""

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config.llm_config import OllamaConfig
from models.llm_models import CallStatus, LLMTriageRecommendation
from services.evidence_collector import EvidenceCollector, evaluate_grounding
from services.llm_errors import (
    GroundingValidationError,
    InsufficientToolEvidenceError,
    LLMConnectionError,
    LLMIterationLimitError,
    LLMModelError,
    LLMOutputValidationError,
)
from services.llm_orchestrator import LLMOrchestrator, extract_json_object
from services.scenario_service import get_scenario
from tests.fakes.fake_tool_calling_model import (
    ScriptedChatModel,
    answer_step,
    happy_path_model,
    standard_calls,
    tool_calls_message,
)
from tools.langchain_security_tools import TOOL_NAMES, SecurityToolbox

A = get_scenario("A")


def run(model: ScriptedChatModel, scenario=A, config: OllamaConfig | None = None, **kwargs):
    return LLMOrchestrator(model, config, **kwargs).run(scenario)


def messages_of(model: ScriptedChatModel, turn: int) -> list[Any]:
    return model.invocations[turn]


def tool_messages(messages: list[Any]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def statuses(result) -> list[str]:
    return [c.status.value for c in result.trace.tool_calls]


# ---------------------------------------------------------------- LO: orchestration


def test_lo_01_model_receives_the_scenario_message() -> None:
    model = happy_path_model(A)
    run(model)
    first = messages_of(model, 0)
    assert isinstance(first[0], SystemMessage) and isinstance(first[1], HumanMessage)
    for value in (A.scenario_id, A.alert_id, A.asset_id, A.policy_id, A.target_account):
        assert value in first[1].content


def test_lo_01_expected_characteristics_are_not_leaked_to_the_model() -> None:
    model = happy_path_model(A)
    run(model)
    prompt = " ".join(str(m.content) for m in messages_of(model, 0))
    assert "expected_characteristics" not in prompt
    assert all(note not in prompt for note in A.expected_characteristics)


def test_lo_02_tool_results_are_not_preloaded_into_the_first_prompt() -> None:
    model = happy_path_model(A)
    run(model)
    first = messages_of(model, 0)
    assert tool_messages(first) == [] and len(first) == 2
    assert "source_record_ids" not in first[1].content and "AUTH-" not in first[1].content


def test_lo_03_tools_are_bound_to_the_model() -> None:
    model = happy_path_model(A)
    run(model)
    assert model.bound_tool_names == TOOL_NAMES


def test_lo_04_and_05_one_call_gives_one_tool_message_with_the_same_id() -> None:
    first, second, *rest = standard_calls(A)
    model = ScriptedChatModel([
        tool_calls_message((*first, "id-alpha")), tool_calls_message((*second, "id-beta")), answer_step(A)])
    result = run(model)
    after_first = tool_messages(messages_of(model, 1))
    assert [m.tool_call_id for m in after_first] == ["id-alpha"]
    assert [m.tool_call_id for m in tool_messages(messages_of(model, 2))] == ["id-alpha", "id-beta"]
    assert [c.tool_call_id for c in result.trace.tool_calls] == ["id-alpha", "id-beta"]


def test_lo_06_multiple_calls_in_one_response_all_execute() -> None:
    model = happy_path_model(A, parallel=True)
    result = run(model)
    assert len(tool_messages(messages_of(model, 1))) == 4
    assert result.trace.iterations == 2 and statuses(result) == ["SUCCESS"] * 4


def test_lo_07_sequential_calls_work() -> None:
    model = happy_path_model(A, parallel=False)
    result = run(model)
    assert result.trace.iterations == 5 and [c.iteration for c in result.trace.tool_calls] == [1, 2, 3, 4]


def test_lo_08_known_tools_dispatch_to_the_phase_6_service() -> None:
    toolbox = SecurityToolbox()
    result = run(happy_path_model(A), toolbox_factory=lambda: toolbox)
    logged = [entry.tool_name for entry in toolbox.service.get_invocation_log()]
    assert logged == [c.tool_name for c in result.trace.tool_calls] == [n for n, _ in standard_calls(A)]


def test_lo_09_unknown_tool_is_never_executed() -> None:
    toolbox = SecurityToolbox()
    calls = standard_calls(A)[:2]
    model = ScriptedChatModel([
        tool_calls_message(("disable_account", {"account": A.target_account}, "bad-1"), *calls), answer_step(A)])
    result = run(model, toolbox_factory=lambda: toolbox)
    assert statuses(result)[0] == "UNKNOWN_TOOL"
    assert [e.tool_name for e in toolbox.service.get_invocation_log()] == [n for n, _ in calls]
    seen = json.loads(tool_messages(messages_of(model, 1))[0].content)
    assert seen["status"] == "UNKNOWN_TOOL" and seen["executed_actions"] == []


@pytest.mark.parametrize("arguments", [{"alert_id": 5, "bogus": "x"}, {}])
def test_lo_10_malformed_arguments_return_safe_error_evidence(arguments) -> None:
    toolbox = SecurityToolbox()
    model = ScriptedChatModel([
        tool_calls_message(("get_alert_details", arguments, "bad-args"), *standard_calls(A)[1:3]), answer_step(A)])
    result = run(model, toolbox_factory=lambda: toolbox)
    assert statuses(result)[0] == "INVALID_INPUT"
    assert "get_alert_details" not in [e.tool_name for e in toolbox.service.get_invocation_log()]


def test_lo_10_invalid_tool_calls_from_the_model_are_handled_safely() -> None:
    broken = AIMessage(content="", tool_calls=[], invalid_tool_calls=[
        {"name": "get_alert_details", "args": "{not json", "id": "inv-1", "error": "bad json", "type": "invalid_tool_call"}])
    model = ScriptedChatModel([broken, tool_calls_message(*standard_calls(A)[1:3]), answer_step(A)])
    result = run(model)
    assert result.trace.tool_calls[0].status is CallStatus.INVALID_INPUT
    assert result.trace.tool_calls[0].tool_call_id == "inv-1"


def test_lo_11_duplicate_identical_calls_are_skipped_not_re_executed() -> None:
    toolbox = SecurityToolbox()
    first = standard_calls(A)
    model = ScriptedChatModel([
        tool_calls_message((*first[0], "c1"), (*first[0], "c2"), *first[1:3]), answer_step(A)])
    result = run(model, toolbox_factory=lambda: toolbox)
    assert statuses(result) == ["SUCCESS", "DUPLICATE_CALL", "SUCCESS", "SUCCESS"]
    assert len(toolbox.service.get_invocation_log()) == 3


def test_lo_12_execution_stops_at_the_iteration_limit() -> None:
    config = OllamaConfig(max_tool_iterations=2)
    calls = standard_calls(A)
    model = ScriptedChatModel([tool_calls_message(calls[0]), tool_calls_message(calls[1]), AssertionError("never reached")])
    with pytest.raises(LLMIterationLimitError) as caught:
        run(model, config=config)
    assert model.call_count == 2 and caught.value.trace.iterations == 2


def test_lo_13_final_output_is_parsed_through_pydantic() -> None:
    result = run(happy_path_model(A))
    assert isinstance(result.recommendation, LLMTriageRecommendation)
    assert result.recommendation.decision == "INVESTIGATE"


def test_lo_13_code_fences_and_think_blocks_are_stripped_but_prose_is_not_parsed() -> None:
    assert extract_json_object('<think>hmm</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(Exception):
        extract_json_object('Sure! Here is the answer: {"a": 1}')


def _fenced_answer(scenario=A, **overrides):
    inner = answer_step(scenario, **overrides)
    return lambda messages: AIMessage(content="<think>private</think>```json\n" + inner(messages).content + "\n```")


def test_lo_13_fenced_json_answer_is_accepted() -> None:
    model = ScriptedChatModel([tool_calls_message(*standard_calls(A)), _fenced_answer()])
    assert run(model).trace.final_output_valid


def test_lo_14_malformed_output_gets_exactly_one_repair_attempt() -> None:
    model = ScriptedChatModel([
        tool_calls_message(*standard_calls(A)), AIMessage(content="I think it is HIGH priority."), answer_step(A)])
    result = run(model)
    repairs = [m for m in messages_of(model, 2) if isinstance(m, HumanMessage) and "rejected" in m.content]
    assert len(repairs) == 1 and result.trace.final_output_valid
    assert any("repair" in w for w in result.trace.warnings)


def test_lo_15_persistent_malformed_output_fails_clearly() -> None:
    model = ScriptedChatModel([
        tool_calls_message(*standard_calls(A)), AIMessage(content="not json"), AIMessage(content="still not json")])
    with pytest.raises(LLMOutputValidationError, match="still invalid") as caught:
        run(model)
    assert model.call_count == 3 and caught.value.trace.final_output_valid is False


def test_lo_15_schema_violation_is_repaired_or_fails() -> None:
    def drop_priority(answer: dict) -> None:
        del answer["priority"]
    model = ScriptedChatModel([
        tool_calls_message(*standard_calls(A)), answer_step(A, mutate=drop_priority), answer_step(A, mutate=drop_priority)])
    with pytest.raises(LLMOutputValidationError):
        run(model)


def test_lo_16_model_exceptions_fail_safely_with_mapped_errors() -> None:
    with pytest.raises(LLMModelError):
        run(ScriptedChatModel([RuntimeError("boom")]))
    with pytest.raises(LLMConnectionError):
        run(ScriptedChatModel([ConnectionError("refused")]))
    with pytest.raises(LLMConnectionError):
        run(ScriptedChatModel([TimeoutError("slow")]))


def test_lo_17_no_raw_stack_trace_or_path_reaches_the_user() -> None:
    secret = RuntimeError('File "C:\\Users\\someone\\secret.py", line 3 Traceback boom')
    with pytest.raises(LLMModelError) as caught:
        run(ScriptedChatModel([secret]))
    shown = str(caught.value) + json.dumps(caught.value.trace.model_dump(mode="json"))
    assert "Traceback" not in shown and "secret.py" not in shown and "C:\\" not in shown


def test_lo_18_trace_has_observable_calls_not_reasoning() -> None:
    model = ScriptedChatModel([tool_calls_message(*standard_calls(A)), _fenced_answer()])
    dump = json.dumps(run(model).trace.model_dump(mode="json"))
    assert "private" not in dump  # the <think> content never enters the trace
    for forbidden in ("reasoning", "chain_of_thought", "thought"):
        assert forbidden not in dump
    assert "get_alert_details" in dump and "tool_call_id" in dump


def test_orchestrator_is_reusable_and_keeps_no_state_between_runs() -> None:
    orchestrator = LLMOrchestrator(happy_path_model(A))
    first = orchestrator.run(A)
    second_model = happy_path_model(A)
    second = LLMOrchestrator(second_model).run(A)
    assert [c.tool_name for c in first.trace.tool_calls] == [c.tool_name for c in second.trace.tool_calls]
    assert first.trace.run_id != second.trace.run_id


def test_missing_tool_call_id_is_synthesised_with_a_warning() -> None:
    message = AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": None, "type": "tool_call"} for name, args in standard_calls(A)])
    result = run(ScriptedChatModel([message, answer_step(A)]))
    assert len({c.tool_call_id for c in result.trace.tool_calls}) == 4 and result.trace.warnings


# ---------------------------------------------------------------- MT: minimum tool requirement


def _insufficient(*first_turns: Any) -> InsufficientToolEvidenceError:
    model = ScriptedChatModel([*first_turns, answer_step(A), answer_step(A)])
    with pytest.raises(InsufficientToolEvidenceError) as caught:
        run(model)
    return caught.value


def test_mt_01_zero_tool_calls_cannot_succeed() -> None:
    assert _insufficient()


def test_mt_02_one_tool_call_cannot_succeed() -> None:
    assert _insufficient(tool_calls_message(standard_calls(A)[0]))


def test_mt_03_two_calls_to_the_same_tool_do_not_satisfy_distinct_tools() -> None:
    search = ("search_authentication_events", {"alert_id": A.alert_id})
    other = ("search_authentication_events", {"account": A.target_account})
    assert _insufficient(tool_calls_message(search, other))


def test_mt_04_two_distinct_relevant_tools_satisfy_the_minimum() -> None:
    calls = [standard_calls(A)[0], standard_calls(A)[2]]
    model = ScriptedChatModel([tool_calls_message(*calls), answer_step(A)])
    result = run(model)
    assert result.collector.has_minimum_evidence() and result.trace.final_output_valid


def test_mt_05_irrelevant_tool_call_does_not_count() -> None:
    other = get_scenario("B")
    calls = [standard_calls(A)[0], ("get_asset_criticality", {"asset_id": other.asset_id})]
    assert _insufficient(tool_calls_message(*calls))


def test_mt_06_insufficient_evidence_earns_exactly_one_bounded_correction() -> None:
    model = ScriptedChatModel([answer_step(A), tool_calls_message(*standard_calls(A)), answer_step(A)])
    result = run(model)
    corrections = [
        m
        for m in messages_of(model, 1)
        if isinstance(m, HumanMessage)
        and "successful tool evidence" in m.content
        and "Do not return the final decision yet" in m.content
    ]

    assert len(corrections) == 1
    assert result.trace.final_output_valid


def test_mt_07_second_failure_raises_insufficient_tool_evidence() -> None:
    model = ScriptedChatModel([AIMessage(content="done"), AIMessage(content="still done")])
    with pytest.raises(InsufficientToolEvidenceError):
        run(model)
    assert model.call_count == 2


def test_mt_08_successful_result_lists_the_actual_tools_called() -> None:
    result = run(happy_path_model(A))
    assert set(result.recommendation.tools_called) == {c.tool_name for c in result.trace.tool_calls}
    assert set(result.recommendation.tools_called) <= set(TOOL_NAMES)


# ---------------------------------------------------------------- LG: grounding


def grounded(scenario_id: str = "A"):
    scenario = get_scenario(scenario_id)
    result = run(happy_path_model(scenario), scenario)
    return scenario, result.recommendation, result.collector


def mutated(recommendation: LLMTriageRecommendation, change) -> LLMTriageRecommendation:
    data = recommendation.model_dump(mode="json")
    change(data)
    return LLMTriageRecommendation.model_validate(data)


def failures(recommendation, collector, scenario) -> set[str]:
    results, _ = evaluate_grounding(recommendation, collector, scenario)
    return {r.name for r in results if not r.passed}


def item_for(data: dict, tool_name: str) -> dict:
    return next(i for i in data["evidence"] if i["tool_name"] == tool_name)


def test_lg_01_every_evidence_item_references_an_actual_call() -> None:
    scenario, recommendation, collector = grounded()
    assert failures(recommendation, collector, scenario) == set()
    assert all(collector.get(item.tool_call_id) is not None for item in recommendation.evidence)


def test_lg_02_invented_tool_call_id_is_rejected() -> None:
    scenario, recommendation, collector = grounded()
    bad = mutated(recommendation, lambda d: d["evidence"][0].update(tool_call_id="call_made_up"))
    assert "tool_calls_exist" in failures(bad, collector, scenario)


def test_lg_03_invented_source_id_is_rejected() -> None:
    scenario, recommendation, collector = grounded()
    bad_item = mutated(recommendation, lambda d: d["evidence"][0]["source_record_ids"].append("AUTH-FAKE-999"))
    bad_top = mutated(recommendation, lambda d: d["source_record_ids"].append("AUTH-FAKE-999"))
    assert "source_ids_real" in failures(bad_item, collector, scenario)
    assert "source_ids_real" in failures(bad_top, collector, scenario)


def test_lg_04_wrong_tool_or_source_association_is_rejected() -> None:
    scenario, recommendation, collector = grounded()

    def swap_sources(d: dict) -> None:
        item_for(d, "get_alert_details")["source_record_ids"] = list(item_for(d, "get_asset_criticality")["source_record_ids"])

    def rename_tool(d: dict) -> None:
        item_for(d, "get_alert_details")["tool_name"] = "get_asset_criticality"

    assert "source_ids_real" in failures(mutated(recommendation, swap_sources), collector, scenario)
    assert "tool_names_match" in failures(mutated(recommendation, rename_tool), collector, scenario)


def _collector_with(scenario, tool_name: str, arguments: dict, payload: dict, status: CallStatus) -> EvidenceCollector:
    collector = EvidenceCollector(scenario)
    collector.record("call-x", tool_name, arguments, payload, status)
    return collector


def _item_recommendation(tool_name: str, sources: list[str], **item_overrides) -> LLMTriageRecommendation:
    item = {"claim": "A claim.", "tool_name": tool_name, "tool_call_id": "call-x", "source_record_ids": sources}
    item.update(item_overrides)
    _, base, _ = grounded()
    return mutated(base, lambda d: d.update(evidence=[item], tools_called=[tool_name], source_record_ids=sources or ["X"]))


def test_lg_05_error_result_cannot_support_a_factual_claim() -> None:
    payload = {"status": "ERROR", "summary": "failed", "evidence": None, "source_record_ids": ["ASSET-IDP-01"]}
    collector = _collector_with(A, "get_asset_criticality", {"asset_id": A.asset_id}, payload, CallStatus.ERROR)
    recommendation = _item_recommendation("get_asset_criticality", ["ASSET-IDP-01"])
    assert "reliable_status_only" in failures(recommendation, collector, A)


def test_lg_06_ambiguous_result_cannot_be_selected_evidence() -> None:
    execution = SecurityToolbox().execute("get_response_policy", {"severity": "HIGH"})
    assert execution.status is CallStatus.AMBIGUOUS
    collector = _collector_with(A, "get_response_policy", execution.arguments, execution.payload, execution.status)
    recommendation = _item_recommendation("get_response_policy", ["POL-PRIV-LOGIN-001"], tool_call_id="call-x")
    assert "reliable_status_only" in failures(recommendation, collector, A)
    assert not collector.has_minimum_evidence() and collector.evidence_tool_names == ()


def test_lg_07_authentication_count_exceeding_the_result_is_rejected() -> None:
    scenario, recommendation, collector = grounded()
    asserted = mutated(recommendation, lambda d: item_for(d, "search_authentication_events")["asserted_facts"].update(failed_login_count=20))
    claimed = mutated(recommendation, lambda d: item_for(d, "search_authentication_events").update(claim="20 failed logins occurred."))
    assert "asserted_facts_match" in failures(asserted, collector, scenario)
    assert "asserted_facts_match" in failures(claimed, collector, scenario)


def test_lg_08_wrong_asset_criticality_is_rejected() -> None:
    scenario, recommendation, collector = grounded()
    wrong = mutated(recommendation, lambda d: item_for(d, "get_asset_criticality")["asserted_facts"].update(asset_criticality="LOW"))
    assert "asserted_facts_match" in failures(wrong, collector, scenario)


def test_lg_09_false_policy_permission_is_rejected() -> None:
    scenario, recommendation, collector = grounded()
    false_permission = mutated(recommendation, lambda d: item_for(d, "get_response_policy")["asserted_facts"].update(
        automatic_disruptive_action_permitted=True))
    truthful = mutated(recommendation, lambda d: item_for(d, "get_response_policy")["asserted_facts"].update(
        automatic_disruptive_action_permitted=False, human_authorization_required=True))
    assert "asserted_facts_match" in failures(false_permission, collector, scenario)
    assert failures(truthful, collector, scenario) == set()


def test_lg_10_two_distinct_tools_must_ground_the_recommendation() -> None:
    scenario, recommendation, collector = grounded()
    only_one = mutated(recommendation, lambda d: d.update(evidence=[item_for(d, "get_alert_details")]))
    assert "two_distinct_tools" in failures(only_one, collector, scenario)


def test_lg_11_deduplicated_source_list_contains_only_real_sources() -> None:
    _, recommendation, collector = grounded()
    assert len(set(recommendation.source_record_ids)) == len(recommendation.source_record_ids)
    assert set(recommendation.source_record_ids) <= set(collector.source_record_ids)


def test_lg_12_no_match_observation_stays_explicitly_labelled() -> None:
    toolbox = SecurityToolbox()
    empty = toolbox.execute("search_authentication_events", {"alert_id": A.alert_id, "success": True})
    asset = toolbox.execute("get_asset_criticality", {"asset_id": A.asset_id})
    assert empty.payload["evidence"]["matched_count"] == 0
    collector = EvidenceCollector(A)
    collector.record("call-empty", "search_authentication_events", empty.arguments, empty.payload, empty.status)
    collector.record("call-asset", "get_asset_criticality", asset.arguments, asset.payload, asset.status)
    assert collector.get("call-empty").is_absence

    def build(absence_flag: bool) -> LLMTriageRecommendation:
        _, base, _ = grounded()
        items = [
            {"claim": "No successful logins were found.", "tool_name": "search_authentication_events",
             "tool_call_id": "call-empty", "source_record_ids": [], "is_absence_observation": absence_flag},
            {"claim": "The asset is HIGH criticality.", "tool_name": "get_asset_criticality",
             "tool_call_id": "call-asset", "source_record_ids": ["ASSET-IDP-01"]},
        ]
        return mutated(base, lambda d: d.update(
            evidence=items, tools_called=["search_authentication_events", "get_asset_criticality"],
            source_record_ids=["ASSET-IDP-01"]))

    assert failures(build(True), collector, A) == set()
    # A "nothing found" claim that is not labelled as an absence observation must cite records, and is rejected.
    with pytest.raises(ValueError):
        build(False)

    _, base, full_collector = grounded()
    mislabelled = mutated(base, lambda d: item_for(d, "search_authentication_events").update(is_absence_observation=True))
    assert "absence_labelled_correctly" in failures(mislabelled, full_collector, A)


def test_lg_13_scenario_labels_do_not_bypass_grounding() -> None:
    relabelled = A.model_copy(update={"scenario_id": "B", "title": "Totally benign"})
    result = run(happy_path_model(relabelled), relabelled)
    assert all(g.passed for g in result.grounding_results)  # grounding follows the records, not the label

    scenario, recommendation, collector = grounded("A")
    other = get_scenario("B")
    replay = EvidenceCollector(other)  # the same calls, judged against a different scenario's own records
    for o in collector.observations:
        payload = {"evidence": o.evidence, "source_record_ids": o.source_record_ids, "summary": o.summary}
        replay.record(o.tool_call_id, o.tool_name, o.arguments, payload, o.status)
    assert "relevant_tools_only" in failures(recommendation, replay, other)


def test_grounding_failure_gets_one_repair_then_succeeds() -> None:
    steps = [tool_calls_message(*standard_calls(A)),
             answer_step(A, mutate=lambda a: a["evidence"][0].update(tool_call_id="call_made_up")), answer_step(A)]
    result = run(ScriptedChatModel(steps))
    assert result.trace.final_output_valid and any("repair" in w for w in result.trace.warnings)


def test_persistent_grounding_failure_raises_grounding_error() -> None:
    bad = answer_step(A, mutate=lambda a: a["evidence"][0].update(tool_call_id="call_made_up"))
    with pytest.raises(GroundingValidationError):
        run(ScriptedChatModel([tool_calls_message(*standard_calls(A)), bad, bad]))


@pytest.mark.parametrize("scenario_id", ["A", "B", "C"])
def test_all_scenarios_ground_cleanly_in_both_call_styles(scenario_id: str) -> None:
    scenario = get_scenario(scenario_id)
    for parallel in (True, False):
        result = run(happy_path_model(scenario, parallel=parallel), scenario)
        assert all(g.passed for g in result.grounding_results + result.guardrail_results)
