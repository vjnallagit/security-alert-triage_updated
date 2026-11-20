"""Evidence collection and deterministic grounding validation for the LLM agent.

EvidenceCollector keeps what the tools actually returned, with provenance (which call, which status,
which source records). Only a SUCCESSFUL call whose result concerns the scenario's own alert, asset,
policy or account counts as evidence. ERROR, INVALID_INPUT, AMBIGUOUS, unknown-tool and duplicate
results are recorded but never treated as factual evidence; a NOT_FOUND or an empty search is kept
as an explicit "nothing found" observation.

validate_grounding then checks every claim the model made against those observations. It never
fills in or invents anything: it only accepts or rejects.
"""

import json
import re
from collections.abc import Mapping
from typing import Any

from models.llm_models import (
    CallStatus,
    GroundedEvidenceItem,
    GuardrailResult,
    LLMToolResultRecord,
    LLMTriageRecommendation,
    ToolObservation,
)
from models.schemas import SecurityScenario

# Which asserted facts each tool can legitimately support, and the evidence field that holds each.
_FACT_FIELDS: dict[str, dict[str, str]] = {
    "search_authentication_events": {
        "failed_login_count": "failed_login_count",
        "successful_login_count": "successful_login_count",
        "matched_count": "matched_count",
    },
    "get_asset_criticality": {"asset_criticality": "criticality"},
    "get_alert_details": {"severity": "severity", "confirmed_impact": "confirmed_impact"},
    "get_response_policy": {
        "automatic_disruptive_action_permitted": "automatic_disruptive_action_permitted",
        "human_authorization_required": "human_authorization_required",
    },
}
_FAILED_COUNT = re.compile(r"\b(\d+)\s+(?:failed|failures?)\b", re.IGNORECASE)
_SUCCESS_COUNT = re.compile(r"\b(\d+)\s+(?:successful|successes|success)\b", re.IGNORECASE)


def call_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Stable identity of a call, so identical repeats can be recognised."""
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"


class EvidenceCollector:
    """Per-run store of tool observations. Create one per run; it holds no state between runs."""

    def __init__(self, scenario: SecurityScenario, minimum_distinct_tools: int = 2) -> None:
        self._scenario = scenario
        self._minimum = minimum_distinct_tools
        self._observations: dict[str, ToolObservation] = {}
        self._signatures: set[str] = set()

    # --- recording -------------------------------------------------------------------------

    def has_executed(self, tool_name: str, arguments: Mapping[str, Any]) -> bool:
        return call_signature(tool_name, arguments) in self._signatures

    def record(
        self, tool_call_id: str, tool_name: str, arguments: Mapping[str, Any], payload: Mapping[str, Any],
        status: CallStatus,
    ) -> ToolObservation:
        evidence = payload.get("evidence")
        observation = ToolObservation(
            tool_call_id=tool_call_id, tool_name=tool_name, status=status, arguments=dict(arguments),
            source_record_ids=tuple(payload.get("source_record_ids") or ()), evidence=evidence,
            summary=str(payload.get("summary") or ""), warnings=tuple(payload.get("warnings") or ()),
            limitations=tuple(payload.get("limitations") or ()),
            is_relevant=self._is_relevant(tool_name, arguments, evidence),
            is_absence=status is CallStatus.NOT_FOUND
            or (status is CallStatus.SUCCESS and tool_name == "search_authentication_events"
                and (evidence or {}).get("matched_count") == 0),
        )
        self._observations[tool_call_id] = observation
        if status is not CallStatus.DUPLICATE_CALL:
            self._signatures.add(call_signature(tool_name, arguments))
        return observation

    def _is_relevant(self, tool_name: str, arguments: Mapping[str, Any], evidence: Mapping[str, Any] | None) -> bool:
        s, evidence = self._scenario, evidence or {}
        if tool_name == "get_alert_details":
            return arguments.get("alert_id") == s.alert_id
        if tool_name == "get_asset_criticality":
            return arguments.get("asset_id") == s.asset_id
        if tool_name == "get_response_policy":
            return s.policy_id in (arguments.get("policy_id"), evidence.get("policy_id"))
        if tool_name == "search_authentication_events":
            names_scenario_entity = (
                arguments.get("account") == s.target_account or arguments.get("source") == s.source
                or arguments.get("asset_id") == s.asset_id or arguments.get("alert_id") == s.alert_id
                or arguments.get("scenario_id") == s.scenario_id
            )
            returns_scenario_events = any(e.get("alert_id") == s.alert_id for e in evidence.get("events") or ())
            return names_scenario_entity or returns_scenario_events
        return False

    # --- reading ---------------------------------------------------------------------------

    @property
    def observations(self) -> tuple[ToolObservation, ...]:
        return tuple(self._observations.values())

    def get(self, tool_call_id: str) -> ToolObservation | None:
        return self._observations.get(tool_call_id)

    @property
    def called_tool_names(self) -> tuple[str, ...]:
        """Names of tools that were actually requested (in first-use order)."""
        return tuple(dict.fromkeys(o.tool_name for o in self._observations.values()))

    @property
    def evidence_tool_names(self) -> tuple[str, ...]:
        """Distinct tools with at least one successful, relevant call."""
        return tuple(dict.fromkeys(o.tool_name for o in self._observations.values() if o.counts_as_evidence))

    @property
    def source_record_ids(self) -> tuple[str, ...]:
        """Deduplicated source IDs from evidence-bearing calls only, in first-seen order."""
        ids = (i for o in self._observations.values() if o.counts_as_evidence for i in o.source_record_ids)
        return tuple(dict.fromkeys(ids))

    def has_minimum_evidence(self) -> bool:
        return len(self.evidence_tool_names) >= self._minimum

    def compact_context(self) -> dict[str, object]:
        """Compact grounded evidence for the final LLM decision stage.

        Only successful, relevant tool observations are included. Provenance
        remains attached to each observation so the application can later
        assemble and validate the full recommendation deterministically.
        """

        observations: list[dict[str, object]] = []

        for observation in self._observations.values():
            if not observation.counts_as_evidence:
                continue

            observations.append(
                {
                    "tool_name": observation.tool_name,
                    "tool_call_id": observation.tool_call_id,
                    "summary": observation.summary,
                    "evidence": observation.evidence or {},
                    "source_record_ids": list(
                        observation.source_record_ids
                    ),
                    "warnings": list(
                        observation.warnings
                    ),
                    "limitations": list(
                        observation.limitations
                    ),
                    "is_absence_observation": (
                        observation.is_absence
                    ),
                }
            )

        return {
            "scenario_id": self._scenario.scenario_id,
            "successful_tools": list(
                self.evidence_tool_names
            ),
            "source_record_ids": list(
                self.source_record_ids
            ),
            "observations": observations,
        }  
    def result_records(self) -> tuple[LLMToolResultRecord, ...]:
        return tuple(
            LLMToolResultRecord(
                tool_call_id=o.tool_call_id, tool_name=o.tool_name, status=o.status, summary=o.summary[:300],
                source_record_ids=o.source_record_ids, warnings=o.warnings, is_relevant=o.is_relevant,
                is_absence=o.is_absence,
            )
            for o in self._observations.values()
        )


def _check_item(item: GroundedEvidenceItem, observation: ToolObservation | None, scenario: SecurityScenario) -> dict[str, list[str]]:
    """Problems with one evidence item, grouped by the grounding check that found them."""
    problems: dict[str, list[str]] = {}

    def fail(check: str, message: str) -> None:
        problems.setdefault(check, []).append(message)

    if observation is None:
        fail("tool_calls_exist", f"evidence cites tool_call_id '{item.tool_call_id}', which never occurred")
        return problems
    if observation.tool_name != item.tool_name:
        fail("tool_names_match", f"evidence says '{item.tool_name}' but call '{item.tool_call_id}' was '{observation.tool_name}'")
        return problems

    usable_absence = item.is_absence_observation and observation.status in (CallStatus.NOT_FOUND, CallStatus.SUCCESS)
    if not observation.is_reliable and not (usable_absence and observation.is_absence):
        fail("reliable_status_only", f"call '{item.tool_call_id}' returned {observation.status.value}; "
                                     "that cannot support a factual claim")
        return problems
    if not observation.is_relevant:
        fail("relevant_tools_only", f"call '{item.tool_call_id}' ({item.tool_name}) does not concern this scenario's own records")
    unknown = sorted(set(item.source_record_ids) - set(observation.source_record_ids))
    if unknown:
        fail("source_ids_real", f"source IDs {unknown} were not returned by call '{item.tool_call_id}'")
    if item.is_absence_observation and not observation.is_absence:
        fail("absence_labelled_correctly", f"call '{item.tool_call_id}' returned records, so it is not a no-match observation")

    stated = item.asserted_facts.stated()
    allowed = _FACT_FIELDS.get(item.tool_name, {})
    evidence = observation.evidence or {}
    for fact, value in stated.items():
        if fact not in allowed:
            fail("asserted_facts_match", f"'{fact}' cannot be supported by {item.tool_name}")
        elif fact == "confirmed_impact" and value is True and scenario.confirmed_impact is False:
            fail("scenario_impact_not_rewritten", "evidence asserts confirmed impact, but the scenario states it is not confirmed")
        elif evidence.get(allowed[fact]) != value and not (fact == "confirmed_impact" and value is False):
            fail("asserted_facts_match", f"{fact}={value} contradicts the tool result ({evidence.get(allowed[fact])})")

    if item.tool_name == "search_authentication_events" and evidence:
        for pattern, key, label in ((_FAILED_COUNT, "failed_login_count", "failed"),
                                    (_SUCCESS_COUNT, "successful_login_count", "successful")):
            for match in pattern.finditer(item.claim):
                if int(match.group(1)) != evidence.get(key):
                    fail("asserted_facts_match",
                         f"claim says {match.group(1)} {label} events but the tool result has {evidence.get(key)}")
    return problems


def evaluate_grounding(
    recommendation: LLMTriageRecommendation, collector: EvidenceCollector, scenario: SecurityScenario,
    minimum_distinct_tools: int = 2,
) -> tuple[tuple[GuardrailResult, ...], tuple[str, ...]]:
    """Run every grounding check. Returns (per-check results, flat list of problems)."""
    problems: dict[str, list[str]] = {}
    for item in recommendation.evidence:
        for check, messages in _check_item(item, collector.get(item.tool_call_id), scenario).items():
            problems.setdefault(check, []).extend(messages)

    supporting = {
        item.tool_name for item in recommendation.evidence
        if (obs := collector.get(item.tool_call_id)) is not None and obs.tool_name == item.tool_name
        and obs.counts_as_evidence and not _check_item(item, obs, scenario)
    }
    if len(supporting) < minimum_distinct_tools:
        problems.setdefault("two_distinct_tools", []).append(
            f"only {len(supporting)} distinct tool(s) ground the recommendation; {minimum_distinct_tools} are required")

    real_sources = set(collector.source_record_ids)
    invented = sorted(set(recommendation.source_record_ids) - real_sources)
    if invented:
        problems.setdefault("source_ids_real", []).append(f"top-level source IDs {invented} were never returned by a tool")
    never_called = sorted(set(recommendation.tools_called) - set(collector.called_tool_names))
    if never_called:
        problems.setdefault("tools_called_real", []).append(f"tools_called lists {never_called}, which were never requested")

    names = ("tool_calls_exist", "tool_names_match", "reliable_status_only", "relevant_tools_only", "source_ids_real",
             "absence_labelled_correctly", "asserted_facts_match", "scenario_impact_not_rewritten",
             "two_distinct_tools", "tools_called_real")
    results = tuple(
        GuardrailResult(name=name, passed=name not in problems, detail="; ".join(problems.get(name, []))[:400])
        for name in names
    )
    return results, tuple(m for name in names for m in problems.get(name, []))
