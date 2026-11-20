"""Model-Based Reflex Agent: prior state + percept -> updated state -> ordered rules.

Stateful observation is not the same as learning. The Model-Based Reflex Agent
updates its internal representation of the environment, while the future Learning
Agent will update decisions using outcome feedback. Here the rules and thresholds
never change, no success/failure feedback is processed, and no file is read or
written.

Order of work inside decide():
  1. receive the current SecurityScenario
  2. take the prior state (the agent's private copy)
  3. update the model from the percept (transition function, returns a new state)
  4. apply ordered condition-action rules to the percept AND the updated state
  5. record the resulting priority and decision in the state
  6. return the AgentDecision
The new state replaces the old one only after step 6 succeeds.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agents.base import BaseAgent
from agents.common import (
    MULTIPLE_CORRELATED_ALERTS_THRESHOLD,
    RECOMMENDED_ACTIONS,
    REPEATED_FAILED_LOGIN_THRESHOLD,
    TriageDecision,
    audit_note,
    safety_metadata,
    scenario_evidence,
    uncertainty_statement,
)
from models.agent_state import ModelBasedState
from models.schemas import (
    AccountPrivilege,
    AgentDecision,
    AgentType,
    AssetCriticality,
    EvidenceStrength,
    Priority,
    SecurityScenario,
    Severity,
    SourceFamiliarity,
)

AGENT_NAME = "Model-Based Reflex Agent"
DECISION_MECHANISM = "internal_state_plus_condition_action_rules"


# --- Environment transition/update model -----------------------------------


def update_state(prior: ModelBasedState, scenario: SecurityScenario) -> ModelBasedState:
    """Return the state after observing `scenario`. The prior state is left untouched."""
    state = prior.model_copy(deep=True)
    state.processed_scenario_count += 1
    state.last_scenario_id = scenario.scenario_id
    state.last_alert_id = scenario.alert_id
    _increment(state.account_observations, scenario.target_account)
    _increment(state.asset_observations, scenario.asset_id)
    _increment(state.alert_type_observations, scenario.alert_type)
    if scenario.source_familiarity == SourceFamiliarity.UNFAMILIAR:
        state.unfamiliar_source_observations += 1
    if scenario.correlated_alert_count >= MULTIPLE_CORRELATED_ALERTS_THRESHOLD:
        state.correlated_risk_observations += 1
    return state


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


# --- Conditions: look at the percept and the UPDATED state ------------------


def _seen_before(s: SecurityScenario, state: ModelBasedState) -> bool:
    """The updated counters include the current percept, so more than 1 means seen earlier."""
    return (
        state.account_observations.get(s.target_account, 0) > 1
        or state.alert_type_observations.get(s.alert_type, 0) > 1
    )


def _critical_asset_with_correlated_unusual_access(
    s: SecurityScenario, state: ModelBasedState
) -> bool:
    return (
        s.asset_criticality == AssetCriticality.CRITICAL
        and s.unusual_access
        and s.correlated_alert_count >= MULTIPLE_CORRELATED_ALERTS_THRESHOLD
    )


def _privileged_threat_pattern(s: SecurityScenario) -> bool:
    return (
        s.account_privilege == AccountPrivilege.PRIVILEGED
        and s.source_familiarity == SourceFamiliarity.UNFAMILIAR
        and s.failed_login_count >= REPEATED_FAILED_LOGIN_THRESHOLD
    )


def _repeated_privileged_threat_pattern(s: SecurityScenario, state: ModelBasedState) -> bool:
    return _privileged_threat_pattern(s) and _seen_before(s, state)


def _first_privileged_threat_pattern(s: SecurityScenario, state: ModelBasedState) -> bool:
    return _privileged_threat_pattern(s) and not _seen_before(s, state)


def _weak_low_severity_alert(s: SecurityScenario) -> bool:
    return (
        s.severity == Severity.LOW
        and s.evidence_strength == EvidenceStrength.LIMITED
        and not s.confirmed_impact
    )


def _repeated_weak_low_severity_alert(s: SecurityScenario, state: ModelBasedState) -> bool:
    return _weak_low_severity_alert(s) and _seen_before(s, state)


def _first_weak_low_severity_alert(s: SecurityScenario, state: ModelBasedState) -> bool:
    return _weak_low_severity_alert(s) and not _seen_before(s, state)


def _high_or_critical_severity(s: SecurityScenario, state: ModelBasedState) -> bool:
    return s.severity in (Severity.HIGH, Severity.CRITICAL)


def _always(s: SecurityScenario, state: ModelBasedState) -> bool:
    return True


@dataclass(frozen=True)
class ModelBasedRule:
    rule_id: str
    name: str
    condition: Callable[[SecurityScenario, ModelBasedState], bool]
    priority: Priority
    decision: TriageDecision
    reasoning: str  # may use {account_count}, {asset_count}, {alert_type_count}
    interpretation: str


# Order matters: first match wins, so specific safety-relevant rules come first.
RULES: tuple[ModelBasedRule, ...] = (
    ModelBasedRule(
        "MB-01",
        "Critical asset with correlated alerts and unusual access",
        _critical_asset_with_correlated_unusual_access,
        Priority.CRITICAL,
        TriageDecision.ESCALATE_TO_SECURITY_ANALYST,
        "the current alert involves a critical asset, shows unusual access and has multiple "
        "correlated alerts. Potential consequence is high, although compromise is not confirmed",
        "Possible serious incident on a critical service; unconfirmed and needs human review.",
    ),
    ModelBasedRule(
        "MB-02",
        "Repeated privileged-account threat pattern",
        _repeated_privileged_threat_pattern,
        Priority.CRITICAL,
        TriageDecision.ESCALATE_TO_SECURITY_ANALYST,
        "the current alert targets a privileged account with repeated failed logins from an "
        "unfamiliar source, and the internal state shows this account or alert pattern was "
        "already observed ({account_count} observation(s) of this account, {alert_type_count} "
        "of this alert type). The persistence of the pattern raises concern",
        "A persistent, repeated attempt against a privileged account; compromise is unconfirmed.",
    ),
    ModelBasedRule(
        "MB-03",
        "First privileged-account threat pattern",
        _first_privileged_threat_pattern,
        Priority.HIGH,
        TriageDecision.INVESTIGATE,
        "the current alert targets a privileged account with repeated failed logins from an "
        "unfamiliar source, and the internal state shows this pattern has not been observed "
        "before",
        "Possible credential-guessing attempt against a privileged account; unconfirmed.",
    ),
    ModelBasedRule(
        "MB-04",
        "Repeated weak low-severity alert",
        _repeated_weak_low_severity_alert,
        Priority.LOW,
        TriageDecision.MONITOR,
        "the current alert has LOW severity, LIMITED evidence and no confirmed impact, and "
        "the internal state shows a similar alert was observed before without escalation "
        "(alert type seen {alert_type_count} time(s)); continued monitoring is proportionate",
        "Recurring low-risk noise; keep watching for change rather than escalate.",
    ),
    ModelBasedRule(
        "MB-05",
        "First weak low-severity alert",
        _first_weak_low_severity_alert,
        Priority.LOW,
        TriageDecision.COLLECT_MORE_EVIDENCE,
        "the current alert has LOW severity, LIMITED evidence and no confirmed impact, and "
        "the internal state shows this is the first such observation",
        "Likely benign or one-off event; evidence is too weak to conclude.",
    ),
    ModelBasedRule(
        "MB-06",
        "High or critical severity",
        _high_or_critical_severity,
        Priority.HIGH,
        TriageDecision.INVESTIGATE,
        "the current alert has HIGH or CRITICAL severity and no more specific rule matched",
        "Significant alert whose cause is not yet understood.",
    ),
    ModelBasedRule(
        "MB-DEFAULT",
        "Default",
        _always,
        Priority.MEDIUM,
        TriageDecision.COLLECT_MORE_EVIDENCE,
        "no more specific rule matched the current alert and internal state",
        "Unclear situation; more information is needed.",
    ),
)


class ModelBasedReflexAgent(BaseAgent):
    """Owns a private ModelBasedState; every instance starts with a fresh, empty one."""

    name = AGENT_NAME

    def __init__(self) -> None:
        self._state = ModelBasedState()

    def get_state_snapshot(self) -> ModelBasedState:
        """Deep copy, so callers cannot mutate the agent's private state."""
        return self._state.model_copy(deep=True)

    def decide(self, scenario: SecurityScenario) -> AgentDecision:
        prior = self._state
        updated = update_state(prior, scenario)
        rule = self._first_matching_rule(scenario, updated)

        counts = self._observation_counts(scenario, updated)
        decision = AgentDecision(
            agent_name=AGENT_NAME,
            agent_type=AgentType.MODEL_BASED_REFLEX,
            priority=rule.priority,
            decision=rule.decision.value,
            reasoning=(
                f"Rule {rule.rule_id} fired because {rule.reasoning.format(**counts)}. "
                f"The agent recommends {rule.decision.value} using the current scenario "
                "and its updated internal state."
            ),
            evidence=scenario_evidence(scenario) + self._state_evidence(counts),
            uncertainty=uncertainty_statement(scenario),
            likely_interpretation=rule.interpretation,
            recommended_actions=list(RECOMMENDED_ACTIONS[rule.decision]),
            audit_note=audit_note(scenario, AGENT_NAME, rule.rule_id, rule.decision),
            metadata={
                "matched_rule_id": rule.rule_id,
                "matched_rule_name": rule.name,
                "decision_mechanism": DECISION_MECHANISM,
                "uses_internal_state": True,
                "current_percept_only": False,
                "scenario_id": scenario.scenario_id,
                "state_summary": {
                    **counts,
                    "processed_scenario_count": updated.processed_scenario_count,
                    "pattern_seen_previously": _seen_before(scenario, updated),
                    "previous_decision": prior.last_decision,
                    "previous_priority": prior.last_priority.value if prior.last_priority else None,
                },
                **safety_metadata(),
            },
        )

        self._state = updated.model_copy(
            update={"last_decision": decision.decision, "last_priority": decision.priority}
        )
        return decision

    @staticmethod
    def _first_matching_rule(scenario: SecurityScenario, state: ModelBasedState) -> ModelBasedRule:
        for rule in RULES:
            if rule.condition(scenario, state):
                return rule
        raise RuntimeError("MB-DEFAULT always matches; the rule list was modified incorrectly")

    @staticmethod
    def _observation_counts(scenario: SecurityScenario, state: ModelBasedState) -> dict[str, int]:
        return {
            "account_count": state.account_observations[scenario.target_account],
            "asset_count": state.asset_observations[scenario.asset_id],
            "alert_type_count": state.alert_type_observations[scenario.alert_type],
        }

    @staticmethod
    def _state_evidence(counts: dict[str, Any]) -> list[str]:
        return [
            f"Observations of this account (including current): {counts['account_count']}",
            f"Observations of this asset (including current): {counts['asset_count']}",
            f"Observations of this alert type (including current): {counts['alert_type_count']}",
        ]
