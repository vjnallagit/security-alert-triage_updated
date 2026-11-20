"""Simple Reflex Agent: current percept -> ordered condition-action rules -> decision.

The agent has NO memory. decide() reads only the SecurityScenario it is given and
the first rule whose condition matches wins, so identical input always gives an
identical output.
"""

from collections.abc import Callable
from dataclasses import dataclass

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

AGENT_NAME = "Simple Reflex Agent"
DECISION_MECHANISM = "ordered_condition_action_rules"


# --- Conditions: each looks only at the current scenario -------------------


def _critical_asset_with_correlated_unusual_access(s: SecurityScenario) -> bool:
    return (
        s.asset_criticality == AssetCriticality.CRITICAL
        and s.unusual_access
        and s.correlated_alert_count >= MULTIPLE_CORRELATED_ALERTS_THRESHOLD
        and s.severity in (Severity.HIGH, Severity.CRITICAL)
    )


def _privileged_repeated_failures_from_unfamiliar_source(s: SecurityScenario) -> bool:
    return (
        s.account_privilege == AccountPrivilege.PRIVILEGED
        and s.source_familiarity == SourceFamiliarity.UNFAMILIAR
        and s.failed_login_count >= REPEATED_FAILED_LOGIN_THRESHOLD
    )


def _weak_low_severity_alert(s: SecurityScenario) -> bool:
    return (
        s.severity == Severity.LOW
        and s.evidence_strength == EvidenceStrength.LIMITED
        and not s.confirmed_impact
    )


def _high_or_critical_severity(s: SecurityScenario) -> bool:
    return s.severity in (Severity.HIGH, Severity.CRITICAL)


def _always(_: SecurityScenario) -> bool:
    return True


@dataclass(frozen=True)
class SimpleReflexRule:
    rule_id: str
    name: str
    condition: Callable[[SecurityScenario], bool]
    priority: Priority
    decision: TriageDecision
    reasoning: str
    interpretation: str


# Order matters: first match wins, so specific safety-relevant rules come first.
RULES: tuple[SimpleReflexRule, ...] = (
    SimpleReflexRule(
        "SR-01",
        "Critical asset with correlated alerts and unusual access",
        _critical_asset_with_correlated_unusual_access,
        Priority.CRITICAL,
        TriageDecision.ESCALATE_TO_SECURITY_ANALYST,
        "the current alert involves a critical asset, shows unusual access, has multiple "
        "correlated alerts and has HIGH or CRITICAL severity. Potential consequence is high, "
        "although compromise is not confirmed",
        "Possible serious incident on a critical service; unconfirmed and needs human review.",
    ),
    SimpleReflexRule(
        "SR-02",
        "Privileged account with repeated failed logins from unfamiliar source",
        _privileged_repeated_failures_from_unfamiliar_source,
        Priority.HIGH,
        TriageDecision.INVESTIGATE,
        "the current alert targets a privileged account, contains repeated failed logins "
        "and originates from an unfamiliar source",
        "Possible credential-guessing attempt against a privileged account; unconfirmed.",
    ),
    SimpleReflexRule(
        "SR-03",
        "Low severity, limited evidence, no confirmed impact",
        _weak_low_severity_alert,
        Priority.LOW,
        TriageDecision.COLLECT_MORE_EVIDENCE,
        "the current alert has LOW severity, LIMITED evidence and no confirmed impact, "
        "so escalation is not justified",
        "Likely benign or one-off event; evidence is too weak to conclude.",
    ),
    SimpleReflexRule(
        "SR-04",
        "High or critical severity",
        _high_or_critical_severity,
        Priority.HIGH,
        TriageDecision.INVESTIGATE,
        "the current alert has HIGH or CRITICAL severity and no more specific rule matched",
        "Significant alert whose cause is not yet understood.",
    ),
    SimpleReflexRule(
        "SR-DEFAULT",
        "Default",
        _always,
        Priority.MEDIUM,
        TriageDecision.COLLECT_MORE_EVIDENCE,
        "no more specific rule matched the current alert",
        "Unclear situation; more information is needed.",
    ),
)


class SimpleReflexAgent(BaseAgent):
    """Stateless: instances hold no history, so every call is independent."""

    name = AGENT_NAME

    def decide(self, scenario: SecurityScenario) -> AgentDecision:
        rule = self._first_matching_rule(scenario)
        return AgentDecision(
            agent_name=AGENT_NAME,
            agent_type=AgentType.SIMPLE_REFLEX,
            priority=rule.priority,
            decision=rule.decision.value,
            reasoning=(
                f"Rule {rule.rule_id} fired because {rule.reasoning}. "
                f"The agent recommends {rule.decision.value} based only on the current scenario."
            ),
            evidence=scenario_evidence(scenario),
            uncertainty=uncertainty_statement(scenario),
            likely_interpretation=rule.interpretation,
            recommended_actions=list(RECOMMENDED_ACTIONS[rule.decision]),
            audit_note=audit_note(scenario, AGENT_NAME, rule.rule_id, rule.decision),
            metadata={
                "matched_rule_id": rule.rule_id,
                "matched_rule_name": rule.name,
                "decision_mechanism": DECISION_MECHANISM,
                "uses_internal_state": False,
                "current_percept_only": True,
                "scenario_id": scenario.scenario_id,
                **safety_metadata(),
            },
        )

    @staticmethod
    def _first_matching_rule(scenario: SecurityScenario) -> SimpleReflexRule:
        for rule in RULES:
            if rule.condition(scenario):
                return rule
        raise RuntimeError("SR-DEFAULT always matches; the rule list was modified incorrectly")
