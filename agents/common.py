"""Vocabulary and output helpers shared by the classical agents.

Only constants and text/format helpers live here. Each agent keeps its own rules
and its own decision logic, so the agents stay genuinely different mechanisms.
"""

from enum import StrEnum

from models.schemas import SecurityScenario

# Failed logins needed before a percept counts as "repeated failed logins".
REPEATED_FAILED_LOGIN_THRESHOLD = 3
# Correlated alerts needed before a percept counts as "multiple correlated alerts".
MULTIPLE_CORRELATED_ALERTS_THRESHOLD = 2


class TriageDecision(StrEnum):
    """Primary recommendations. All are recommendations; none executes anything."""

    MONITOR = "MONITOR"
    COLLECT_MORE_EVIDENCE = "COLLECT_MORE_EVIDENCE"
    INVESTIGATE = "INVESTIGATE"
    ESCALATE_TO_SECURITY_ANALYST = "ESCALATE_TO_SECURITY_ANALYST"


RECOMMENDED_ACTIONS: dict[TriageDecision, list[str]] = {
    TriageDecision.MONITOR: [
        "Continue monitoring the activity for further related alerts.",
        "Re-evaluate if new correlated evidence appears.",
    ],
    TriageDecision.COLLECT_MORE_EVIDENCE: [
        "Gather additional evidence about the alert.",
        "Review authentication events for the affected account.",
        "Validate whether the source is expected.",
    ],
    TriageDecision.INVESTIGATE: [
        "Open a security investigation.",
        "Review authentication events for the targeted account.",
        "Validate whether the source is expected.",
        "Review correlated alerts and assess the affected asset.",
    ],
    TriageDecision.ESCALATE_TO_SECURITY_ANALYST: [
        "Escalate to a security analyst or the SOC for review.",
        "Review correlated alerts and authentication events together.",
        "Assess the affected asset and notify its owner.",
        "Request authorized containment review only if evidence later justifies it.",
    ],
}

NO_ACTION_STATEMENT = "No disruptive action was executed; this is a recommendation only."
HUMAN_AUTHORIZATION_STATEMENT = "Any consequential security action requires human authorization."


def scenario_evidence(scenario: SecurityScenario) -> list[str]:
    """Facts stated directly in the scenario, formatted identically for every agent."""
    return [
        f"Alert type: {scenario.alert_type}",
        f"Severity: {scenario.severity.value}",
        f"Account privilege: {scenario.account_privilege.value}",
        f"Source familiarity: {scenario.source_familiarity.value}",
        f"Failed login count: {scenario.failed_login_count}",
        f"Correlated alert count: {scenario.correlated_alert_count}",
        f"Asset criticality: {scenario.asset_criticality.value}",
        f"Unusual access: {scenario.unusual_access}",
        f"Evidence strength: {scenario.evidence_strength.value}",
        f"Confirmed impact: {scenario.confirmed_impact}",
    ]


def uncertainty_statement(scenario: SecurityScenario) -> str:
    """Always states the confirmed-impact status so potential impact is not mistaken for fact."""
    if scenario.confirmed_impact:
        impact = "Impact is marked as confirmed in the scenario."
    else:
        impact = "Compromise or impact is NOT confirmed; the concern is potential risk only."
    return f"{impact} Evidence strength is {scenario.evidence_strength.value}."


def audit_note(
    scenario: SecurityScenario, agent_name: str, rule_id: str, decision: TriageDecision
) -> str:
    return (
        f"Scenario {scenario.scenario_id} | Agent: {agent_name} | Rule: {rule_id} | "
        f"Decision: {decision.value} | {NO_ACTION_STATEMENT} {HUMAN_AUTHORIZATION_STATEMENT}"
    )


def safety_metadata() -> dict[str, object]:
    """Structured safety flags so tests and the UI need not parse prose."""
    return {
        "recommendation_only": True,
        "executed_actions": [],
        "human_authorization_required_for_consequential_action": True,
    }
