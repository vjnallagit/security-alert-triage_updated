"""Shared Pydantic models for the Security Alert Triage assistant.

Two groups live here:

* Environment/percept models (SecurityScenario and the mock supporting data).
  These describe the world and never contain an agent's decision.
* The output contract (AgentDecision) that every agent must return.
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

NonEmptyStr = Field(min_length=1)


class StrictModel(BaseModel):
    """Base model: trims strings and rejects unknown fields so typos fail loudly."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ScenarioCategory(StrEnum):
    NORMAL = "NORMAL"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"
    EDGE_AMBIGUOUS = "EDGE_AMBIGUOUS"


class AccountPrivilege(StrEnum):
    STANDARD = "STANDARD"
    PRIVILEGED = "PRIVILEGED"
    SERVICE = "SERVICE"
    UNKNOWN = "UNKNOWN"


class SourceFamiliarity(StrEnum):
    KNOWN = "KNOWN"
    UNFAMILIAR = "UNFAMILIAR"
    UNKNOWN = "UNKNOWN"


class AssetCriticality(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceStrength(StrEnum):
    LIMITED = "LIMITED"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    CONFLICTING = "CONFLICTING"


class AgentType(StrEnum):
    SIMPLE_REFLEX = "SIMPLE_REFLEX"
    MODEL_BASED_REFLEX = "MODEL_BASED_REFLEX"
    GOAL_BASED = "GOAL_BASED"
    UTILITY_BASED = "UTILITY_BASED"
    LEARNING = "LEARNING"
    LLM = "LLM"


class Priority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# --------------------------------------------------------------------------
# Shared scenario (the percept sent to every agent)
# --------------------------------------------------------------------------


class SecurityScenario(StrictModel):
    """One synthetic security situation. Describes the environment only."""

    scenario_id: str = NonEmptyStr
    title: str = NonEmptyStr
    description: str = NonEmptyStr
    scenario_category: ScenarioCategory
    alert_id: str = NonEmptyStr
    alert_type: str = NonEmptyStr
    severity: Severity
    target_account: str = NonEmptyStr
    account_privilege: AccountPrivilege
    source: str = NonEmptyStr
    source_familiarity: SourceFamiliarity
    asset_id: str = NonEmptyStr
    asset_name: str = NonEmptyStr
    asset_criticality: AssetCriticality
    failed_login_count: int = Field(ge=0)
    correlated_alert_count: int = Field(ge=0)
    unusual_access: bool
    confirmed_impact: bool
    evidence_strength: EvidenceStrength
    policy_id: str = NonEmptyStr
    # Documents the design intent of the fixture for graders/tests; never a decision.
    expected_characteristics: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Mock supporting data
# --------------------------------------------------------------------------


class AlertDetail(StrictModel):
    alert_id: str = NonEmptyStr
    timestamp: AwareDatetime
    alert_type: str = NonEmptyStr
    severity: Severity
    summary: str = NonEmptyStr
    target: str = NonEmptyStr
    source: str = NonEmptyStr
    evidence: list[str] = Field(min_length=1)


class AuthenticationEvent(StrictModel):
    event_id: str = NonEmptyStr
    alert_id: str = NonEmptyStr
    timestamp: AwareDatetime
    account: str = NonEmptyStr
    source: str = NonEmptyStr
    event_type: str = NonEmptyStr
    success: bool
    asset_id: str = NonEmptyStr


class Asset(StrictModel):
    asset_id: str = NonEmptyStr
    asset_name: str = NonEmptyStr
    asset_type: str = NonEmptyStr
    criticality: AssetCriticality
    description: str = NonEmptyStr


class ResponsePolicy(StrictModel):
    """Defensive response policy. Automation of disruptive actions is never allowed."""

    policy_id: str = NonEmptyStr
    title: str = NonEmptyStr
    description: str = NonEmptyStr
    recommended_actions: list[str] = Field(min_length=1)
    escalation_guidance: str = NonEmptyStr
    prohibited_automated_actions: list[str] = Field(min_length=1)
    restriction: str = NonEmptyStr
    # Literal types make a policy that permits automation fail validation at load time.
    automated_actions_permitted: Literal[False]
    human_authorization_required: Literal[True]


class ExperienceOutcome(StrictModel):
    """Simulated success/failure counts. The rate is derived, never stored."""

    success: int = Field(ge=0)
    failure: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.success + self.failure

    @property
    def success_rate(self) -> float | None:
        """success / (success + failure), or None when there is no history."""
        return self.success / self.total if self.total else None


class ExperienceData(StrictModel):
    """Seed experience for the future Learning Agent: alert_type -> action -> outcome."""

    description: str = NonEmptyStr
    simulated: Literal[True]
    records: dict[str, dict[str, ExperienceOutcome]] = Field(min_length=1)


# --------------------------------------------------------------------------
# Common agent output contract
# --------------------------------------------------------------------------


class AgentDecision(StrictModel):
    """Output contract shared by all six agents (no behaviour in Phase 1)."""

    agent_name: str = NonEmptyStr
    agent_type: AgentType
    priority: Priority
    decision: str = NonEmptyStr
    reasoning: str = NonEmptyStr
    evidence: list[str] = Field(default_factory=list)
    uncertainty: str = NonEmptyStr
    likely_interpretation: str = NonEmptyStr
    recommended_actions: list[str] = Field(min_length=1)
    audit_note: str = NonEmptyStr

    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    utility_score: float | None = None
    experience_used: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
