"""Internal-state model for the Model-Based Reflex Agent.

The state is a compact record of what the agent has *observed*. Observing is not
learning: nothing here is ever fed back into the rules or thresholds.
"""

from pydantic import Field

from models.schemas import Priority, StrictModel


class ModelBasedState(StrictModel):
    """What the Model-Based Reflex Agent remembers about past percepts."""

    processed_scenario_count: int = Field(default=0, ge=0)
    last_scenario_id: str | None = None
    last_alert_id: str | None = None
    last_decision: str | None = None
    last_priority: Priority | None = None
    # Counters keyed by the (synthetic) identifier seen in the percept.
    account_observations: dict[str, int] = Field(default_factory=dict)
    asset_observations: dict[str, int] = Field(default_factory=dict)
    alert_type_observations: dict[str, int] = Field(default_factory=dict)
    unfamiliar_source_observations: int = Field(default=0, ge=0)
    correlated_risk_observations: int = Field(default=0, ge=0)
