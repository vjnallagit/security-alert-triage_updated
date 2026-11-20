"""Common interface that all six agents will implement in later phases."""

from abc import ABC, abstractmethod

from models.schemas import AgentDecision, SecurityScenario


class BaseAgent(ABC):
    """Every agent receives the same SecurityScenario and returns an AgentDecision."""

    name: str

    @abstractmethod
    def decide(self, scenario: SecurityScenario) -> AgentDecision:
        """Return the agent's triage decision for the scenario."""
