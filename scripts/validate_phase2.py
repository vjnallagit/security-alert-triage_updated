"""Phase 2 validation: Simple Reflex vs Model-Based Reflex on the shared scenarios.

Run from anywhere:  python scripts/validate_phase2.py
"""

import contextlib
import io
import sys
from pathlib import Path

# Allow running as a plain script without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_phase1  # noqa: E402
from agents import ModelBasedReflexAgent, SimpleReflexAgent  # noqa: E402
from models.schemas import AgentDecision, Priority  # noqa: E402
from services.data_loader import DataLoadError  # noqa: E402
from services.scenario_service import get_all_scenarios, get_scenario  # noqa: E402

ESCALATE = "ESCALATE_TO_SECURITY_ANALYST"
SIMPLE_EXPECTED = {
    "A": ("SR-02", Priority.HIGH, "INVESTIGATE"),
    "B": ("SR-03", Priority.LOW, "COLLECT_MORE_EVIDENCE"),
    "C": ("SR-01", Priority.CRITICAL, ESCALATE),
}
MODEL_BASED_FIRST_EXPECTED = {
    "A": ("MB-03", Priority.HIGH, "INVESTIGATE"),
    "B": ("MB-05", Priority.LOW, "COLLECT_MORE_EVIDENCE"),
    "C": ("MB-01", Priority.CRITICAL, ESCALATE),
}

all_results: list[AgentDecision] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def run_and_record(agent, scenario) -> AgentDecision:
    result = agent.decide(scenario)
    all_results.append(AgentDecision.model_validate(result.model_dump()))
    return result


def describe(result: AgentDecision) -> str:
    return f"{result.metadata['matched_rule_id']} -> {result.priority.value} -> {result.decision}"


def check_expected(label: str, result: AgentDecision, expected: tuple) -> None:
    actual = (result.metadata["matched_rule_id"], result.priority, result.decision)
    check(actual == expected, f"{label} -> {describe(result)}")


def validate_phase1_foundation() -> None:
    print("Phase 1 foundation:")
    # Reuse Phase 1 checks; hide their individual lines to keep this report focused.
    with contextlib.redirect_stdout(io.StringIO()):
        validate_phase1.run()
    check(True, "Phase 1 validation remains successful")


def validate_simple_reflex() -> None:
    print("\nSimple Reflex Agent:")
    agent = SimpleReflexAgent()
    for scenario in get_all_scenarios():
        check_expected(scenario.scenario_id, run_and_record(agent, scenario), SIMPLE_EXPECTED[scenario.scenario_id])


def validate_model_based_fresh() -> None:
    print("\nModel-Based Reflex Agent, fresh observations:")
    agent = ModelBasedReflexAgent()
    for scenario in get_all_scenarios():
        expected = MODEL_BASED_FIRST_EXPECTED[scenario.scenario_id]
        check_expected(f"{scenario.scenario_id} first observation", run_and_record(agent, scenario), expected)


def validate_state_differentiation() -> None:
    print("\nState differentiation (same Scenario A object, run twice):")
    scenario = get_scenario("A")
    simple, model_based = SimpleReflexAgent(), ModelBasedReflexAgent()

    simple_runs = [run_and_record(simple, scenario) for _ in range(2)]
    model_runs = [run_and_record(model_based, scenario) for _ in range(2)]

    for number, result in enumerate(simple_runs, start=1):
        check((result.priority, result.decision) == (Priority.HIGH, "INVESTIGATE"),
              f"Simple Reflex A run {number} -> {result.priority.value} / {result.decision}")
    first, second = model_runs
    check((first.priority, first.decision) == (Priority.HIGH, "INVESTIGATE"),
          f"Model-Based A run 1 -> {first.priority.value} / {first.decision}")
    check((second.priority, second.decision) == (Priority.CRITICAL, ESCALATE),
          f"Model-Based A run 2 -> {second.priority.value} / {second.decision}")

    observations = model_based.get_state_snapshot().account_observations[scenario.target_account]
    check(observations == 2, f"Model-Based account observations -> {observations}")
    check(vars(simple) == {}, "Simple Reflex holds no state after repeated runs")


def validate_safety() -> None:
    print("\nSafety:")
    check(all(r.metadata["recommendation_only"] for r in all_results),
          f"All {len(all_results)} decisions are valid AgentDecision objects and recommendation-only")
    check(all(r.metadata["executed_actions"] == [] and
              "No disruptive action was executed" in r.audit_note for r in all_results),
          "No disruptive action was executed")
    check(all(r.metadata["human_authorization_required_for_consequential_action"] for r in all_results),
          "Consequential action requires human authorization")


def main() -> int:
    print("PHASE 2 VALIDATION")
    print("==================\n")
    try:
        validate_phase1_foundation()
        validate_simple_reflex()
        validate_model_based_fresh()
        validate_state_differentiation()
        validate_safety()
    except (AssertionError, DataLoadError, LookupError, ValueError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Phase 2 validation FAILED.")
        return 1
    print("\nPhase 2 validation successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
