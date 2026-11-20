"""Phase 3 validation: the Goal-Based Agent on the shared scenarios.

Run from anywhere:  python scripts/validate_phase3.py
"""

import contextlib
import io
import sys
from pathlib import Path

# Allow running as a plain script without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_phase1  # noqa: E402
import validate_phase2  # noqa: E402
from agents import GoalBasedAgent  # noqa: E402
from agents.goal_based import CANDIDATE_ACTIONS  # noqa: E402
from models.schemas import AgentDecision, Priority  # noqa: E402
from services.data_loader import DATA_DIR, DataLoadError  # noqa: E402
from services.scenario_service import get_all_scenarios  # noqa: E402

EXPECTED = {
    "A": (Priority.HIGH, "INVESTIGATE"),
    "B": (Priority.LOW, "COLLECT_MORE_EVIDENCE"),
    "C": (Priority.CRITICAL, "ESCALATE_TO_SECURITY_ANALYST"),
}
# Candidate that must be rejected for each scenario, and the wording its reason must carry.
MUST_REJECT = {
    "A": ("MONITOR", "inadequate", "privileged"),
    "B": ("ESCALATE_TO_SECURITY_ANALYST", "disproportionate", "no active risk"),
    "C": ("MONITOR", "inadequate", "critical service"),
}

results: list[AgentDecision] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def validate_previous_phases() -> None:
    print("Previous phases:")
    with contextlib.redirect_stdout(io.StringIO()):
        validate_phase1.run()
    check(True, "Phase 1 validation remains successful")
    with contextlib.redirect_stdout(io.StringIO()):
        phase2_status = validate_phase2.main()
    check(phase2_status == 0, "Phase 2 validation remains successful")


def validate_scenario(agent: GoalBasedAgent, scenario) -> None:
    sid = scenario.scenario_id
    result = AgentDecision.model_validate(agent.decide(scenario).model_dump())
    results.append(result)
    meta = result.metadata
    goal = meta["goal"]
    rejected = {r["decision"]: r["reasons"] for r in meta["rejected_actions"]}

    print(f"\nScenario {sid}")
    check(True, f"Goal: {goal['goal_id']} - {goal['name']}")
    check(
        meta["candidate_actions_considered"] == [c.decision for c in CANDIDATE_ACTIONS]
        and len(meta["candidate_actions_considered"]) == 4,
        f"Candidates considered: {', '.join(meta['candidate_actions_considered'])}",
    )

    decision, word, subject = MUST_REJECT[sid]
    reason = " ".join(rejected.get(decision, [])).lower()
    check(word in reason and subject in reason, f"{decision} rejected: {rejected[decision][0]}")
    for name, reasons in rejected.items():
        if name != decision:
            print(f"       also rejected {name}: {reasons[0]}")
    for item in meta["acceptable_actions_not_selected"]:
        print(f"       acceptable but not selected {item['decision']}: more intensive than necessary")

    expected_priority, expected_decision = EXPECTED[sid]
    check(
        (result.priority, result.decision) == (expected_priority, expected_decision),
        f"Selected: {result.priority.value} / {result.decision}",
    )
    for number, step in enumerate(meta["selected_plan"]["steps"], start=1):
        print(f"       plan step {number}: {step}")
    check(
        meta["selected_plan"]["goal_satisfied"] and not meta["goal_evaluation"]["unsatisfied_conditions"],
        "Selected plan satisfies goal",
    )
    check(
        not scenario.confirmed_impact and "NOT confirmed" in result.uncertainty,
        "Confirmed compromise was not claimed",
    )
    if sid == "C":
        check(meta["predicted_outcome"]["requires_human_review"], "Escalation requires human review")
        check(meta["executed_actions"] == [], "No containment was executed")


def validate_boundaries(agent: GoalBasedAgent) -> None:
    print("\nArchitectural boundaries:")
    experience = DATA_DIR / "experience.json"
    before = experience.read_bytes()
    scenarios = get_all_scenarios()
    runs = [[GoalBasedAgent().decide(s) for s in scenarios] for _ in range(2)]

    check(all(r.metadata["decision_mechanism"] == "goal_satisfaction_and_plan_selection" for r in results),
          "Goal satisfaction used")
    check(all(r.metadata["uses_utility_scoring"] is False and r.utility_score is None for r in results),
          "No utility scoring used")
    check(all(r.metadata["uses_learning"] is False and r.experience_used is None for r in results)
          and vars(agent) == {} and experience.read_bytes() == before,
          "No learning used (no history, experience.json untouched)")
    check(all(r.metadata["uses_llm"] is False and r.tool_calls == [] for r in results), "No LLM used")
    check(runs[0] == runs[1] and [agent.decide(s) for s in scenarios] == runs[0],
          "Deterministic repeated decisions")


def validate_safety() -> None:
    print("\nSafety:")
    check(all(r.metadata["recommendation_only"] for r in results), "All decisions are recommendation-only")
    check(all(r.metadata["executed_actions"] == [] and "No disruptive action was executed" in r.audit_note
              for r in results), "No disruptive action was executed")
    check(all(r.metadata["human_authorization_required_for_consequential_action"] for r in results),
          "Human authorization remains required")


def main() -> int:
    print("PHASE 3 VALIDATION")
    print("==================\n")
    try:
        validate_previous_phases()
        print("\nGoal-Based Agent:")
        agent = GoalBasedAgent()
        for scenario in get_all_scenarios():
            validate_scenario(agent, scenario)
        validate_boundaries(agent)
        validate_safety()
    except (AssertionError, DataLoadError, LookupError, ValueError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Phase 3 validation FAILED.")
        return 1
    print("\nPhase 3 validation successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
