"""Phase 4 validation: the Utility-Based Agent on the shared scenarios.

Run from anywhere:  python scripts/validate_phase4.py
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
import validate_phase3  # noqa: E402
from agents import UtilityBasedAgent  # noqa: E402
from agents import utility_based as utility_module  # noqa: E402
from models.goal_models import ProhibitedCondition  # noqa: E402
from models.schemas import AgentDecision, Priority  # noqa: E402
from models.utility_models import TOLERANCE, UtilityAction, UtilityStep  # noqa: E402
from services.data_loader import DATA_DIR, DataLoadError  # noqa: E402
from services.scenario_service import get_all_scenarios  # noqa: E402

EXPECTED = {
    "A": (Priority.HIGH, "INVESTIGATE"),
    "B": (Priority.LOW, "COLLECT_MORE_EVIDENCE"),
    "C": (Priority.CRITICAL, "ESCALATE_TO_SECURITY_ANALYST"),
}
FOUR_DECISIONS = ["MONITOR", "COLLECT_MORE_EVIDENCE", "INVESTIGATE", "ESCALATE_TO_SECURITY_ANALYST"]

results: list[AgentDecision] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def recompute_expected_utility(action_meta: dict) -> float:
    """Recomputes EU from probabilities, raw criterion scores and weights only."""
    total = 0.0
    for outcome in action_meta["outcomes"]:
        utility = sum(part["raw_score"] * part["weight"] for part in outcome["criterion_breakdown"])
        total += outcome["probability"] * utility
    return total


def favorable_score(result: AgentDecision, decision: str, criterion: str) -> float:
    action = next(a for a in result.metadata["action_analyses"] if a["decision"] == decision)
    outcome = next(o for o in action["outcomes"] if o["name"] == "FAVORABLE")
    return next(p["raw_score"] for p in outcome["criterion_breakdown"] if p["criterion"] == criterion)


def validate_previous_phases() -> None:
    print("Previous phases:")
    with contextlib.redirect_stdout(io.StringIO()):
        validate_phase1.run()
    check(True, "Phase 1 validation remains successful")
    for number, module in ((2, validate_phase2), (3, validate_phase3)):
        with contextlib.redirect_stdout(io.StringIO()):
            status = module.main()
        check(status == 0, f"Phase {number} validation remains successful")


def validate_profile(agent: UtilityBasedAgent) -> None:
    profile = agent._define_utility_profile()
    print(f"\nUtility profile {profile.profile_id} ({profile.name}):")
    for criterion in profile.criteria:
        check(True, f"{criterion.criterion_id} {criterion.name.value} = {criterion.weight:.2f}")
    check(abs(profile.weight_total - 1.0) <= TOLERANCE, f"Weight total = {profile.weight_total:.2f}")


def validate_scenario(agent: UtilityBasedAgent, scenario) -> None:
    sid = scenario.scenario_id
    result = AgentDecision.model_validate(agent.decide(scenario).model_dump())
    results.append(result)
    meta = result.metadata
    actions = meta["action_analyses"]

    print(f"\nScenario {sid}:")
    context = meta["decision_context"]
    check(True, f"Context: risk need {context['risk_need']:.2f}, asset stake {context['asset_stake']:.2f}, "
                f"evidence support {context['evidence_support']:.2f}, "
                f"required intensity {context['required_intensity']:.2f}")
    check(meta["candidate_actions_considered"] == FOUR_DECISIONS, "Four defensive candidates evaluated")

    for action in actions:
        probabilities = " ".join(f"{o['name'][:3]} P={o['probability']:.4f} U={o['outcome_utility']:.4f}"
                                 for o in action["outcomes"])
        recomputed = recompute_expected_utility(action)
        check(abs(recomputed - action["expected_utility"]) <= TOLERANCE
              and abs(sum(o["probability"] for o in action["outcomes"]) - 1.0) <= TOLERANCE,
              f"{action['decision']} expected utility = {action['expected_utility']:.4f}  [{probabilities}]")

    ranking = "  >  ".join(f"{r['rank']}. {r['decision']} ({r['expected_utility']:.4f})" for r in meta["action_ranking"])
    print(f"       ranking: {ranking}")
    check(True, "Probabilities sum to 1.0 and expected utilities independently recomputed")

    expected_priority, expected_decision = EXPECTED[sid]
    check((result.priority, result.decision) == (expected_priority, expected_decision),
          f"Selected {result.priority.value} / {result.decision} (utility score {result.utility_score:.4f})")
    selected = next(a for a in actions if a["decision"] == result.decision)
    check(selected["rank"] == 1, "Selected action has rank 1")
    check(result.utility_score == selected["expected_utility"] == meta["selected_expected_utility"],
          "utility_score matches the selected expected utility")

    trade = meta["key_trade_off"]
    gains = ", ".join(a["criterion"] for a in trade["advantages"])
    gives = ", ".join(c["criterion"] for c in trade["concessions"]) or "none"
    print(f"       key trade-off vs {trade['runner_up']}: gains on {gains}; concedes {gives}")

    if sid == "B":
        check(trade["advantages"][0]["criterion"] == "EVIDENCE_ALIGNMENT"
              and favorable_score(result, "COLLECT_MORE_EVIDENCE", "PROPORTIONALITY")
              > favorable_score(result, "ESCALATE_TO_SECURITY_ANALYST", "PROPORTIONALITY"),
              "Evidence alignment led the ranking and proportionality separated it from escalation")
    if sid == "C":
        top_two = [a["criterion"] for a in trade["advantages"][:2]]
        check(set(top_two) == {"ASSET_PROTECTION", "RISK_REDUCTION"},
              "Asset protection and risk reduction drove the ranking")
    check(not scenario.confirmed_impact and "NOT confirmed" in result.uncertainty,
          "Compromise remains unconfirmed")
    if sid == "C":
        check(any("authorized SOC or security-analyst" in s for s in result.recommended_actions),
              "Human review required")


def print_worked_example(agent: UtilityBasedAgent) -> None:
    scenario = next(s for s in get_all_scenarios() if s.scenario_id == "A")
    meta = agent.decide(scenario).metadata
    action = next(a for a in meta["action_analyses"] if a["decision"] == "INVESTIGATE")
    print("\nWorked example (Scenario A, INVESTIGATE):")
    for outcome in action["outcomes"]:
        print(f"       {outcome['name']:<11} P = {outcome['probability']:.4f}  U = {outcome['outcome_utility']:.4f}"
              f"  P x U = {outcome['contribution_to_expected_utility']:.4f}")
    first = action["outcomes"][0]
    print(f"       {first['name']} utility breakdown (raw score x weight):")
    for part in first["criterion_breakdown"]:
        print(f"         {part['criterion']:<23} {part['raw_score']:.4f} x {part['weight']:.2f} = "
              f"{part['weighted_contribution']:.4f}")
    print(f"       EU(INVESTIGATE) = {action['expected_utility']:.4f}")


def validate_boundaries(agent: UtilityBasedAgent) -> None:
    print("\nArchitectural boundaries:")
    experience = DATA_DIR / "experience.json"
    before = experience.read_bytes()
    scenarios = get_all_scenarios()
    runs = [[UtilityBasedAgent().decide(s) for s in scenarios] for _ in range(2)]

    check(all(r.metadata["decision_mechanism"] == "expected_utility_maximization"
              and r.metadata["uses_goal_satisfaction_selection"] is False for r in results),
          "Expected utility used (not goal-satisfaction selection)")
    check(all(r.metadata["uses_learning"] is False and r.experience_used is None for r in results)
          and vars(agent) == {"_profile": utility_module.DEFAULT_UTILITY_PROFILE},
          "No learning used (fixed profile, no history)")
    check(experience.read_bytes() == before, "No experience update (experience.json unchanged)")
    check(all(r.metadata["uses_llm"] is False and r.tool_calls == [] for r in results), "No LLM used")
    check(runs[0] == runs[1] and [agent.decide(s) for s in scenarios] == runs[0], "Deterministic results")


def validate_safety() -> None:
    print("\nSafety:")
    check(all(r.metadata["recommendation_only"] for r in results), "All decisions are recommendation-only")
    check(all(r.metadata["executed_actions"] == [] and "No disruptive action was executed" in r.audit_note
              for r in results), "No disruptive action was executed")

    # A prohibited candidate that scores higher than every safe one must still never be selected.
    unsafe = UtilityAction(
        action_id="UB-ACT-UNSAFE", decision="ESCALATE_TO_SECURITY_ANALYST", description="Unsafe test candidate.",
        risk_handling=1.0, asset_protection=1.0, evidence_development=1.0, intensity=0.85, human_control=1.0,
        plan_steps=(UtilityStep(text="Placeholder step."),),
        side_effects=(ProhibitedCondition.AUTOMATIC_USER_BLOCKING,),
    )
    original = utility_module.CANDIDATE_ACTIONS
    utility_module.CANDIDATE_ACTIONS = (*original, unsafe)
    try:
        scenario = next(s for s in get_all_scenarios() if s.scenario_id == "C")
        meta = UtilityBasedAgent().decide(scenario).metadata
    finally:
        utility_module.CANDIDATE_ACTIONS = original
    by_id = {a["action_id"]: a for a in meta["action_analyses"]}
    best_safe = max(a["expected_utility"] for a in by_id.values() if a["eligible"])
    check(by_id["UB-ACT-UNSAFE"]["expected_utility"] > best_safe
          and not by_id["UB-ACT-UNSAFE"]["eligible"]
          and meta["selected_action_id"] != "UB-ACT-UNSAFE",
          "Safety constraints override utility (higher-utility prohibited action never selected)")
    check(all(r.metadata["human_authorization_required_for_consequential_action"] for r in results),
          "Human authorization remains required")


def main() -> int:
    print("PHASE 4 VALIDATION")
    print("==================\n")
    try:
        validate_previous_phases()
        agent = UtilityBasedAgent()
        validate_profile(agent)
        for scenario in get_all_scenarios():
            validate_scenario(agent, scenario)
        print_worked_example(agent)
        validate_boundaries(agent)
        validate_safety()
    except (AssertionError, DataLoadError, LookupError, ValueError, RuntimeError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Phase 4 validation FAILED.")
        return 1
    print("\nPhase 4 validation successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
