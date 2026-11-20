"""Phase 5 validation: the Learning Agent and its feedback-driven behavior change.

Run from anywhere:  python scripts/validate_phase5.py

All learning happens in temporary experience stores. The committed data/experience.json is
only ever read (copied as a seed) and its SHA-256 hash is verified before and after.
"""

import contextlib
import hashlib
import io
import sys
import tempfile
from pathlib import Path

# Allow running as a plain script without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_phase1  # noqa: E402
import validate_phase2  # noqa: E402
import validate_phase3  # noqa: E402
import validate_phase4  # noqa: E402
from agents import LearningAgent  # noqa: E402
from agents import learning_agent as learning_module  # noqa: E402
from agents.learning_agent import LEARNING_CANDIDATES, PerformanceElement  # noqa: E402
from models.goal_models import ProhibitedCondition  # noqa: E402
from models.learning_models import FeedbackOutcome, LearningCandidate  # noqa: E402
from models.schemas import AgentDecision, Priority  # noqa: E402
from services.data_loader import DATA_DIR, DataLoadError  # noqa: E402
from services.experience_repository import ExperienceRepository  # noqa: E402
from services.scenario_service import get_all_scenarios, get_scenario  # noqa: E402

COMMITTED = DATA_DIR / "experience.json"
PATTERN = "privileged_login_anomaly"
INVESTIGATE = "INVESTIGATE"
ESCALATE = "ESCALATE_TO_SECURITY_ANALYST"
COLLECT = "COLLECT_MORE_EVIDENCE"
MONITOR = "MONITOR"
EXPECTED = {
    "A": (Priority.HIGH, INVESTIGATE),
    "B": (Priority.LOW, COLLECT),
    "C": (Priority.CRITICAL, ESCALATE),
}

results: list[AgentDecision] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothed(success: int, failure: int) -> float:
    return (success + 1) / (success + failure + 2)


def validate_previous_phases() -> None:
    print("Previous phases:")
    with contextlib.redirect_stdout(io.StringIO()):
        validate_phase1.run()
    check(True, "Phase 1")
    for number, module in ((2, validate_phase2), (3, validate_phase3), (4, validate_phase4)):
        with contextlib.redirect_stdout(io.StringIO()):
            status = module.main()
        check(status == 0, f"Phase {number}")


def show_candidate(candidate: dict) -> None:
    if not candidate["eligible"]:
        print(f"  {candidate['decision']}: excluded before history was used "
              f"({candidate['ineligibility_reasons'][0].split(':')[0]})  "
              f"[history {candidate['success']}/{candidate['failure']}]")
        return
    raw = "n/a" if candidate["raw_success_rate"] is None else f"{candidate['raw_success_rate']:.4f}"
    print(f"  {candidate['decision']}")
    print(f"    success = {candidate['success']}   failure = {candidate['failure']}   "
          f"raw rate = {raw}   smoothed rate = {candidate['smoothed_success_rate']:.4f}")


def validate_baseline(agent: LearningAgent, repository: ExperienceRepository) -> None:
    print("\nLearning Agent baseline (temporary copy of the seed experience):")
    before = repository.path.read_bytes()
    for scenario in get_all_scenarios():
        sid = scenario.scenario_id
        result = AgentDecision.model_validate(agent.decide(scenario).model_dump())
        results.append(result)
        meta = result.metadata
        print(f"\nScenario {sid}")
        print(f"Pattern: {meta['learning_pattern']}")
        for candidate in meta["candidate_experience"]:
            show_candidate(candidate)
        guidance = meta["problem_generator"]
        print(f"  experience sufficient: {meta['experience_sufficient']}   "
              f"problem generator used: {meta['problem_generator_used']}")
        print(f"  problem generator: {guidance['guidance']}")
        check((result.priority, result.decision) == EXPECTED[sid],
              f"Selected {result.priority.value} / {result.decision}  (basis: {meta['selection_basis']})")
        eligible = [c for c in meta["candidate_experience"] if c["eligible"]]
        if len(eligible) > 1:
            best = max(eligible, key=lambda c: c["smoothed_success_rate"])
            check(best["decision"] == result.decision and meta["selection_basis"]
                  == "safe_action_then_smoothed_historical_success",
                  "Experience affected selection (highest smoothed success among eligible actions)")
        else:
            check(True, "Only one eligible action: current-scenario safety/adequacy decided, history confirmed it")
    check(repository.path.read_bytes() == before, "decide() alone did not modify experience")


def validate_learning_cycle(tmp: Path) -> None:
    print("\nLearning cycle demonstration (controlled temporary store):")
    repository = ExperienceRepository.create(
        tmp / "controlled.json", {PATTERN: {INVESTIGATE: (4, 1), ESCALATE: (3, 2)}}
    )
    agent = LearningAgent(repository)
    scenario = get_scenario("A")
    snapshot = scenario.model_copy(deep=True)
    select_code = PerformanceElement.select.__code__.co_code
    escalate_start = repository.get_stats(PATTERN, ESCALATE)

    initial = agent.decide(scenario)
    start = repository.get_stats(PATTERN, INVESTIGATE)
    print("\nBEFORE")
    print(f"{INVESTIGATE}: success = {start.success}  failure = {start.failure}  "
          f"smoothed rate = {start.smoothed_success_rate:.4f}")
    print(f"{ESCALATE}: success = {escalate_start.success}  failure = {escalate_start.failure}  "
          f"smoothed rate = {escalate_start.smoothed_success_rate:.4f}")
    check((initial.priority, initial.decision) == (Priority.HIGH, INVESTIGATE), "Initial decision HIGH / INVESTIGATE")

    update = agent.record_feedback(scenario, INVESTIGATE, FeedbackOutcome.FAILURE)
    print("\nFeedback: FAILURE")
    print("\nAFTER")
    print(f"{INVESTIGATE}: success = {update.after_success}  failure = {update.after_failure}  "
          f"smoothed rate = {update.after_smoothed_rate:.4f}")
    check(update.critic_assessment.usable, f"Critic processed FAILURE: {update.critic_assessment.reason}")
    check(update.changed and update.after_failure == update.before_failure + 1
          and update.after_success == update.before_success, "Learning Element updated experience (failure +1 only)")
    check(ExperienceRepository(repository.path).get_stats(PATTERN, INVESTIGATE).failure == 2,
          "Experience persisted (a freshly opened repository sees it)")
    unknown = agent.record_feedback(scenario, INVESTIGATE, FeedbackOutcome.UNKNOWN)
    check(not unknown.changed, "UNKNOWN feedback was not counted")

    print("\nSame Scenario A, repeated explicit FAILURE feedback for INVESTIGATE:")
    expected_switch = next(f for f in range(1, 11) if smoothed(4, 1 + f) < smoothed(3, 2))
    failures, decision, switched_after = 1, agent.decide(scenario), None
    print(f"  after {failures} failure: {INVESTIGATE} smoothed = {update.after_smoothed_rate:.4f}  ->  {decision.decision}")
    while decision.decision == INVESTIGATE and failures < 10:  # safe maximum of 10 feedback events
        step = agent.record_feedback(scenario, INVESTIGATE, FeedbackOutcome.FAILURE)
        failures += 1
        decision = agent.decide(scenario)
        print(f"  after {failures} failures: {INVESTIGATE} smoothed = {step.after_smoothed_rate:.4f}  "
              f"(competitor {ESCALATE} = {escalate_start.smoothed_success_rate:.4f})  ->  {decision.decision}")
        if decision.decision != INVESTIGATE:
            switched_after = failures
    check(decision.decision == ESCALATE and switched_after == expected_switch,
          f"Same Scenario A eventually changed: {INVESTIGATE} -> {ESCALATE}")
    print(f"       exactly {switched_after} failure events were needed "
          f"(smallest n with 5/(7+n) < 4/7, derived from the smoothing formula)")
    check(decision.priority is Priority.HIGH and decision.metadata["experience_changed_default"],
          "Priority stayed grounded in current risk; experience changed the action")
    check(scenario == snapshot, "Scenario facts unchanged")
    check(PerformanceElement.select.__code__.co_code == select_code and repository.get_stats(PATTERN, ESCALATE)
          == escalate_start, "Only experience changed (selection code and the competitor's counts untouched)")
    results.append(AgentDecision.model_validate(decision.model_dump()))

    print("\nSuccess feedback (opposite direction):")
    for _ in range(3):
        step = agent.record_feedback(scenario, ESCALATE, FeedbackOutcome.SUCCESS)
    check(step.after_smoothed_rate > escalate_start.smoothed_success_rate,
          f"Repeated SUCCESS raised {ESCALATE} smoothed rate {escalate_start.smoothed_success_rate:.4f} "
          f"-> {step.after_smoothed_rate:.4f}")

    print("\nPersistence:")
    check(ExperienceRepository(repository.path).path != COMMITTED, "Temporary experience persisted (separate file)")
    student = LearningAgent(ExperienceRepository(repository.path))
    check(student.decide(scenario).decision == ESCALATE, "New agent observes learned update")


def validate_problem_generator(tmp: Path) -> None:
    print("\nProblem Generator (cold start, no history):")
    cold = LearningAgent(ExperienceRepository.create(tmp / "cold.json", {"unrelated": {MONITOR: (0, 0)}}))
    b = cold.decide(get_scenario("B"))
    check(not b.metadata["experience_sufficient"], "Sparse experience detected")
    check(b.decision == COLLECT and b.metadata["problem_generator"]["recommended_decision"] == COLLECT,
          "Evidence-gathering guidance generated where it is safe (Scenario B)")
    print(f"       {b.metadata['problem_generator']['guidance']}")
    c = cold.decide(get_scenario("C"))
    check(c.decision == ESCALATE and c.priority is Priority.CRITICAL
          and c.metadata["selection_basis"] != "problem_generator_evidence_gathering",
          "Critical-risk safety overrides exploration (Scenario C is not reduced to evidence-gathering)")
    results.extend(AgentDecision.model_validate(r.model_dump()) for r in (b, c))


def validate_boundaries() -> None:
    print("\nArchitectural boundaries:")
    check(all(r.metadata["learning_components"] == {
        "performance_element": True, "critic": True, "learning_element": True, "problem_generator": True,
    } for r in results), "Performance Element, Critic, Learning Element and Problem Generator are all present")
    check(all(r.utility_score is None and r.metadata["uses_utility_scoring"] is False
              and "action_analyses" not in r.metadata for r in results), "No expected utility used")
    check(all(r.metadata["uses_llm"] is False and r.tool_calls == [] for r in results)
          and not {"langchain", "ollama", "streamlit"} & set(sys.modules), "No LLM used")
    check(not any(hasattr(learning_module, name) for name in ("random", "secrets", "time")), "No randomness used")
    check(all(r.experience_used for r in results) and all(r.metadata["decide_modifies_experience"] is False
                                                          for r in results), "experience_used populated; decide() is read-only")


def validate_safety(tmp: Path) -> None:
    print("\nSafety:")
    check(all(r.metadata["recommendation_only"] and r.metadata["executed_actions"] == [] for r in results),
          "Recommendation-only")
    check(all("No disruptive action was executed" in r.audit_note for r in results), "No disruptive action was executed")
    check(all(r.metadata["human_authorization_required_for_consequential_action"] for r in results),
          "Human authorization required")

    # An unsafe action with a flawless history must never be selectable.
    repository = ExperienceRepository.create(tmp / "unsafe.json", {
        "low_severity_limited_evidence": {MONITOR: (100, 0), COLLECT: (1, 9), INVESTIGATE: (1, 9)}})
    unsafe = LearningCandidate(action_id="LA-ACT-UNSAFE", decision=MONITOR,
                               side_effects=(ProhibitedCondition.AUTOMATIC_USER_BLOCKING,))
    original = learning_module.LEARNING_CANDIDATES
    learning_module.LEARNING_CANDIDATES = tuple(unsafe if c.decision == MONITOR else c for c in original)
    try:
        meta = LearningAgent(repository).decide(get_scenario("B")).metadata
    finally:
        learning_module.LEARNING_CANDIDATES = original
    entry = next(c for c in meta["candidate_experience"] if c["action_id"] == "LA-ACT-UNSAFE")
    check(not entry["eligible"] and entry["success"] == 100
          and meta["selected_experience"]["action_id"] != "LA-ACT-UNSAFE",
          "Learning cannot override safety (unsafe action with 100 successes / 0 failures stayed ineligible)")
    check(original == LEARNING_CANDIDATES, "Production candidate catalogue restored")


def main() -> int:
    print("PHASE 5 VALIDATION")
    print("==================\n")
    before_hash = digest(COMMITTED)
    try:
        validate_previous_phases()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            seed_copy = ExperienceRepository(COMMITTED).copy_to(tmp / "seed_copy.json")
            validate_baseline(LearningAgent(seed_copy), seed_copy)
            validate_learning_cycle(tmp)
            validate_problem_generator(tmp)
            print("\nCommitted data:")
            check(digest(COMMITTED) == before_hash, "Committed experience.json unchanged")
            validate_boundaries()
            validate_safety(tmp)
        check(digest(COMMITTED) == before_hash, "Committed experience.json still unchanged at the end")
    except (AssertionError, DataLoadError, LookupError, ValueError, RuntimeError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Phase 5 validation FAILED.")
        return 1
    print("\nPhase 5 validation successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
