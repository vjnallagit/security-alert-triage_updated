"""Phase 1 validation: checks that the shared data foundation is consistent.

Run from anywhere:  python scripts/validate_phase1.py
"""

import sys
from pathlib import Path

# Allow running as a plain script without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import ScenarioCategory  # noqa: E402
from services.data_loader import DataLoadError  # noqa: E402
from services.scenario_service import (  # noqa: E402
    get_alert_detail,
    get_all_policies,
    get_all_scenarios,
    get_asset,
    get_authentication_events,
    get_experience,
    get_policy,
)

EXPECTED_CATEGORIES = {
    "A": ScenarioCategory.CRITICAL_FAILURE,
    "B": ScenarioCategory.NORMAL,
    "C": ScenarioCategory.EDGE_AMBIGUOUS,
}
REQUIRED_PROHIBITED_ACTIONS = {
    "block_user",
    "disable_account",
    "modify_firewall_rule",
    "terminate_process",
    "quarantine_host",
    "offensive_response",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def run() -> None:
    scenarios = get_all_scenarios()
    check(
        {s.scenario_id for s in scenarios} == set(EXPECTED_CATEGORIES) and len(scenarios) == 3,
        f"Loaded {len(scenarios)} scenarios",
    )

    for scenario in scenarios:
        expected = EXPECTED_CATEGORIES[scenario.scenario_id]
        check(
            scenario.scenario_category == expected,
            f"Scenario {scenario.scenario_id} validated ({expected.value})",
        )

    for scenario in scenarios:
        get_alert_detail(scenario.alert_id)
    check(True, "Supporting alert data validated")

    event_counts = {s.scenario_id: len(get_authentication_events(s.alert_id)) for s in scenarios}
    check(all(event_counts.values()), f"Authentication event data loaded {event_counts}")

    for scenario in scenarios:
        get_asset(scenario.asset_id)
    check(True, "Asset references validated")

    for scenario in scenarios:
        get_policy(scenario.policy_id)
    check(True, "Policy references validated")

    experience = get_experience()
    check(bool(experience.records) and experience.simulated, "Experience data loaded (simulated)")

    policies = get_all_policies()
    check(
        all(
            not p.automated_actions_permitted
            and p.human_authorization_required
            and REQUIRED_PROHIBITED_ACTIONS <= set(p.prohibited_automated_actions)
            for p in policies
        ),
        "Defensive safety policy verified",
    )


def main() -> int:
    print("PHASE 1 VALIDATION")
    print("==================\n")
    try:
        run()
    except (AssertionError, DataLoadError, LookupError, ValueError) as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("Phase 1 validation FAILED.")
        return 1
    print("\nPhase 1 validation successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
