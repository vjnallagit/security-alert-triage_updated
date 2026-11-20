"""Loads and validates scenarios and their supporting mock records.

This module only reads data. It contains no agent or decision logic.
"""

from pathlib import Path
from typing import TypeVar

from models.schemas import (
    AlertDetail,
    Asset,
    AuthenticationEvent,
    ExperienceData,
    ResponsePolicy,
    SecurityScenario,
)
from pydantic import BaseModel
from services.data_loader import DATA_DIR, load_json_list, load_json_object

ModelT = TypeVar("ModelT", bound=BaseModel)


class ScenarioNotFoundError(LookupError):
    """No scenario exists with the requested scenario_id."""


class RecordNotFoundError(LookupError):
    """A supporting record (alert, asset or policy) referenced by a scenario is missing."""


def _load_models(filename: str, model: type[ModelT], data_dir: Path) -> list[ModelT]:
    return [model.model_validate(item) for item in load_json_list(filename, data_dir)]


def _find_one(records: list[ModelT], field: str, value: str, kind: str) -> ModelT:
    for record in records:
        if getattr(record, field) == value:
            return record
    raise RecordNotFoundError(f"No {kind} found with {field}='{value}'")


def get_all_scenarios(data_dir: Path = DATA_DIR) -> list[SecurityScenario]:
    scenarios = _load_models("scenarios.json", SecurityScenario, data_dir)
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scenario_id values in scenarios.json: {ids}")
    return scenarios


def get_scenario(scenario_id: str, data_dir: Path = DATA_DIR) -> SecurityScenario:
    scenarios = get_all_scenarios(data_dir)
    for scenario in scenarios:
        if scenario.scenario_id == scenario_id:
            return scenario
    known = ", ".join(scenario.scenario_id for scenario in scenarios)
    raise ScenarioNotFoundError(f"Unknown scenario_id '{scenario_id}'. Available: {known}")


def get_alert_detail(alert_id: str, data_dir: Path = DATA_DIR) -> AlertDetail:
    records = _load_models("alert_details.json", AlertDetail, data_dir)
    return _find_one(records, "alert_id", alert_id, "alert detail")


def get_authentication_events(alert_id: str, data_dir: Path = DATA_DIR) -> list[AuthenticationEvent]:
    events = _load_models("authentication_events.json", AuthenticationEvent, data_dir)
    return [event for event in events if event.alert_id == alert_id]


def get_asset(asset_id: str, data_dir: Path = DATA_DIR) -> Asset:
    records = _load_models("assets.json", Asset, data_dir)
    return _find_one(records, "asset_id", asset_id, "asset")


def get_policy(policy_id: str, data_dir: Path = DATA_DIR) -> ResponsePolicy:
    records = _load_models("policies.json", ResponsePolicy, data_dir)
    return _find_one(records, "policy_id", policy_id, "policy")


def get_all_policies(data_dir: Path = DATA_DIR) -> list[ResponsePolicy]:
    return _load_models("policies.json", ResponsePolicy, data_dir)


def get_experience(data_dir: Path = DATA_DIR) -> ExperienceData:
    return ExperienceData.model_validate(load_json_object("experience.json", data_dir))
