"""Final-submission tests: documentation accuracy, wording, rubric mapping and the live-LLM presentation.

The README quotes real numbers; these tests recompute them from the agents so the documents cannot drift.
No Ollama is contacted (the live-presentation test uses a simulated adapter).
"""

import getpass
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agents import SimpleReflexAgent, UtilityBasedAgent
from models.comparison_models import AGENT_ORDER
from models.learning_models import FeedbackOutcome
from models.schemas import AgentType
from services.comparison_service import LearningFeedbackDemo, ModelBasedMemoryDemo
from services.scenario_service import get_all_scenarios, get_scenario
from tests.app_helpers import (
    REQUIRED_HEADINGS,
    ROOT,
    all_text,
    comparison_rows,
    expander,
    expander_text,
    run_all_agents,
    run_app,
)
from tests.fakes.service import THINKING_MARKER, SimulatedLiveAdapter, mock_service
from ui.formatters import comparison_rows as formatted_rows

pytestmark = pytest.mark.usefixtures("clean_runtime_environment")

README = (ROOT / "README.md").read_text(encoding="utf-8")
SECTIONS = (
    "1. Overview", "2. What the Application Produces", "3. Architecture", "4. Shared Security Scenarios",
    "5. Agent Comparison", "6. Five Classical Agents", "7. Security Evidence Tools", "8. LLM Agent",
    "9. Local Ollama Setup", "10. Streamlit Interface", "11. Safety", "12. Testing", "13. Running the Application",
    "14. Screenshots", "15. Limitations", "16. Project Structure", "17. Assignment Rubric Mapping",
)
# Assembled from fragments so this file does not contain the identifiers it forbids.
FORBIDDEN_CASE_INSENSITIVE = ["la" + "ptop", "inv" + "esco", "corp" + "orate", "ub" + "untu", "ama" + "zon",
                              "ep" + "yc", "v" + "cpu", "uni" + "versity"]
FORBIDDEN_CASE_SENSITIVE = [r"\bBI" + r"TS\b", r"\bV" + r"M\b"]
SUBMISSION_ENV = {
    "LLM_RUNTIME_MODE": "LIVE", "ENABLE_MOCK_LLM": "false", "RUN_OLLAMA_INTEGRATION": "true",
    "SUBMISSION_MODE": "true", "OLLAMA_MODEL": "qwen3:4b",
}


def project_text_files() -> list[Path]:
    skip = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    suffixes = {".py", ".md", ".toml", ".txt", ".ini", ".example", ".json", ".cfg"}
    return [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in suffixes and not skip & set(p.parts)]


# ------------------------------------------------------------------------------ README structure and wording


def test_readme_has_every_required_section_in_order() -> None:
    headings = re.findall(r"^## (.+)$", README, re.M)
    assert [h for h in headings if h in SECTIONS] == list(SECTIONS)
    assert README.startswith("# Security Alert Triage and Prioritization Assistant")


def test_readme_output_contract_lists_all_eight_items() -> None:
    section = README.split("## 2. What the Application Produces")[1].split("## 3.")[0]
    for item in ("Priority", "Decision", "Reasoning", "Evidence", "Uncertainty", "Likely interpretation",
                 "Recommended investigation / escalation steps", "Audit note"):
        assert f"**{item}**" in section, item


def test_readme_reads_plainly_and_makes_no_marks_claims() -> None:
    lowered = README.lower()
    for phrase in ("robust", "production-grade", "enterprise-grade", "world-class", "non-negotiable", "20+ years",
                   "principal engineer", "this demonstrates", "the system ensures", "full marks", "guaranteed",
                   "estimated marks"):
        assert phrase not in lowered, phrase
    assert "we built" in lowered and "our code" in lowered


def test_readme_rubric_mapping_table_is_complete() -> None:
    table = README.split("## 17. Assignment Rubric Mapping")[1]
    for row in ("Five classical agents", "LLM + tool calling", "Shared scenarios/testing", "Streamlit/comparison",
                "Learning/evaluation", "Documentation/demo"):
        assert f"| {row} |" in table, row


def test_readme_screenshot_section_uses_placeholders_without_fake_images() -> None:
    names = ("main-comparison", "scenario-a", "scenario-b", "scenario-c", "model-based-memory", "utility-ranking",
             "learning-before", "learning-after", "llm-tool-calling", "tests")
    for name in names:
        assert f"(screenshots/{name}.png)" in README, name
    assert (ROOT / "screenshots").is_dir()
    images = [p for p in (ROOT / "screenshots").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    assert images == [], "screenshots must come from a real run; none are generated"


def test_readme_documents_localhost_only_ollama_settings() -> None:
    for line in ("OLLAMA_BASE_URL=http://127.0.0.1:11434", "OLLAMA_MODEL=qwen3:4b", "OLLAMA_TEMPERATURE=0",
                 "OLLAMA_REQUEST_TIMEOUT_SECONDS=300", "OLLAMA_MAX_TOOL_ITERATIONS=8", "OLLAMA_MINIMUM_TOOL_CALLS=2"):
        assert line in README, line
    assert "No API key is required" in README and "bound to localhost" in README


def test_readme_states_the_two_distinct_tool_rule_and_has_no_reasoning_discussion() -> None:
    assert "At least two distinct relevant tools must complete successfully before the LLM result is accepted." in README
    for word in ("chain-of-thought", "chain of thought", "thinking", "reasoning_content"):
        assert word not in README.lower(), word


def test_documents_say_live_validation_is_pending_and_fabricate_nothing() -> None:
    results = (ROOT / "docs" / "test_results.md").read_text(encoding="utf-8")
    assert "Pending live Ollama validation" in README and "Pending live Ollama validation" in results
    rows = [ln for ln in results.splitlines() if ln.startswith("|") and any(
        k in ln for k in ("validate_ollama", "-m integration", "run_llm_scenarios"))]
    scenario_rows = [ln for ln in results.splitlines() if re.match(r"\| [ABC] \|", ln)]
    assert len(rows) == 4 and len(scenario_rows) == 3
    assert all("Pending live run" in ln for ln in rows + scenario_rows)
    assert "validation passed" not in README.lower() and "integration tests passed" not in README.lower()


@pytest.mark.parametrize("name", ["submission_summary.md", "demo_guide.md", "test_results.md", "local_ollama_setup.md"])
def test_required_documents_exist_and_are_not_empty(name: str) -> None:
    assert len((ROOT / "docs" / name).read_text(encoding="utf-8").strip()) > 500


def test_submission_summary_has_the_requested_sections() -> None:
    text = (ROOT / "docs" / "submission_summary.md").read_text(encoding="utf-8")
    for heading in ("Problem", "Solution", "Agents", "Scenarios", "LLM and Tools", "Learning Demonstration",
                    "Streamlit Comparison", "Testing", "Safety and Limitations", "How the Rubric is Addressed"):
        assert f"## {heading}" in text, heading
    assert "marks" not in text.lower()


def test_demo_guide_covers_the_fifteen_step_flow_and_one_sentence_explanations() -> None:
    text = (ROOT / "docs" / "demo_guide.md").read_text(encoding="utf-8")
    assert re.findall(r"^(\d+)\. \*\*", text, re.M) == [str(n) for n in range(1, 16)]
    flat = " ".join(text.split())
    for sentence in ("Uses current conditions and fixed rules.", "Uses current conditions plus remembered state.",
                     "Chooses a plan that satisfies a security goal.", "Ranks safe actions using expected utility.",
                     "Changes future preference using explicit outcome feedback.",
                     "Uses a local language model to dynamically gather evidence through tools before producing a "
                     "grounded recommendation."):
        assert sentence in flat, sentence


def test_no_environment_specific_wording_anywhere_in_the_repository() -> None:
    sensitive = re.compile("|".join(FORBIDDEN_CASE_SENSITIVE))
    insensitive = re.compile("|".join(FORBIDDEN_CASE_INSENSITIVE), re.IGNORECASE)
    offenders = [str(p.relative_to(ROOT)) for p in project_text_files()
                 if sensitive.search(p.read_text(encoding="utf-8", errors="ignore"))
                 or insensitive.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert offenders == []


def test_the_developer_username_and_home_path_are_not_in_the_repository() -> None:
    names = {n.lower() for n in (getpass.getuser(), Path.home().name) if len(n) >= 4}
    offenders = [str(p.relative_to(ROOT)) for p in project_text_files()
                 if any(n in p.read_text(encoding="utf-8", errors="ignore").lower() for n in names)]
    assert offenders == []


def test_no_env_file_and_the_example_has_only_the_documented_settings() -> None:
    assert not (ROOT / ".env").exists()
    keys = {ln.split("=")[0] for ln in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#") and "=" in ln}
    assert keys == {
        "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_TEMPERATURE", "OLLAMA_REQUEST_TIMEOUT_SECONDS",
        "OLLAMA_MAX_TOOL_ITERATIONS", "OLLAMA_MINIMUM_TOOL_CALLS", "OLLAMA_VALIDATE_MODEL_ON_INIT",
        "OLLAMA_ALLOW_REMOTE_HOST", "LLM_RUNTIME_MODE", "ENABLE_MOCK_LLM", "RUN_OLLAMA_INTEGRATION",
        "SUBMISSION_MODE", "OLLAMA_DEBUG"}


# ------------------------------------------------------------------------------ numbers in the docs match the code


def test_scenario_table_matches_the_scenario_data() -> None:
    a, b, c = get_all_scenarios()
    assert (a.severity.value, b.severity.value, c.severity.value) == ("HIGH", "LOW", "HIGH")
    assert a.failed_login_count == 12
    assert (a.correlated_alert_count, b.correlated_alert_count, c.correlated_alert_count) == (2, 0, 3)
    assert (a.asset_criticality.value, b.asset_criticality.value, c.asset_criticality.value) == ("HIGH", "LOW", "CRITICAL")
    assert not (a.confirmed_impact or b.confirmed_impact or c.confirmed_impact)
    assert (a.account_privilege.value, b.account_privilege.value, c.account_privilege.value) == (
        "PRIVILEGED", "STANDARD", "SERVICE")
    assert "12 failed logins" in README and "| Related alerts | 2 | 0 | 3 |" in README


def test_utility_numbers_and_weights_in_the_readme_match_the_agent() -> None:
    metadata = UtilityBasedAgent().decide(get_scenario("A")).metadata
    for item in metadata["action_ranking"]:
        assert f"| {item['rank']} | {item['decision']} | {item['expected_utility']:.4f} |" in README
    weights = {c["name"]: c["weight"] for c in metadata["utility_profile"]["criteria"]}
    assert weights == {"RISK_REDUCTION": 0.25, "ASSET_PROTECTION": 0.2, "EVIDENCE_ALIGNMENT": 0.2,
                       "PROPORTIONALITY": 0.15, "HUMAN_CONTROL": 0.1, "OPERATIONAL_CONTINUITY": 0.1}
    for label, weight in (("Risk Reduction", "0.25"), ("Asset Protection", "0.20"), ("Evidence Alignment", "0.20"),
                          ("Proportionality", "0.15"), ("Human Control", "0.10"), ("Operational Continuity", "0.10")):
        assert f"| {label} | {weight} |" in README


def test_learning_numbers_in_the_readme_match_the_agent() -> None:
    scenario = get_scenario("A")
    demo = LearningFeedbackDemo()
    try:
        def rates() -> dict[str, tuple[int, int, float]]:
            experience = demo.decide(scenario).metadata["candidate_experience"]
            return {e["decision"]: (e["success"], e["failure"], round(e["smoothed_success_rate"], 4))
                    for e in experience if e["eligible"]}

        assert rates() == {"INVESTIGATE": (8, 2, 0.75), "ESCALATE_TO_SECURITY_ANALYST": (5, 3, 0.6)}
        assert "success 8, failure 2, smoothed 0.7500" in README and "success 5, failure 3, smoothed 0.6000" in README
        for _ in range(3):
            demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
        assert rates()["INVESTIGATE"] == (8, 5, 0.6) and demo.decide(scenario).decision == "INVESTIGATE"
        assert demo.decide(scenario).metadata["tie_breaker_used"] is True
        assert "| After 3 recorded failures | 8 / 5, smoothed 0.6000 |" in README
        demo.record_feedback(scenario, FeedbackOutcome.FAILURE)
        assert rates()["INVESTIGATE"] == (8, 6, 0.5625)
        assert demo.decide(scenario).decision == "ESCALATE_TO_SECURITY_ANALYST"
        assert "| After 4 recorded failures | 8 / 6, smoothed 0.5625 |" in README
    finally:
        demo.close()
    assert "(S + 1) / (S + F + 2)" in README


def test_model_based_numbers_in_the_readme_match_the_agent() -> None:
    demo = ModelBasedMemoryDemo()
    first, second = demo.observe(get_scenario("A")), demo.observe(get_scenario("A"))
    assert (first.matched_rule_id, first.priority.value, first.decision) == ("MB-03", "HIGH", "INVESTIGATE")
    assert (second.matched_rule_id, second.priority.value, second.decision) == (
        "MB-02", "CRITICAL", "ESCALATE_TO_SECURITY_ANALYST")
    assert "rule MB-03" in README and "rule MB-02" in README


def test_simple_reflex_rules_in_the_readme_match_the_agent() -> None:
    for scenario_id, rule in (("A", "SR-02"), ("B", "SR-03"), ("C", "SR-01")):
        assert SimpleReflexAgent().decide(get_scenario(scenario_id)).metadata["matched_rule_id"] == rule
        assert f"| {scenario_id} | {rule} " in README


# ------------------------------------------------------------------------------ comparison and presentation


def test_all_six_agents_return_the_full_output_contract_for_every_scenario() -> None:
    for scenario in get_all_scenarios():
        result = mock_service().compare(scenario)
        assert tuple(e.agent_type for e in result.results) == AGENT_ORDER and result.all_succeeded
        for entry in result.results:
            for field in ("priority", "decision", "reasoning_summary", "evidence", "uncertainty",
                          "likely_interpretation", "recommended_actions", "audit_note"):
                assert getattr(entry, field), (entry.agent_name, scenario.scenario_id, field)


def test_the_comparison_table_has_the_columns_the_brief_asks_for() -> None:
    result = mock_service().compare(get_scenario("A"))
    rows = formatted_rows(result)
    assert list(rows[0]) == ["Agent", "Mechanism", "Priority", "Decision", "Runtime"]
    assert [r["Agent"] for r in rows] == [
        "Simple Reflex", "Model-Based Reflex", "Goal-Based", "Utility-Based", "Learning Agent", "LLM Agent"]
    assert AgentType.LLM in AGENT_ORDER and len(AGENT_ORDER) == 6


def test_the_live_llm_presentation_shows_runtime_model_tools_sources_and_validation(
    clean_runtime_environment: pytest.MonkeyPatch,
) -> None:
    import agents.llm_agent as llm_agent_module

    for name, value in SUBMISSION_ENV.items():
        clean_runtime_environment.setenv(name, value)
    clean_runtime_environment.setattr(llm_agent_module, "OllamaChatModelAdapter", SimulatedLiveAdapter)
    SimulatedLiveAdapter.leak_thinking = True
    try:
        at = run_all_agents(run_app())
    finally:
        SimulatedLiveAdapter.leak_thinking = False
    assert not at.exception
    block = expander(at, "LLM Agent - ")
    text = expander_text(block)
    for heading in REQUIRED_HEADINGS:  # the eight required fields come first
        assert heading in text, heading
    assert "Runtime: **OLLAMA**" in text and "Model: `qwen3:4b`" in text
    trace = block.dataframe[0].value
    assert trace["Tool"].nunique() >= 2  # at least two distinct tools are visible
    assert set(trace["Status"]) == {"SUCCESS"} and all(trace["Evidence returned"]) and all(trace["Sources"])
    assert "Grounding validation" in text and "Guardrail validation" in text and "checks passed" in text
    page = (all_text(at) + text).lower()
    assert THINKING_MARKER.lower() not in page
    for hidden in ("thinking", "chain of thought", "chain-of-thought", "raw prompt", "traceback"):
        assert hidden not in page
    assert comparison_rows(at)[5]["Runtime"] == "OLLAMA"


# ------------------------------------------------------------------------------ the final validation script


def test_the_final_validation_script_succeeds_without_a_live_model() -> None:
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_final_submission.py")], cwd=ROOT,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Final submission structure validated." in run.stdout
    assert "Live Ollama integration results must still come from an actual live run." in run.stdout
    assert "[FAIL]" not in run.stdout
