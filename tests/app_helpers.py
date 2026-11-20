"""Helpers for driving the real Streamlit app headlessly with Streamlit's AppTest (no browser, no server)."""

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

REQUIRED_HEADINGS = (
    "Priority", "Decision", "Why / reasoning", "Evidence", "Uncertainty", "Likely interpretation",
    "Recommended investigation / escalation steps", "Audit note",
)


def run_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=180).run()


def run_all_agents(at: AppTest) -> AppTest:
    return at.sidebar.button(key="run_all").click().run()


def select_scenario(at: AppTest, scenario_id: str) -> AppTest:
    return at.sidebar.selectbox(key="scenario_id").select(scenario_id).run()


def expander(at: AppTest, label_start: str) -> Any:
    matches = [e for e in at.expander if e.label.startswith(label_start)]
    assert len(matches) == 1, f"expected one expander starting with {label_start!r}, found {len(matches)}"
    return matches[0]


def expander_text(block: Any) -> str:
    return "\n".join(str(item.value) for item in [*block.markdown, *block.caption])


def comparison_rows(at: AppTest) -> list[dict[str, str]]:
    """The six-agent table (the first dataframe on the page after a run)."""
    return at.dataframe[0].value.to_dict("records")


def all_text(at: AppTest) -> str:
    """Every piece of text the page shows, including table cells and button labels."""
    parts: list[str] = []
    for group in (at.title, at.header, at.subheader, at.markdown, at.caption, at.text, at.info, at.warning,
                  at.error, at.success):
        parts += [str(item.value) for item in group]
    parts += [str(button.label) for button in at.button]
    for frame in at.dataframe:
        parts.append(frame.value.to_string())
    parts += [e.label for e in at.expander]
    return "\n".join(parts)
