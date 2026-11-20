# Demo Guide

A 5 to 7 minute walk-through. Run the app with the live settings from the README (`SUBMISSION_MODE=true`) so
the LLM Agent uses the local Ollama model. Live inference on CPU can take a few minutes, so press **Run All
Agents** early and talk while it runs.

## Flow

1. **Open the application.** `streamlit run app.py`. Point out the title and the note that this is an academic
   simulation that only recommends actions.
2. **Select Scenario A** in the sidebar and read out the scenario summary: privileged account, unfamiliar
   source, 12 failed logins, HIGH asset, no confirmed impact.
3. **Explain the shared input.** All six agents get the same scenario; the line under the table shows one
   fingerprint for all of them.
4. **Run all six agents** and wait for the table. Note that they agree on HIGH / INVESTIGATE for A.
5. **Simple Reflex.** Open its expander. It matched rule SR-02 and has no memory.
6. **Model-Based memory.** Open "How Model-Based Memory Changes a Decision" and run Scenario A twice: first
   HIGH / INVESTIGATE, then CRITICAL / ESCALATE_TO_SECURITY_ANALYST. Same scenario, different memory.
7. **Goal-Based planning.** Show the goal GB-G01, the rejected options and the ordered plan.
8. **Utility ranking.** Show the table (INVESTIGATE 0.7812 first) and the six weighted criteria.
9. **Learning feedback.** In the Learning demo press "Record FAILURE" four times. Show the before/after counts
   and the decision changing to ESCALATE_TO_SECURITY_ANALYST. The scenario did not change, only the history.
10. **Live LLM tool calls.** Open the LLM Agent expander: runtime OLLAMA, the model, at least two distinct tools
    with status, evidence and source IDs, then the grounding and guardrail results.
11. **Select Scenario B** and run all agents.
12. **Low-risk proportionality.** Everyone chooses LOW / COLLECT_MORE_EVIDENCE; weak evidence is not escalated.
13. **Select Scenario C** and run all agents.
14. **Ambiguity and the critical asset.** The asset is critical and several alerts correlate, but the evidence is
    mixed and nothing is confirmed, so the result is an escalation for human review, not a claim of compromise.
15. **Show the test results.** Run `python -m pytest -m "not integration" -q` and open `docs/test_results.md`.

## One-sentence explanations

- **Simple Reflex:** "Uses current conditions and fixed rules."
- **Model-Based:** "Uses current conditions plus remembered state."
- **Goal-Based:** "Chooses a plan that satisfies a security goal."
- **Utility-Based:** "Ranks safe actions using expected utility."
- **Learning:** "Changes future preference using explicit outcome feedback."
- **LLM:** "Uses a local language model to dynamically gather evidence through tools before producing a
  grounded recommendation."

## If a question comes up

- *Does it take any action?* No. It only recommends, and every policy says consequential actions need a human.
- *Is the data real?* No, all data is synthetic.
- *What if Ollama is down?* The five classical agents still run; the LLM Agent shows an error and a retry button.
- *Why is the Model-Based result in the table not CRITICAL?* The table always uses a fresh agent so the
  comparison is fair; the memory demo reuses one agent on purpose.
