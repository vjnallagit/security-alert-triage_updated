# Security Alert Triage and Prioritization Assistant

I built this project as part of my Agentic AI assignment to compare how five classical AI agents and one LLM-based agent respond to the same simulated security alert.

The project uses synthetic security data only. Each agent receives the same scenario and independently recommends whether the alert should be monitored, investigated, escalated, or needs more evidence.

The application is recommendation-only and does not perform real security actions.

## 1. Project Objective

The user picks a scenario in the Streamlit app. The same scenario is sent to all six agents. Each agent uses a different decision method, and the app shows the results side by side so they are easy to compare.

```text
Select Scenario
      |
      v
Same Security Scenario
      |
      +--> Simple Reflex
      +--> Model-Based Reflex
      +--> Goal-Based
      +--> Utility-Based
      +--> Learning Agent
      +--> LLM Agent
      |
      v
Streamlit Comparison
```

The agents recommend one of four safe actions: `MONITOR`, `COLLECT_MORE_EVIDENCE`, `INVESTIGATE` or `ESCALATE_TO_SECURITY_ANALYST`.

## 2. Quick Start

```bash
cd security-alert-triage        # use your actual project folder name

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

ollama pull llama3.2:1b

set -a
source .env
set +a

streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Then:

1. Open the Streamlit URL.
2. Select Scenario A, B, or C.
3. Click **Run All Agents**.
4. Wait for the local LLM to finish.
5. Review the six-agent comparison.

You need a `.env` file first. See section 13.

## 3. Agents Implemented

| Agent | Main idea |
|---|---|
| Simple Reflex | Uses current facts and rules |
| Model-Based Reflex | Uses current facts plus internal state |
| Goal-Based | Selects a safe action that satisfies a defensive goal |
| Utility-Based | Ranks safe actions using expected utility |
| Learning Agent | Uses historical success/failure outcomes |
| LLM Agent | Selects evidence functions and uses returned evidence |

### Simple Reflex Agent

- Uses the current scenario facts only.
- Applies condition-action rules (for example SR-01, SR-02, SR-03).
- Has no historical memory.
- Source: `agents/simple_reflex.py`

### Model-Based Reflex Agent

- Uses the current facts plus an internal state (observations of the account, asset and alert type).
- Earlier observations can change a later decision.
- The main comparison uses a fresh agent every time, so the comparison stays fair.
- A separate "How Model-Based Memory Changes a Decision" panel in the UI reuses the agent on purpose to show the memory effect.
- Source: `agents/model_based_reflex.py`

### Goal-Based Agent

- Has one defensive goal (GB-G01, safe and proportionate triage).
- Checks the candidate response plans against that goal.
- Selects the least disruptive plan that satisfies it, and shows why other plans were rejected.
- Source: `agents/goal_based.py`

### Utility-Based Agent

- Compares the four safe actions.
- Calculates an expected utility for each one (profile UP-01).
- Ranks them and picks the highest.
- The values are simulated assumptions, not real statistics.
- Source: `agents/utility_based.py`

### Learning Agent

- Uses synthetic historical outcomes from `data/experience.json`.
- Looks at success and failure counts and a smoothed success rate.
- History picks between actions that are already safe for the scenario.
- History changes only when feedback is recorded explicitly.
- Source: `agents/learning_agent.py`

### LLM Agent

- Uses a local Ollama model.
- Receives the selected scenario.
- Chooses read-only evidence functions by itself. Python validates and executes them.
- At least two distinct successful evidence functions are needed.
- The returned evidence is used for the recommendation.
- Source: `services/simple_llm_agent.py`

## 4. Test Scenarios

| Scenario | Description | Demonstrated final result |
|---|---|---|
| A | Repeated failed logins against a privileged account | HIGH / INVESTIGATE |
| B | Low-severity alert with limited evidence | LOW / COLLECT_MORE_EVIDENCE |
| C | Correlated unusual activity on a critical service | CRITICAL / ESCALATE_TO_SECURITY_ANALYST |

These are demonstrated results. I read them from the saved Streamlit runs in `docs/Scenario A.pdf`, `docs/Scenario B.pdf` and `docs/Scenario C.pdf`.

The scenario facts shown in those runs:

| Scenario | Severity | Account | Source | Failed logins | Asset criticality | Correlated alerts | Confirmed impact | Evidence |
|---|---|---|---|---|---|---|---|---|
| A | High | Privileged | Unfamiliar | 12 | High | 2 | No | Strong |
| B | Low | Standard | Unknown | 1 | Low | 0 | No | Limited |
| C | High | Service | Known | 1 | Critical | 3 | No | Conflicting |

All six agents receive the same selected scenario. The app also prints a short scenario fingerprint (a hash of the scenario data) under the comparison table. It is the same for all six agents, which helps confirm they got identical input. The fingerprints in the saved runs are `008c9840b06c` (A), `4276b656d217` (B) and `41130af1dd09` (C).

## 5. Final Scenario Comparison

| Scenario | Simple Reflex | Model-Based | Goal-Based | Utility-Based | Learning | LLM |
|---|---|---|---|---|---|---|
| A | HIGH / INVESTIGATE | HIGH / INVESTIGATE | HIGH / INVESTIGATE | HIGH / INVESTIGATE | HIGH / INVESTIGATE | HIGH / INVESTIGATE |
| B | LOW / COLLECT_MORE_EVIDENCE | LOW / COLLECT_MORE_EVIDENCE | LOW / COLLECT_MORE_EVIDENCE | LOW / COLLECT_MORE_EVIDENCE | LOW / COLLECT_MORE_EVIDENCE | LOW / COLLECT_MORE_EVIDENCE |
| C | CRITICAL / ESCALATE_TO_SECURITY_ANALYST | CRITICAL / ESCALATE_TO_SECURITY_ANALYST | CRITICAL / ESCALATE_TO_SECURITY_ANALYST | CRITICAL / ESCALATE_TO_SECURITY_ANALYST | CRITICAL / ESCALATE_TO_SECURITY_ANALYST | CRITICAL / ESCALATE_TO_SECURITY_ANALYST |

The agents can reach the same final result while using different decision methods.

- Simple Reflex uses a matching rule (SR-02, SR-03, SR-01 for A, B, C).
- Model-Based also considers state (rules MB-03, MB-05, MB-01, each on the first observation).
- Goal-Based checks defensive goals and rejects unsuitable plans.
- Utility-Based ranks actions numerically (top utility 0.7812, 0.8139 and 0.7502).
- Learning uses historical outcomes (smoothed rates 75.0%, 83.3% and 80.0% for the chosen action).
- LLM gathers and interprets evidence through function calling.

## 6. LLM Agent and Function Calling

```text
Security Scenario
      |
      v
Local LLM
      |
      v
Select Evidence Function 1
      |
      v
Python Executes Function
      |
      v
Select Different Evidence Function 2
      |
      v
Python Executes Function
      |
      v
Collected Evidence
      |
      v
Final Recommendation
```

The LLM does not directly execute security actions. The model selects evidence functions, while Python controls and validates their execution.

- Model: `llama3.2:1b`
- Runtime: Ollama, running locally on `127.0.0.1:11434`
- At least two distinct evidence functions must succeed before the final recommendation. The code requires this (`MINIMUM_DISTINCT_TOOLS = 2`).
- In all three saved runs the model selected `inspect_alert` and `inspect_asset_criticality`, and both returned SUCCESS.
- The final answer contains priority, decision, uncertainty, likely interpretation, recommended actions and an audit note.

The "Run All Agents" button uses `services/llm_runtime.py` and `services/comparison_service.py`, which call `run_llm_triage()` in `services/simple_llm_agent.py`. The model name and host are constants in that file.

### LangChain Status

The assignment rubric mentions a local Ollama model through LangChain. In the final active path this is only partly true.

- The model call uses `ollama.Client` directly. `ChatOllama` and `bind_tools()` are **not** used in the "Run All Agents" path.
- LangChain is used only for the evidence tools. `tools/langchain_security_tools.py` wraps them as `StructuredTool` objects (`SecurityToolbox`), and Python runs them.
- Older LangChain-based LLM code (`agents/llm_agent.py`, `services/llm_orchestrator.py` and related files) is still in the repository, but the final Streamlit workflow does not use it.

So this is a known gap against the rubric wording. I documented it instead of claiming otherwise.

### Policy Calibration

The LLM first proposes a recommendation using the collected evidence. A deterministic defensive-security policy layer then calibrates the final priority and decision from the retrieved evidence. This helps prevent a small local model from consistently over-prioritizing or under-prioritizing alerts.

- The calibration looks at the highest severity or asset criticality in the evidence the tools returned (CRITICAL gives CRITICAL / ESCALATE, HIGH gives HIGH / INVESTIGATE, MEDIUM gives MEDIUM / INVESTIGATE, anything else gives LOW / COLLECT_MORE_EVIDENCE).
- According to the code comments, it works from retrieved evidence and not from the scenario letter.
- The model's own proposal is kept in the result metadata, but the displayed final result is the calibrated one. So the LLM agreeing with the other agents is partly because of this policy step.
- A safety check also rejects wording that claims an action was executed.

## 7. Evidence Functions

| Evidence tool | Purpose |
|---|---|
| `get_alert_details()` | Reads synthetic alert details |
| `search_authentication_events()` | Searches synthetic authentication observations |
| `get_asset_criticality()` | Reads asset criticality |
| `get_response_policy()` | Reads synthetic response-policy information |

The LLM sees small wrapper functions. These wrappers call the underlying validated project evidence tools.

| Function the LLM sees | Underlying tool |
|---|---|
| `inspect_alert` | `get_alert_details()` |
| `inspect_authentication_activity` | `search_authentication_events()` |
| `inspect_asset_criticality` | `get_asset_criticality()` |
| `inspect_response_policy` | `get_response_policy()` |

All evidence functions are local, synthetic, and read-only. They read the JSON files in `data/`.

Source record IDs shown in the saved runs: `ALERT-A-001` and `ASSET-IDP-01` (A), `ALERT-B-001` and `ASSET-WIKI-01` (B), `ALERT-C-001` and `ASSET-PAY-01` (C).

## 8. Learning Agent

The Learning Agent uses synthetic historical outcomes for different actions. It looks at success and failure counts and calculates a smoothed success rate.

The agent first checks which actions are safe for the current scenario. It then uses historical results to choose between eligible actions.

Learning history changes only when explicit feedback is recorded.

- Smoothing is Beta(1,1) / Laplace: `(successes + 1) / (successes + failures + 2)`. For Scenario A, INVESTIGATE has 8 successes and 2 failures, so `(8 + 1) / (8 + 2 + 2) = 75.0%`.
- In Scenario C only ESCALATE_TO_SECURITY_ANALYST is eligible, so history does not change that result.
- The feedback demonstration in the UI writes to a temporary copy of the experience data.
- The submitted seed file, `data/experience.json`, stays unchanged.

## 9. Project Structure

```text
security-alert-triage/
├── .streamlit/      # local-only Streamlit settings
├── agents/          # agent implementations
├── config/          # application and LLM configuration
├── data/            # synthetic scenario, evidence and experience data
├── devtools/        # offline stand-in LLM used only by tests/demos
├── docs/            # scenario evidence PDFs and notes
├── models/          # domain and validation models
├── notebooks/       # LLM function-calling demonstration
├── screenshots/     # placeholder folder (no images committed)
├── scripts/         # helper and validation scripts
├── services/        # scenario, comparison and LLM services
├── tests/           # automated tests
├── tools/           # read-only synthetic evidence tools
├── ui/              # Streamlit components
├── app.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## 10. Prerequisites

- Python 3.11 or newer
- `pip`
- Ollama installed locally
- The packages in `requirements.txt`: pydantic, pytest, langchain-core, langchain-ollama and streamlit

The LLM code imports the `ollama` package directly. It is not listed in `requirements.txt` by itself, but it comes in as a dependency of `langchain-ollama`.

## 11. Detailed Setup

```bash
cd security-alert-triage

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

## 12. Ollama Setup

```bash
ollama pull llama3.2:1b
```

Check installed models:

```bash
ollama list
```

The model only needs to be downloaded once on a persistent machine. On a new or temporary VM it may need to be downloaded again.

Make sure the Ollama service is running on `127.0.0.1:11434` before you click **Run All Agents**.

## 13. Environment Configuration

Runtime configuration comes from environment variables, which I keep in a local `.env` file. The real `.env` is not part of this README and should never be committed (it is in `.gitignore`).

There is no `.env.example` file in this copy of the repository, so create `.env` yourself with safe values like these:

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:1b
LLM_RUNTIME_MODE=LIVE
RUN_OLLAMA_INTEGRATION=true
ENABLE_MOCK_LLM=false
```

What they do:

- `RUN_OLLAMA_INTEGRATION=true` and `LLM_RUNTIME_MODE=LIVE` turn on the live Ollama runtime in the UI. Without them the LLM Agent is not offered.
- `OLLAMA_MODEL` must be set for the live runtime to be accepted. The model actually called is the constant `llama3.2:1b` in `services/simple_llm_agent.py`, so keep the two values the same.
- `OLLAMA_BASE_URL` must point to a local host. The active LLM function uses the fixed address `http://127.0.0.1:11434`.
- `ENABLE_MOCK_LLM=false` keeps the offline stand-in disabled.

On Linux or the BITS VM, load the file before starting Streamlit:

```bash
set -a
source .env
set +a
```

This loads the environment variables into the current terminal before Streamlit is started.

A newly created VM may need the environment and the Ollama model to be prepared again.

## 14. Run the Application

```bash
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

1. Open the Streamlit URL.
2. Select Scenario A, B, or C in the sidebar.
3. Confirm the LLM Agent runtime in the sidebar (OLLAMA).
4. Click **Run All Agents**.
5. Wait for all six agents to complete.
6. Review the comparison table.
7. Expand each agent section to see its details.

The classical agents run quickly. CPU-based local LLM inference can take longer.

The page also has two small demonstrations (Model-Based memory and Learning feedback) and a requirement coverage table at the bottom.

## 15. How to Check the Output

The comparison table should show six agents: Simple Reflex, Model-Based Reflex, Goal-Based, Utility-Based, Learning Agent and LLM Agent. Below it you should see a line saying all six agents received the identical scenario, with the fingerprint.

Compare what you see with the saved runs.

### Scenario A

```text
HIGH / INVESTIGATE
```

### Scenario B

```text
LOW / COLLECT_MORE_EVIDENCE
```

### Scenario C

```text
CRITICAL / ESCALATE_TO_SECURITY_ANALYST
```

For the LLM Agent, open its section and check that:

- the local runtime (OLLAMA) and the model (`llama3.2:1b`) are visible;
- at least two evidence functions were selected;
- both functions succeeded;
- source record IDs are shown (for example `ALERT-A-001`, `ASSET-IDP-01`);
- priority is present;
- decision is present;
- uncertainty is present;
- likely interpretation is present;
- recommended actions are present;
- the audit note says no remediation action was executed.

The local model can vary, so a live run might not match the saved runs word for word. The priority and decision should still follow the evidence.

## 16. How to Interpret the Output

### Priority

Priority describes urgency or risk.

- **LOW**: Low concern based on available evidence.
- **MEDIUM**: Suspicious evidence exists, but risk or impact is moderate.
- **HIGH**: Evidence is important enough for investigation.
- **CRITICAL**: Potential impact is high enough for prompt human security review.

A high priority does not automatically mean that compromise is confirmed.

### Decision

- **MONITOR**: Continue observing the alert.
- **COLLECT_MORE_EVIDENCE**: Current evidence is not strong enough for a stronger decision.
- **INVESTIGATE**: Evidence is suspicious enough for analyst investigation.
- **ESCALATE_TO_SECURITY_ANALYST**: Potential impact is high enough for prompt human review.

Escalation is a recommendation only. The application does not perform containment.

### Evidence

Facts used to support the decision, such as severity, account type, failed login count, asset criticality and correlated alerts.

### Uncertainty

What the system still does not know. Example from the saved runs:

```text
Compromise or confirmed impact is not established by the retrieved evidence.
```

### Likely interpretation

A cautious explanation of what the alert may represent.

### Recommended actions

Safe next steps for a human analyst, for example reviewing authentication events or escalating to an analyst.

### Audit note

A short trace of how the recommendation was created, plus confirmation that no remediation was executed.

## 17. Why the Agents Can Give Different Results

Different agents use different decision methods.

- Simple Reflex uses a matching rule.
- Model-Based also uses internal state.
- Goal-Based evaluates defensive goals.
- Utility-Based calculates numerical scores.
- Learning uses historical outcomes.
- LLM selects and interprets evidence.

Different outputs do not automatically mean that an agent is wrong. The purpose of the project is to compare how different architectures make decisions.

Even when all agents reach the same final decision, the reasoning method is still different. In my three saved runs all six agents agree, so the difference is in the explanation each one gives, not in the final answer.

## 18. Testing

Offline test command:

```bash
python -m pytest -m "not integration" -q
```

The recorded result in `docs/test_results.md` is:

```text
1379 passed, 0 failed
```

I did not re-run the tests while updating this README. Run the offline suite and confirm zero failures before submission.

Tests marked `integration` need a live Ollama server and `RUN_OLLAMA_INTEGRATION=true`. They are skipped otherwise, and `docs/test_results.md` lists them as pending.

The test suite covers:

- classical-agent logic;
- shared scenarios;
- state behavior;
- utility calculations;
- learning;
- evidence tools;
- comparison;
- LLM error handling;
- safety;
- edge cases.

One honest note: no test calls `run_llm_triage()` directly. The active LLM path is shown by the saved live runs. The offline LLM tests cover the earlier orchestration code.

## 19. Evidence and Screenshots

```text
docs/Scenario A.pdf
docs/Scenario B.pdf
docs/Scenario C.pdf
```

These PDFs were captured from actual Streamlit runs and include the scenario summary, six-agent comparison, and detailed agent output. Each one has six pages.

The `screenshots/` folder only contains a `README.md` and a `.gitkeep`. It has no image files.

## 20. Troubleshooting

### `RUN_OLLAMA_INTEGRATION` configuration error

The environment was not loaded. Load it before starting Streamlit:

```bash
set -a
source .env
set +a
```

Then start Streamlit from the same terminal.

### Model not found

Check:

```bash
ollama list
```

If `llama3.2:1b` is missing:

```bash
ollama pull llama3.2:1b
```

### LLM Agent unavailable

The app shows a message like "Live Ollama is unavailable". Check that Ollama is running on `127.0.0.1:11434`, that the model is installed, and that `.env` is loaded. The classical agents still run, and the other results are not thrown away.

### Local LLM is slow

The BITS/training VM may use CPU inference. Classical agents finish quickly, but the LLM Agent can take longer.

### Fresh VM

A fresh training VM may not retain the previously downloaded Ollama model or previously loaded shell environment variables.

## 21. Safety

This project is recommendation-only and uses synthetic data.

The application does not:

- block accounts;
- disable users;
- isolate hosts;
- terminate sessions;
- modify firewalls;
- execute containment;
- execute remediation;
- perform offensive security operations.

Any escalation means human review.

## 22. Limitations

- All data is synthetic academic data and the scenario set is small (three scenarios).
- No live SIEM, no live IAM and no live threat intelligence.
- Local LLM responses can vary, and the evidence functions the model picks can vary.
- CPU-based inference can be slow.
- The LLM model call does not go through LangChain (see section 6).
- The final LLM priority and decision are calibrated by a deterministic policy layer.
- Utility values and learning history are simulated assumptions, not measured results.
- No automatic remediation. Human review is required.
- The saved runs show `Grounding validation: 0/0 checks passed`, `Guardrail validation: 0/0 checks passed` and an empty "Tool calls used" table in the LLM section. The current UI evidence shows zero registered validation checks for these counters, so the 0/0 display should not be treated as proof of validation coverage.
- The saved runs show only the starting state of the learning feedback demonstration, not a before and after change.
- The status text in the app ("Pending live Ollama validation") and some older notes under `docs/` were written for an earlier LLM setup. The code and the evidence PDFs are the source of truth.

## 23. Assignment Requirement Mapping

| Assignment requirement | Status |
|---|---|
| Five classical agents | Implemented |
| Local LLM Agent | Implemented |
| Two LLM evidence functions | Implemented |
| Same scenario for all six agents | Implemented |
| Three scenarios | Implemented |
| Streamlit comparison | Implemented |
| Learning history | Implemented |
| Defensive-only behavior | Implemented |
| Automated tests | Implemented |
| Scenario evidence | Included |
| LangChain in active LLM path | Gap: the model call uses `ollama.Client` directly. LangChain only wraps the evidence tools |
