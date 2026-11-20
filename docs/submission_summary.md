# Security Alert Triage and Prioritization Assistant

## Problem

A security operations team gets many alerts and has to decide which ones need attention first. A wrong call in
either direction is costly: ignoring a real attack, or escalating weak evidence and wasting analyst time. We
wanted to see how different kinds of AI agents handle that decision when they are all given the same alert.

## Solution

We built a defensive triage assistant on synthetic data. For one selected scenario it runs six agents and returns
the same result structure from each: priority, decision, reasoning, evidence, uncertainty, likely
interpretation, recommended investigation or escalation steps, and an audit note. A Streamlit page shows the
scenario, a six-agent comparison table, and a detail panel for each agent.

The assistant only recommends. It never blocks, disables, quarantines, changes a firewall, terminates a session
or contains anything. Consequential actions are left to a human.

## Agents

- **Simple Reflex** matches the current alert against fixed condition/action rules. No memory.
- **Model-Based Reflex** also remembers what it has seen (accounts, assets, alert types), so a repeated pattern
  can change its answer.
- **Goal-Based** has an explicit goal (safe and proportionate triage), checks four defensive options against it
  and picks the least disruptive option that satisfies the goal.
- **Utility-Based** scores each option on six weighted criteria and picks the highest expected utility.
- **Learning** prefers the action with the better smoothed success rate in its outcome history, and the
  history is updated only through explicit feedback.
- **LLM Agent** uses a local Ollama model (`qwen3:4b`) through LangChain and calls evidence tools itself.

All of them implement the same `decide(scenario)` interface and return an `AgentDecision`.

## Scenarios

| | A | B | C |
|---|---|---|---|
| Situation | repeated login failures on a privileged account from an unfamiliar source | low-severity alert, limited evidence | correlated, unusual activity on a critical service |
| Role | high-risk case | normal case | edge/ambiguous case |
| Expected result for the classical agents | HIGH / INVESTIGATE | LOW / COLLECT_MORE_EVIDENCE | CRITICAL / ESCALATE_TO_SECURITY_ANALYST |

In none of the scenarios is compromise confirmed. Scenario C is the edge case: the consequence of being wrong
is high, but the evidence is mixed (known source and account, some successful logins), so an agent has to
handle uncertainty rather than assume the worst.

All six agents receive the same `SecurityScenario`. The comparison service gives each agent an equal copy,
records a SHA-256 fingerprint of the scenario's canonical JSON, and the page shows that the fingerprint was
identical for every agent. The fingerprint is the same on Windows and Linux.

## LLM and Tools

The LLM Agent is given the scenario but no evidence. It has four read-only tools: `get_alert_details`,
`search_authentication_events`, `get_asset_criticality` and `get_response_policy`. The model decides which ones
to call; our orchestration code validates the call, runs it against the synthetic data and returns the result.

A recommendation is accepted only if:

- at least two distinct tools returned successful results that concern the scenario's own records;
- every evidence item cites a tool call that happened and source IDs that call returned;
- stated counts, asset criticality and policy facts match the tool results;
- the output is valid structured JSON (one repair attempt is allowed);
- deterministic guardrails find no claim or recommendation of blocking, disabling, quarantine, firewall changes,
  containment or offensive action, and no unsupported claim of compromise.

If Ollama is unavailable the page shows an error for the LLM Agent only, keeps the five classical results, and
offers a retry. A failed live run is never replaced by a scripted result.

## Learning Demonstration

The Learning Agent keeps success and failure counts per pattern and action, and uses a Beta(1,1)-smoothed rate
`(S + 1) / (S + F + 2)`. For Scenario A the seed history is INVESTIGATE 8 successes / 2 failures (0.7500) and
ESCALATE_TO_SECURITY_ANALYST 5 / 3 (0.6000), so it recommends INVESTIGATE.

Recording explicit FAILURE feedback against INVESTIGATE lowers its rate: after three failures the two actions tie
at 0.6000 and the less disruptive one is kept, and after a fourth failure (0.5625) the decision changes to
ESCALATE_TO_SECURITY_ANALYST. The scenario never changed; only the simulated history did. The page runs this on
a temporary copy, so the committed experience file stays untouched (checked by hash in the tests).

## Streamlit Comparison

The sidebar selects the scenario and runs all agents. The main area shows the scenario facts, a table of agent,
mechanism, priority, decision and runtime, and one expander per agent with the required output fields followed by
mechanism-specific details: matched rule, internal state, goal and plan, utility ranking, experience history and
the LLM tool trace. Two small demonstrations show Model-Based memory (HIGH / INVESTIGATE then CRITICAL /
ESCALATE_TO_SECURITY_ANALYST for the same scenario) and the Learning feedback effect.

## Testing

The offline suite has 1379 tests, all passing. It covers each classical agent, the tools, the two-tool rule,
grounding, guardrails, the comparison service, the output contract for all six agents, the Streamlit page and
cross-platform behavior. In those tests the language model is replaced by a scripted stand-in, so they run
without Ollama.

Live checks against a local Ollama service are in place (connection check, tool-calling check, 24 integration
tests and a script that runs scenarios A, B and C). Pending live Ollama validation: their results are recorded in
`docs/test_results.md` only after they are actually run.

## Safety and Limitations

The application is recommendation-only. Policies state that automated disruptive action is not permitted and
that consequential actions need human authorization. Data is synthetic and no real systems are connected.

Limitations: utility weights and outcome probabilities are our own assumptions; learning history is simulated
and the JSON store is meant for a single-process demonstration; the quality of the LLM Agent depends on the
local model and hardware, and CPU inference can take minutes; and the LLM can produce an invalid answer, which
the grounding and guardrail checks reject rather than repair silently.

## How the Rubric is Addressed

- **Five classical agents:** five separate implementations in `agents/`, each with its own mechanism.
- **LLM agent and tool calling:** `LLMAgent` with LangChain and a local Ollama model, four evidence tools, the
  two-distinct-tools rule, source grounding and guardrails.
- **Shared scenarios and testing:** scenarios A, B and C, one shared `SecurityScenario`, fingerprint check,
  automated tests and a manual demo flow.
- **Streamlit UI and comparison:** `app.py` shows the scenario, six-agent comparison and detail panels.
- **Learning and evaluation:** outcome history, explicit feedback, before/after demonstration with a changed
  decision, grounding of LLM evidence, and stated limitations.
- **Documentation and demonstration:** README, setup guide, screenshots section, test results, this summary and
  a demo guide.
