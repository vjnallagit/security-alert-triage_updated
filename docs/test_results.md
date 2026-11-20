# Test Results

## Offline Tests

| Command | Result |
|---|---|
| `python -m pytest -m "not integration" -q` | 1379 passed, 0 failed |
| `python scripts/validate_phase1.py` to `validate_phase6.py` | all successful |
| `python scripts/validate_llm_offline.py` | successful |
| `python scripts/validate_comparison.py` | successful |
| `python scripts/validate_submission_readiness.py` | successful |
| `python scripts/validate_final_submission.py` | successful |

The offline tests replace the language model with a scripted stand-in. They check the orchestration,
grounding, guardrails, comparison and Streamlit page, but they do not show that a real model calls tools
reliably.

## Local Ollama Integration

| Command | Result |
|---|---|
| `python scripts/validate_ollama_connection.py` | Pending live run |
| `python scripts/validate_ollama_tool_calling.py` | Pending live run |
| `python -m pytest -m integration -v` (24 tests) | Pending live run |
| `python scripts/run_llm_scenarios.py` | Pending live run |

Pending live Ollama validation. Fill this table in only after the commands have been run against a local
Ollama service with `qwen3:4b`, using the actual counts and output.

## Scenario Results

| Scenario | LLM Priority | LLM Decision | Tools Used | Grounding Result | Safety Result |
|---|---|---|---|---|---|
| A | Pending live run | Pending live run | Pending live run | Pending live run | Pending live run |
| B | Pending live run | Pending live run | Pending live run | Pending live run | Pending live run |
| C | Pending live run | Pending live run | Pending live run | Pending live run | Pending live run |

## Notes

A direct request to the Ollama API (outside this project) showed that `qwen3:4b` can emit a tool call
(`get_asset_criticality` with `asset_id = ASSET-PAY-01`). That is a quick capability check only; it is not a
result from this project's orchestration, grounding or guardrails.
