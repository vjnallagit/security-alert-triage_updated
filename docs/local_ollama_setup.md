# Local Ollama setup and live testing

The LLM Agent uses a local Ollama model. Everything else in the project runs without Ollama. This guide
covers installing the Python requirements, preparing the model, and running the live checks.

Ollama must be available locally when live LLM integration is enabled. The commands below are for a Linux or
macOS shell; on Windows, activate the virtual environment with `.venv\Scripts\Activate.ps1` and set variables
with `$env:NAME="value"`.

## 1. Python environment

```bash
cd security-alert-triage
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Offline checks (no Ollama needed)

```bash
unset RUN_OLLAMA_INTEGRATION
export LLM_RUNTIME_MODE="MOCK"
export ENABLE_MOCK_LLM="true"
export SUBMISSION_MODE="false"

python -m pytest -m "not integration" -q
python scripts/validate_llm_offline.py
python scripts/validate_comparison.py
python scripts/validate_submission_readiness.py
python scripts/validate_final_submission.py
```

The offline suite replaces the language model with a scripted stand-in, so it tests the orchestration, grounding,
guardrails and UI without Ollama. It does not show that a real model can call tools reliably; the live checks
below do.

## 3. Install Ollama and the model

Install Ollama from its official distribution, then check it:

```bash
ollama --version
ollama list
ollama pull qwen3:4b
```

`qwen3:4b` is the configured submission model. Ollama listens on `127.0.0.1:11434`; keep it bound to localhost
and do not expose the port. No API key is needed.

Warm the model once so the first request is not slower than the rest:

```bash
ollama run qwen3:4b "Respond with exactly: OLLAMA_WARM"
ollama ps
```

## 4. Live configuration

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="qwen3:4b"
export OLLAMA_TEMPERATURE="0"
export OLLAMA_REQUEST_TIMEOUT_SECONDS="300"
export OLLAMA_MAX_TOOL_ITERATIONS="8"
export OLLAMA_MINIMUM_TOOL_CALLS="2"

export LLM_RUNTIME_MODE="LIVE"
export ENABLE_MOCK_LLM="false"
export RUN_OLLAMA_INTEGRATION="true"
export SUBMISSION_MODE="false"     # set to "true" only after the live checks below pass

env | grep -E 'OLLAMA_|LLM_RUNTIME|ENABLE_MOCK|RUN_OLLAMA|SUBMISSION'
```

CPU inference can take several minutes for one scenario, which is why the request timeout is 300 seconds. The
application sends one live request at a time.

## 5. Live checks

```bash
python scripts/validate_ollama_connection.py
python scripts/validate_ollama_tool_calling.py
python -m pytest -m integration -v
python scripts/run_llm_scenarios.py
```

Do not run the integration tests and the Streamlit app at the same time; both would send requests to the same
model. If Ollama is not reachable, each script prints the configured URL and the next checks to make
(`ollama --version`, `ollama list`, restart Ollama). Add `--debug` to see technical detail.

## 6. Run the Streamlit app

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. With `SUBMISSION_MODE=true` the app runs the live Ollama model only. If Ollama
is unavailable the five classical agents still run and the LLM Agent shows an error with a **Retry LLM Agent**
button.

## 7. Record the results

Copy the actual output of step 5 into [test_results.md](test_results.md) and take the screenshots listed in the
README. Only record results that were actually produced by a live run.
