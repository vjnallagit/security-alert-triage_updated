"""DEVELOPMENT / TEST ONLY tools. Nothing in the production runtime may import this package except
services/llm_runtime.py, and only when the explicit mock gate (ENABLE_MOCK_LLM=true, not SUBMISSION_MODE) passes.
This package can be omitted from a submission copy without affecting the live runtime."""
