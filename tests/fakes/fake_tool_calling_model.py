"""Test-facing import path for the scripted model. The single implementation lives in
devtools/mock_llm_runtime.py (development/test only); it is re-exported here so unit tests keep one stable import."""

from devtools.mock_llm_runtime import *  # noqa: F401,F403
from devtools.mock_llm_runtime import __all__  # noqa: F401
