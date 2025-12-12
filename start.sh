#! /bin/bash

DEBUG_MODE=1 CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK=1 poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
