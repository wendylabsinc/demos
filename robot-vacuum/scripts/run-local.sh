#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen
exec uv run --frozen robot-vacuum serve "$@"
