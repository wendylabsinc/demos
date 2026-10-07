#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
fi
exec .venv/bin/python demo.py "$@"
