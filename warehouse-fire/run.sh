#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  "${PYTHON:-python3.12}" -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python demo.py "$@"
