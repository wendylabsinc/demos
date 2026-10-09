#!/usr/bin/env python3
"""Launch the checked physical NX2 redeployment from the demo's root folder."""

import os
from pathlib import Path
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    script = root / "physical/g1_can_handoff/redeploy.py"
    venv = root.parents[1] / "demos-g1-nx2-yolo/.venv/bin/python"
    python = str(venv if venv.is_file() else Path(sys.executable))
    os.execv(python, [python, str(script), *sys.argv[1:]])
