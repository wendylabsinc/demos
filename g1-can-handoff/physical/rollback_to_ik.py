#!/usr/bin/env python3
"""Redeploy the immutable first-working IK checkout after operator release."""
from pathlib import Path
import subprocess
import sys

workspace = Path(__file__).resolve().parents[3]
source = workspace / 'demos-g1-ik-rollback/g1-can-handoff/physical'
expected = '69e62188a2f4684701d2c38076f66fb13f7721496d3117f6c0b2b373de7988a3'
check = 'from deployment_version import source_version; assert source_version()["source_digest"] == ' + repr(expected)
subprocess.run([sys.executable, '-c', check], cwd=source, check=True)
subprocess.run([sys.executable, str(source / 'g1_can_handoff/redeploy.py')], cwd=source, check=True)
