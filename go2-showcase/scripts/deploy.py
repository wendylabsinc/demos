#!/usr/bin/env python3
"""Stage the selected profile without changing the project's default config."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=["sim", "hardware", "ros-sim"], default="sim")
parser.add_argument("--device", required=True)
parser.add_argument("--domain", type=int, default=0)
parser.add_argument("--env", action="append", default=[])
args = parser.parse_args()
if not 0 <= args.domain <= 232:
    parser.error("ROS domain must be between 0 and 232")
with tempfile.TemporaryDirectory(prefix="go2-showcase-deploy-") as folder:
    stage = Path(folder)
    for name in ("showcase", "sim"):
        shutil.copytree(root / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("requirements.txt", "requirements-hardware.txt"):
        shutil.copy2(root / name, stage / name)
    config = json.loads((root / ("wendy.json" if args.profile == "sim" else "wendy.hardware.json")).read_text())
    config["env"]["GO2_PROFILE"] = args.profile
    if "ros2" in config.get("frameworks", {}):
        config["frameworks"]["ros2"]["domainId"] = args.domain
    for setting in args.env:
        if "=" not in setting:
            parser.error("Each --env must be KEY=VALUE")
        key, value = setting.split("=", 1)
        config["env"][key] = value
    (stage / "wendy.json").write_text(json.dumps(config, indent=2))
    shutil.copy2(root / ("Dockerfile.sim" if args.profile == "sim" else "Dockerfile"), stage / "Dockerfile")
    subprocess.run(["wendy", "--json", "--device", args.device, "run", "--prefix", str(stage),
                    "--builder", "docker", "--detach", "--yes"], check=True)
