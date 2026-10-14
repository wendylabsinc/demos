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
parser.add_argument("--profile", choices=["sim", "air", "hardware", "ros-sim"], default="sim")
parser.add_argument("--device", required=True)
parser.add_argument("--domain", type=int, default=0)
parser.add_argument("--env", action="append", default=[])
args = parser.parse_args()
if not 0 <= args.domain <= 232:
    parser.error("ROS domain must be between 0 and 232")
with tempfile.TemporaryDirectory(prefix="go2-showcase-deploy-") as folder:
    stage = Path(folder)
    sources = ("showcase", "sim") if args.profile in ("sim", "hardware") else ("showcase",)
    for name in sources:
        shutil.copytree(root / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    requirements = {
        "sim": ("requirements.txt",),
        "air": ("requirements-air.txt",),
        "ros-sim": ("requirements-air.txt",),
        "hardware": ("requirements.txt", "requirements-hardware.txt"),
    }
    for name in requirements[args.profile]:
        shutil.copy2(root / name, stage / name)
    config_name = "wendy.json" if args.profile == "sim" else (
        "wendy.hardware.json" if args.profile == "hardware" else "wendy.air.json")
    config = json.loads((root / config_name).read_text())
    config["env"]["GO2_PROFILE"] = args.profile
    if "ros2" in config.get("frameworks", {}):
        config["frameworks"]["ros2"]["domainId"] = args.domain
    for setting in args.env:
        if "=" not in setting:
            parser.error("Each --env must be KEY=VALUE")
        key, value = setting.split("=", 1)
        config["env"][key] = value
    (stage / "wendy.json").write_text(json.dumps(config, indent=2))
    dockerfile = {"sim": "Dockerfile.sim", "hardware": "Dockerfile",
                  "air": "Dockerfile.air", "ros-sim": "Dockerfile.air"}[args.profile]
    shutil.copy2(root / dockerfile, stage / "Dockerfile")
    subprocess.run(["wendy", "--json", "--device", args.device, "run", "--prefix", str(stage),
                    "--builder", "docker", "--detach", "--yes"], check=True)
