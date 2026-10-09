"""Identify the exact controller source built from this checkout.

The descriptor version is a human label. The source digest also distinguishes
uncommitted edits with the same label. Runtime reads the immutable build stamp;
it never asks Git or hashes source files on the control thread.
"""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
STAMP = "deployment-version.json"


def source_hashes(root=ROOT):
    """Hash the copied Python sources, planner profile and launch configuration."""
    root = Path(root)
    copied = {
        name.strip().strip("\"'")
        for group in re.findall(
            r"paths:\s*\[([^\]]+)\]", (root / "build.stagefile.yaml").read_text()
        )
        for name in group.split(",")
        if name.strip()
    }
    paths = [p for p in root.glob("*.py") if p.name in copied]
    for directory in ("adapter", "g1_can_handoff"):
        paths.extend(
            p for p in (root / directory).rglob("*.py") if "__pycache__" not in p.parts
        )
    paths.extend(
        root / name
        for name in (
            "yolo/frame_limits.py",
            "yolo/confidence.py",
            "g1_can_handoff/noise_aware_planner.json",
            "wendy.json",
            "build.stagefile.yaml",
        )
    )
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
    }


def source_version(root=ROOT):
    root = Path(root)
    descriptor = json.loads((root / "wendy.json").read_text())
    files = source_hashes(root)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no", "--", "."],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "app_id": descriptor["appId"],
        "app_version": descriptor["version"],
        "git_commit": commit,
        "dirty": dirty,
        "source_digest": digest,
    }


def write_stamp(info, root=ROOT):
    stamp = {**info, "built_at_utc": datetime.now(timezone.utc).isoformat()}
    path = Path(root) / STAMP
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stamp, indent=2) + "\n")
    temporary.replace(path)
    return stamp


@lru_cache(maxsize=1)
def deployed_version():
    path = ROOT / STAMP
    if not path.exists():
        return {
            "app_version": None,
            "git_commit": None,
            "source_digest": None,
            "built_at_utc": None,
        }
    return json.loads(path.read_text())


def describe_version(info):
    commit = (info.get("git_commit") or "unknown")[:12]
    suffix = " + local edits" if info.get("dirty") else ""
    digest = (info.get("source_digest") or "unknown")[:12]
    return f"{info.get('app_version') or 'unreported'} | commit {commit}{suffix} | source {digest}"
