#!/usr/bin/env python3
"""Check publishable files without importing demo code or downloading assets.

Run with Python 3.11 or newer from any directory. Git supplies the file list,
so ignored local models, virtual environments, and generated output stay out
of these checks.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tokenize

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required.")

import tomllib


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE = 50 * 1024 * 1024
SHELLS = {"bash", "sh", "dash", "ksh", "zsh"}


def git_output(*args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def shell_for(path: Path) -> str | None:
    with path.open("rb") as source:
        first_line = source.readline(512).decode("utf-8", errors="replace")
    if first_line.startswith("#!"):
        words = shlex.split(first_line[2:].strip())
        if words:
            interpreter = Path(words[0]).name
            if interpreter == "env":
                interpreter = next(
                    (word for word in words[1:] if word in SHELLS), ""
                )
            if interpreter in SHELLS:
                return interpreter
    return "bash" if path.suffix == ".sh" else None


def reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not a valid JSON value")


def main() -> int:
    errors: list[str] = []
    checked: Counter[str] = Counter()
    try:
        names = sorted(
            {
                os.fsdecode(name)
                for name in git_output(
                    "ls-files", "--cached", "--others", "--exclude-standard", "-z"
                ).split(b"\0")
                if name
            }
        )
        for entry in git_output("ls-files", "--stage", "-z").split(b"\0"):
            if entry.startswith(b"160000 "):
                name = os.fsdecode(entry.split(b"\t", 1)[1])
                errors.append(f"{name}: nested Git repository must be vendored or removed")
    except RuntimeError as exc:
        print(f"Repository check failed: {exc}", file=sys.stderr)
        return 1

    for name in names:
        path = ROOT / name
        if path.is_dir() and (path / ".git").exists():
            errors.append(f"{name}: nested Git repository must be vendored or removed")
            continue
        if not path.is_file():
            continue
        checked["files"] += 1
        try:
            size = path.stat().st_size
            if size > MAX_FILE_SIZE:
                errors.append(
                    f"{name}: {size / (1024 * 1024):.1f} MiB exceeds the 50 MiB limit; "
                    "store downloads or generated output outside Git"
                )
            if path.suffix == ".py":
                with tokenize.open(path) as source:
                    compile(source.read(), name, "exec")
                checked["Python"] += 1
            elif path.suffix == ".json":
                with path.open(encoding="utf-8") as source:
                    json.load(source, parse_constant=reject_json_constant)
                checked["JSON"] += 1
            elif path.suffix == ".toml":
                with path.open("rb") as source:
                    tomllib.load(source)
                checked["TOML"] += 1

            shell = shell_for(path)
            if shell:
                executable = shutil.which(shell)
                if executable is None:
                    errors.append(f"{name}: {shell} is required for shell syntax checks")
                    continue
                result = subprocess.run(
                    [executable, "-n", str(path)], capture_output=True, text=True
                )
                if result.returncode:
                    errors.append(f"{name}: {result.stderr.strip()}")
                else:
                    checked["shell"] += 1
        except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")

    summary = ", ".join(f"{count} {kind}" for kind, count in checked.items())
    if errors:
        print(f"Repository check failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        print(f"Checked {summary}.", file=sys.stderr)
        return 1
    print(f"Repository checks passed: {summary}; no gitlinks or files over 50 MiB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
