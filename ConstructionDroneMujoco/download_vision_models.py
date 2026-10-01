"""Download immutable, checksum-verified vision weights. No runtime downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(models_root: Path) -> None:
    manifest = json.loads((ROOT / "vision-models.json").read_text())
    for model in manifest["models"]:
        for asset in model["files"]:
            target = models_root / model["directory"] / asset["local"]
            if target.is_file() and sha256(target) == asset["sha256"]:
                print(f"Verified {target.relative_to(models_root)}", flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".download")
            url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{asset['remote']}"
            print(f"Downloading {target.relative_to(models_root)}", flush=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "wendy-drone-model-downloader/1"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                actual = sha256(temporary)
                if actual != asset["sha256"]:
                    raise ValueError(f"Checksum mismatch for {target.name}: {actual}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    print("Vision models ready.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    download(parser.parse_args().models_root)
