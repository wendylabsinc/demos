"""Resolve checkout assets and installed wheel assets."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def asset_dir(name: str) -> Path:
    bundled = PACKAGE / name
    if bundled.is_dir():
        return bundled
    checkout = ROOT / name
    return checkout
