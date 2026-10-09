"""Prepare a hash-locked, network-free Stagefile context; never deploy or move.

The host fetches artifacts over HTTPS, verifies the committed lock, and unpacks
them into a root overlay. Native Apple Container builds only COPY operations.
CUDA torch/torchvision remain supplied by the original digest-pinned base.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile

BASE = "dustynv/pytorch:2.7-r36.4.0-cu128-24.04"
DIGEST = "sha256:4bb7f0a3852f86629ed271aaf6c4aa000305cc4e4673b8e9f334668ee1b22bc8"
SOURCE_FILES = (
    "service.py",
    "preprocessing.py",
    "postprocessing.py",
    "qualification.py",
    "frame_limits.py",
    "confidence.py",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_artifact(path, row):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Missing locked artifact: " + str(path))
    if path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
        raise ValueError("Locked artifact checksum mismatch: " + str(path))


def fetch_artifact(row, cache, *, offline=False):
    parsed = urllib.parse.urlsplit(row["url"])
    if parsed.scheme != "https" or parsed.hostname not in (
        "files.pythonhosted.org",
        "ports.ubuntu.com",
    ):
        raise ValueError("Unexpected artifact source: " + row["url"])
    name = row["filename"]
    if Path(name).name != name:
        raise ValueError("Artifact filename must be a basename")
    path = Path(cache) / name
    if path.exists():
        verify_artifact(path, row)
        return path
    if offline:
        raise ValueError("Missing locked artifact in offline cache: " + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(row["url"], timeout=30) as response:
            # Redirects must remain on an approved HTTPS artifact host.
            final = urllib.parse.urlsplit(response.url)
            if final.scheme != "https" or final.hostname != parsed.hostname:
                raise ValueError("Unexpected artifact redirect")
            with temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
        verify_artifact(temporary, row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Archive member escapes root: " + name)
    return path


def unpack_wheel(path, root, expected=None):
    library = Path(root) / "opt/venv/lib/python3.12/site-packages"
    with zipfile.ZipFile(path) as archive:
        if expected is not None:
            metadata_files = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ValueError("Wheel has no unique distribution metadata")
            metadata = archive.read(metadata_files[0]).decode()
            version = next(
                (
                    line[9:]
                    for line in metadata.splitlines()
                    if line.startswith("Version: ")
                ),
                None,
            )
            if version != expected["version"]:
                raise ValueError("Wheel metadata version differs from lock")
        for entry in archive.infolist():
            member = safe_member(entry.filename)
            parts = member.parts
            if not parts or entry.is_dir():
                continue
            if parts[0].endswith(".data"):
                scheme, *relative = parts[1:]
                destinations = {
                    "purelib": library,
                    "platlib": library,
                    "data": Path(root) / "opt/venv",
                    "headers": Path(root) / "opt/venv/include",
                    "scripts": Path(root) / "opt/venv/bin",
                }
                if scheme not in destinations:
                    raise ValueError("Unsupported wheel installation scheme: " + scheme)
                target = destinations[scheme].joinpath(*relative)
            else:
                target = library.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
            mode = (entry.external_attr >> 16) & 0o777
            target.chmod(mode or 0o644)


def unpack_deb(path, root, scratch, expected=None):
    members = [
        name.rstrip("/")
        for name in subprocess.check_output(
            ["ar", "t", str(path)], text=True
        ).splitlines()
    ]
    data = [name for name in members if name.startswith("data.tar")]
    if len(data) != 1:
        raise ValueError("Debian artifact has no unique data archive")
    if expected is not None:
        control_names = [name for name in members if name.startswith("control.tar")]
        if len(control_names) != 1:
            raise ValueError("Debian artifact has no unique control archive")
        control = Path(scratch) / control_names[0]
        with control.open("wb") as out:
            subprocess.run(
                ["ar", "p", str(path), control_names[0]], stdout=out, check=True
            )
        metadata = subprocess.check_output(
            ["tar", "-xOf", str(control), "./control"], text=True
        )
        version = next(
            (
                line[9:]
                for line in metadata.splitlines()
                if line.startswith("Version: ")
            ),
            None,
        )
        if version != expected["version"]:
            raise ValueError("Ubuntu package version differs from lock")
        control.unlink()
    tar = Path(scratch) / data[0]
    with tar.open("wb") as out:
        subprocess.run(["ar", "p", str(path), data[0]], stdout=out, check=True)
    # bsdtar supports Ubuntu's zstd data archives. No maintainer script executes.
    names = subprocess.check_output(["tar", "-tf", str(tar)], text=True).splitlines()
    for name in names:
        safe_member(name)
    subprocess.run(["tar", "-xf", str(tar), "-C", str(root)], check=True)
    tar.unlink()


def validate_source(source, lock):
    import yaml

    stage = yaml.safe_load((Path(source) / "build.stagefile.yaml").read_text())
    original = yaml.safe_load((Path(source) / "build.stagefile.lock.yaml").read_text())
    if stage["stages"][0]["from"] != BASE or original["images"].get(BASE) != DIGEST:
        raise ValueError("Offline bundle does not match the pinned CUDA base")
    packages = stage["stages"][0]["install"]["pip"][0]["packages"]
    for pin in packages:
        name, version = pin.split("==", 1)
        if lock["wheels"].get(name, {}).get("version") != version:
            raise ValueError("Offline dependency lock differs from source pin: " + pin)
    if any(
        name.startswith(("torch", "nvidia-cuda", "nvidia-cudnn"))
        for name in lock["wheels"]
    ):
        raise ValueError("Offline overlay must preserve vendor CUDA torch")
    descriptor = json.loads((Path(source) / "wendy.json").read_text())
    if descriptor.get("platform", "linux/arm64") not in ("linux", "linux/arm64"):
        raise ValueError("Offline bundle is restricted to the original arm64 platform")
    return stage


def prepare(source, context, lock, cache, *, offline=False):
    source, context, cache = map(Path, (source, context, cache))
    stage = validate_source(source, lock)
    if (
        context.resolve() == source.resolve()
        or source.resolve() in context.resolve().parents
    ):
        raise ValueError("Generated context must be separate from source")
    # Validate the whole bundle before writing an install overlay. No partial
    # overlay can be mistaken for a complete successful preparation.
    artifacts = {
        kind: [
            (row, fetch_artifact(row, cache, offline=offline))
            for row in lock[kind].values()
        ]
        for kind in ("apt", "wheels")
    }
    context.mkdir(parents=True, exist_ok=True)
    root = context / "root"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    scratch = context / ".scratch"
    scratch.mkdir(exist_ok=True)
    for row, artifact in artifacts["apt"]:
        unpack_deb(artifact, root, scratch, row)
    for row, artifact in artifacts["wheels"]:
        unpack_wheel(artifact, root, row)
    # Do not promote apt's compiler/cache/host configuration into the runtime.
    for rel in ("etc", "var", "usr/share/doc", "usr/share/man"):
        path = root / rel
        if path.exists():
            shutil.rmtree(path)
    for name in SOURCE_FILES + ("wendy.json",):
        shutil.copy2(source / name, context / name)
    for name in ("fixtures", "model"):
        shutil.copytree(source / name, context / name, dirs_exist_ok=True)
    # The installed compiler appends the lock digest; spelling @sha256 in
    # `from` would append it twice. Reuse the original verified image lock.
    dependency_stage = {
        "name": "dependencies",
        "from": BASE,
        "workdir": "/app",
        "env": stage["stages"][0]["env"],
        "copy": [{"from": "local", "paths": ["root/"], "dest": "/"}],
    }
    stages = [dependency_stage] + stage["stages"][1:]
    import yaml

    (context / "build.stagefile.yaml").write_text(
        yaml.safe_dump({"version": 1, "stages": stages}, sort_keys=False)
    )
    shutil.copy2(
        source / "build.stagefile.lock.yaml", context / "build.stagefile.lock.yaml"
    )
    manifest = {
        "base": BASE + "@" + DIGEST,
        "source_sha256": {name: sha256(source / name) for name in SOURCE_FILES},
        "artifacts": {
            row["filename"]: row["sha256"]
            for kind in ("apt", "wheels")
            for row in lock[kind].values()
        },
        "runtime_network_configuration_added": False,
        "motion_commands": 0,
    }
    (context / "offline-build-provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "yolo")
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).with_name("offline_yolo_dependencies.json"),
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare(
                args.source,
                args.context,
                json.loads(args.lock.read_text()),
                args.cache,
                offline=args.offline,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
