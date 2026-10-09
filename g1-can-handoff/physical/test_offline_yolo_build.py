import copy
import hashlib
import json
import io
from pathlib import Path
import tempfile
import tarfile
import unittest
import zipfile

import prepare_offline_yolo_build as build


class OfflineBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = json.loads(
            Path(build.__file__).with_name("offline_yolo_dependencies.json").read_text()
        )
        self.source = Path(build.__file__).parent / "yolo"

    def artifact(self):
        path = self.root / "known.whl"
        path.write_bytes(b"known-good")
        return path, {
            "filename": path.name,
            "size": 10,
            "sha256": hashlib.sha256(b"known-good").hexdigest(),
            "url": "https://files.pythonhosted.org/known.whl",
        }

    def test_cached_artifact_reuses_bytes_without_network(self):
        path, row = self.artifact()
        self.assertEqual(build.fetch_artifact(row, self.root, offline=True), path)

    def test_corrupted_cache_rejected_before_unpack_or_build(self):
        path, row = self.artifact()
        path.write_bytes(b"bad-change")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            build.fetch_artifact(row, self.root, offline=True)

    def test_missing_cache_does_not_try_vm_package_network(self):
        _, row = self.artifact()
        with self.assertRaisesRegex(ValueError, "Missing locked artifact"):
            build.fetch_artifact(row, self.root / "empty", offline=True)

    def test_unapproved_source_rejected(self):
        _, row = self.artifact()
        row["url"] = "http://ports.ubuntu.com/not-secure.deb"
        with self.assertRaisesRegex(ValueError, "Unexpected artifact source"):
            build.fetch_artifact(row, self.root)

    def test_source_pins_match_real_saved_configuration(self):
        stage = build.validate_source(self.source, self.lock)
        self.assertEqual(stage["stages"][0]["from"], build.BASE)

    def test_source_version_change_requires_new_lock(self):
        lock = copy.deepcopy(self.lock)
        lock["wheels"]["ultralytics"]["version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "differs from source pin"):
            build.validate_source(self.source, lock)

    def test_no_cpu_torch_or_cuda_replacement_allowed(self):
        lock = copy.deepcopy(self.lock)
        lock["wheels"]["torch"] = {}
        with self.assertRaisesRegex(ValueError, "preserve vendor CUDA torch"):
            build.validate_source(self.source, lock)

    def test_wheel_unpack_promotes_exact_arm64_bytes_and_metadata(self):
        wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as out:
            out.writestr("example/module.so", b"aarch64-fixture")
            out.writestr(
                "example-1.0.dist-info/METADATA", "Name: example\nVersion: 1.0\n"
            )
            out.writestr("example-1.0.data/data/share/example.txt", b"data")
        overlay = self.root / "overlay"
        build.unpack_wheel(wheel, overlay)
        library = overlay / "opt/venv/lib/python3.12/site-packages"
        self.assertEqual(
            (library / "example/module.so").read_bytes(), b"aarch64-fixture"
        )
        self.assertTrue((library / "example-1.0.dist-info/METADATA").is_file())
        self.assertEqual((overlay / "opt/venv/share/example.txt").read_bytes(), b"data")

    def test_archive_escape_rejected(self):
        wheel = self.root / "bad.whl"
        with zipfile.ZipFile(wheel, "w") as out:
            out.writestr("../outside", b"bad")
        with self.assertRaisesRegex(ValueError, "escapes root"):
            build.unpack_wheel(wheel, self.root / "overlay")

    def test_wheel_metadata_version_must_match_lock(self):
        wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as out:
            out.writestr(
                "example-1.0.dist-info/METADATA", "Name: example\nVersion: 1.0\n"
            )
        with self.assertRaisesRegex(ValueError, "metadata version differs"):
            build.unpack_wheel(wheel, self.root / "overlay", {"version": "2.0"})

    def test_ubuntu_package_version_checked_without_running_maintainer_scripts(self):
        (self.root / "debian-binary").write_text("2.0\n")
        for archive_name, member, content in (
            (
                "control.tar.gz",
                "./control",
                b"Package: example\nVersion: 1.0\nArchitecture: arm64\n",
            ),
            ("data.tar.gz", "./usr/lib/example.so", b"arm64-library"),
        ):
            with tarfile.open(self.root / archive_name, "w:gz") as out:
                item = tarfile.TarInfo(member)
                item.size = len(content)
                out.addfile(item, io.BytesIO(content))
        deb = self.root / "example.deb"
        # Write the standard ar container directly: Apple's ar writer skips
        # non-object input files, while Debian's ar format holds arbitrary data.
        with deb.open("wb") as out:
            out.write(b"!<arch>\n")
            for name in ("debian-binary", "control.tar.gz", "data.tar.gz"):
                data = (self.root / name).read_bytes()
                header = (
                    f"{name:<16}{0:<12}{0:<6}{0:<6}" f'{"100644":<8}{len(data):<10}`\n'
                ).encode()
                self.assertEqual(len(header), 60)
                out.write(header + data + (b"\n" if len(data) % 2 else b""))
        overlay = self.root / "overlay"
        overlay.mkdir()
        scratch = self.root / "scratch"
        scratch.mkdir()
        with self.assertRaisesRegex(ValueError, "Ubuntu package version differs"):
            build.unpack_deb(deb, overlay, scratch, {"version": "2.0"})
        build.unpack_deb(deb, overlay, scratch, {"version": "1.0"})
        self.assertEqual(
            (overlay / "usr/lib/example.so").read_bytes(), b"arm64-library"
        )

    def test_original_rpc_failure_has_distinct_native_workaround(self):
        evidence = self.source.parent / "offline/validation/exposure-500ms-20261008"
        failure = (evidence / "yolo-transfer-failure.txt").read_text()
        self.assertIn("7481342 vs. 4194304", failure)
        # This fixture locks the original symptom, while the native build-only
        # proof checks the positive alternative. Never invoke failed RPC again.
        self.assertNotIn("torch", self.lock["wheels"])
        self.assertEqual(self.lock["wheels"]["opencv-python"]["version"], "4.11.0.86")


if __name__ == "__main__":
    unittest.main()
