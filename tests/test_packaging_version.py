"""Check installed and built distribution metadata against the version source.

Set PCC_DIST_DIR to the directory containing freshly built wheel/sdist artifacts
to enable archive checks (required in CI).
"""
from email.parser import BytesParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class PackagingVersionTests(unittest.TestCase):
    def setUp(self):
        self.version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.license_bytes = (ROOT / "LICENSE").read_bytes()

    def assert_metadata(self, contents):
        message = BytesParser().parsebytes(contents)
        self.assertEqual(message["Name"].replace("-", "_"), "pcc_step")
        self.assertEqual(message["Version"], self.version)
        # Modern setuptools uses PEP 639 License-Expression; older
        # setuptools may still emit the legacy License metadata field.
        self.assertEqual(
            message.get("License-Expression") or message.get("License"),
            "MIT",
        )

    def assert_license(self, contents):
        self.assertEqual(contents, self.license_bytes)
        self.assertIn(b"MIT License", contents)
        self.assertIn(b"Copyright (c) 2026 Fabricio Aparecido Breve", contents)
        self.assertIn(b"Permission is hereby granted", contents)

    def artifacts(self, pattern):
        directory = os.environ.get("PCC_DIST_DIR")
        if directory is None:
            self.skipTest("Set PCC_DIST_DIR to check built archives")
        paths = sorted(Path(directory).glob(pattern))
        self.assertTrue(paths, f"No {pattern} artifacts in {directory}")
        return paths

    def test_installed_metadata(self):
        # Isolated mode excludes checkout egg-info, which can mask an old install.
        output = subprocess.check_output([
            sys.executable, "-I", "-c",
            "import json; from importlib import metadata; "
            "d = metadata.distribution('pcc_step'); "
            "print(json.dumps([d.metadata['Name'], d.version, "
            "d.metadata.get('License-Expression') or d.metadata.get('License')]))",
        ], text=True)
        name, version, license_id = json.loads(output)
        self.assertEqual(name.replace("-", "_"), "pcc_step")
        self.assertEqual(version, self.version)
        self.assertEqual(license_id, "MIT")

    def test_wheel_metadata(self):
        for path in self.artifacts("*.whl"):
            with self.subTest(artifact=path.name), zipfile.ZipFile(path) as archive:
                entries = [name for name in archive.namelist()
                           if name.endswith(".dist-info/METADATA")]
                self.assertEqual(len(entries), 1)
                self.assert_metadata(archive.read(entries[0]))
                license_entries = [
                    name for name in archive.namelist()
                    if ".dist-info/" in name and name.endswith("/LICENSE")
                ]
                self.assertEqual(
                    len(license_entries), 1,
                    f"Expected exactly one LICENSE inside {path.name}",
                )
                self.assert_license(archive.read(license_entries[0]))
                self.assertTrue(path.name.startswith(f"pcc_step-{self.version}-"))

    def test_sdist_metadata_and_version_source(self):
        for path in self.artifacts("*.tar.gz"):
            with self.subTest(artifact=path.name), tarfile.open(path) as archive:
                root = f"pcc_step-{self.version}"
                self.assertEqual(path.name, f"{root}.tar.gz")
                self.assert_metadata(archive.extractfile(f"{root}/PKG-INFO").read())
                source = archive.extractfile(f"{root}/VERSION").read().decode("utf-8")
                self.assertEqual(source.strip(), self.version)
                license_member = archive.extractfile(f"{root}/LICENSE")
                self.assertIsNotNone(license_member)
                self.assert_license(license_member.read())


if __name__ == "__main__":
    unittest.main()
