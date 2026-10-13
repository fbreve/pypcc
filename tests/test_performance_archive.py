"""Archive integrity checks use tiny synthetic files, without PCC execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.archive_performance_results import build_archive, verify_archive


class ArchiveTests(unittest.TestCase):
    def test_preserves_bytes_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "original"
            root.mkdir()
            data = b'{"propagation_s":999999,"outlier":true}\r\n'
            (root / "measurements.jsonl").write_bytes(data)
            (root / "artifact_sha256.json").write_text(json.dumps({
                "measurements.jsonl": hashlib.sha256(data).hexdigest()}), encoding="utf-8")
            output = Path(temporary) / "results.zip"
            self.assertEqual(build_archive(output, [root])["status"], "verified")
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read("results/original/measurements.jsonl"), data)
            (root / "measurements.jsonl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                build_archive(Path(temporary) / "changed.zip", [root])
            self.assertFalse((Path(temporary) / "changed.zip").exists())

    def test_rejects_unlisted_and_corrupted_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.zip"
            for payload, hashes, message in [
                ({"data": b"changed"}, {"data": "0" * 64}, "Checksum mismatch"),
                ({"extra": b"extra"}, {}, "inventory differs"),
                ({"../escape": b"bad"}, {}, "Unsafe archive path"),
            ]:
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("checksums.json", json.dumps(hashes))
                    for name, data in payload.items():
                        archive.writestr(name, data)
                with self.assertRaisesRegex(ValueError, message):
                    verify_archive(path)


if __name__ == "__main__":
    unittest.main()
