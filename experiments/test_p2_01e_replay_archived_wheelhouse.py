"""P2-01E: offline synthetic wheel metadata/replay adversarial unit tests, no PCC."""
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from experiments import p2_01e_replay_archived_wheelhouse as candidate


class CandidateWheelReplayTests(unittest.TestCase):
    def test_exact_13_version_pins_and_duplicate_fail_closed(self):
        pins = candidate.pins_from_text(candidate.PINS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(len(pins), 13)
        self.assertEqual(pins["cython"], "3.3.1")
        self.assertEqual(pins["numpy"], "2.5.4")
        with self.assertRaises(candidate.ReplayBlocked):
            candidate.pins_from_text("numpy>=2\n")
        with self.assertRaises(candidate.ReplayBlocked):
            candidate.pins_from_text(
                candidate.PINS_FILE.read_text(encoding="utf-8") + "\nnumpy==2.5.4\n")

    def test_reject_modified_archived_wheel_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"fake_dist-1.2.3-py3-none-any.whl"
            with ZipFile(p, "w") as z:
                z.writestr("fake_dist-1.2.3.dist-info/METADATA",
                           "Metadata-Version: 2.1\nName: fake-dist\nVersion: 1.2.3\n")
            valid = {"filename": p.name, "sha256": candidate.file_sha256(p),
                     "bytes": p.stat().st_size}
            msg = candidate.inspect_wheel(p, valid)
            self.assertEqual(msg["Version"], "1.2.3")
            with self.assertRaises(candidate.ReplayBlocked):
                candidate.inspect_wheel(p, {**valid, "sha256": "a"*64})
            with self.assertRaises(candidate.ReplayBlocked):
                candidate.inspect_wheel(p, {**valid, "bytes": 1})
            with self.assertRaises(candidate.ReplayBlocked):
                candidate.inspect_wheel(p, {**valid, "filename": "unexpected.whl"})

    def test_wrong_or_repackaged_embedded_metadata_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"fake_dist-1.2.3-py3-none-any.whl"
            with ZipFile(p, "w") as z:
                z.writestr("../leaked-data", b"must reject archive path")
                z.writestr("fake_dist-1.2.3.dist-info/METADATA",
                           "Metadata-Version: 2.1\nName: fake-dist\nVersion: 1.2.3\n")
            entry = {"filename": p.name, "sha256": candidate.file_sha256(p),
                     "bytes": p.stat().st_size}
            with self.assertRaises(candidate.ReplayBlocked):
                candidate.inspect_wheel(p, entry)

    def test_no_retrospective_old_wheel_or_P2_authorization(self):
        system = candidate.runtime_patch_verified()
        baseline = {
            "schema": "pypcc-P2-01D-original-native-forensics-new-wheel-candidates/v1",
            "status":
            "ARCHIVED_NATIVE_LINKAGE_OBSERVED_NEW_PLATFORM_WHEELS_HASHED_NOT_FINAL_P2_LOCK",
            "original_P2_01C_artifact_id": candidate.ARTIFACT_IDS[system],
            "historical_compiler_log_facts": {
                "original_P2_01C_run_id": 43434870932,
                "original_wheel_SHA256_verified": False,
            },
            "candidate_wheels_downloaded_now_NOT_original_P2_01C_install_proof": True,
            "original_dependency_wheel_bytes_sealed": False,
            "P2_01_entire_gate_pass": False,
            "P2_execution_preflight_sha256": None,
            "P2_execution_authorized": False,
            "real_PCC_fits_executed": 0,
            "runtime": {"os": system, "python": candidate.PATCH_VERSION[system]},
        }
        candidate.trusted_archive_record(baseline, system)
        for field, value in (
            ("candidate_wheels_downloaded_now_NOT_original_P2_01C_install_proof", False),
            ("original_dependency_wheel_bytes_sealed", True),
            ("P2_01_entire_gate_pass", True),
            ("P2_execution_authorized", True),
            ("real_PCC_fits_executed", 1),
        ):
            with self.subTest(field=field):
                changed = dict(baseline)
                changed[field] = value
                with self.assertRaises(candidate.ReplayBlocked):
                    candidate.trusted_archive_record(changed, system)

    def test_archived_platform_id_and_python_patch_are_pinned(self):
        system = candidate.runtime_patch_verified()
        self.assertEqual(candidate.PATCH_VERSION[system],
                         candidate.platform.python_version())
        self.assertIn(system, ("Linux", "Windows"))
        self.assertEqual(len(candidate.ARCHIVE_SHA256[system]), 64)
        self.assertEqual(candidate.OLD_NATIVE_RUN, 43434870932)


if __name__ == "__main__":
    unittest.main()
