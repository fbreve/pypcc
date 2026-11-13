"""Adversarial stdlib-only checks: no PCC imports, datasets or propagation."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments import p2_01c_seal_artifacts as seal


class CandidateSealTests(unittest.TestCase):
    def test_complete_candidate_pins(self):
        pins = seal.lock_versions(seal.LOCK.read_text(encoding="utf-8"))
        self.assertEqual(set(pins), seal.REQUIRED)
        self.assertEqual(pins["numpy"], "2.5.4")

    def test_non_exact_and_duplicate_pins_rejected(self):
        with self.assertRaises(seal.SealBlocked):
            seal.lock_versions("numpy>=2.0")
        with self.assertRaises(seal.SealBlocked):
            seal.lock_versions("numpy==2.5.4\nNumPy==2.5.4")

    def test_drifted_dependency_rejected(self):
        expected = {"numpy": "2.5.4"}
        with self.assertRaises(seal.SealBlocked):
            seal.validate_versions(expected, {"numpy": "2.5.3"})

    def test_native_byte_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pcc_step.cp312-test.so"
            path.write_bytes(b"original")
            desc = {"binary_filename": path.name, "bytes": len(b"original"),
                    "sha256": hashlib.sha256(b"original").hexdigest()}
            checked = seal.verify_native_file(path, desc, suffix=".cp312-test.so")
            self.assertEqual(seal.sha256(checked), desc["sha256"])
            path.write_bytes(b"tampered")
            with self.assertRaises(seal.SealBlocked):
                seal.verify_native_file(path, desc, suffix=".cp312-test.so")

    def test_false_p2_go_rejected(self):
        fake = {"result": "NATIVE_ARTIFACTS_BUILT_AND_IMPORTED_NO_PCC_FIT",
                "P1_approved_scientific_sha256": seal.P1_SHA,
                "P2_execution_authorized": True,
                "P2_preflight_sha256": None,
                "PCC_fits_executed": 0, "real_datasets_loaded": 0,
                "label_masks_drawn": 0, "native_Cython_modules":
                dict.fromkeys(seal.MODULES),
                "generated_isolated_Cython_pyx_LF_canonical_sha256":
                "40db83e8df030229c15b3660b6bea03f0b45e250a09872bfd40ebaa69d4f5abf"}
        with self.assertRaises(seal.SealBlocked):
            seal.validate_native_report(fake)


if __name__ == "__main__":
    unittest.main()
