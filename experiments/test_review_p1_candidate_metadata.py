"""Adversarial, no-dataset P1 candidate provenance review tests."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import review_p1_candidate_metadata as audit


class CandidateMetadataReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads(audit.REGISTRY.read_text(encoding="utf-8"))
        cls.historical = audit.HISTORICAL.read_bytes()

    def test_inert_metadata_report_and_source_pin(self):
        report = audit.review(self.registry, historical_raw=self.historical)
        again = audit.review(self.registry, historical_raw=self.historical)
        self.assertEqual(report, again)
        self.assertEqual(report["metadata_review"], "PASS")
        self.assertEqual(report["candidate_ids"], [212, 545, 850])
        self.assertEqual(report["historically_documented_datasets"], 20)
        self.assertTrue(report["candidates_not_in_frozen_20_by_name"])
        self.assertFalse(report["novelty_outside_frozen_20_verified"])
        self.assertEqual(report["datasets_selected"], 0)
        self.assertFalse(report["source_bytes_verified"])
        self.assertFalse(report["source_license_independently_approved"])
        self.assertFalse(report["P1_scientific_protocol_frozen"])
        self.assertFalse(report["execution_authorized"])
        self.assertIsNone(report["P2_execution_preflight_sha256"])
        self.assertEqual(report["datasets_downloaded"], 0)
        self.assertEqual(report["model_fits_executed"], 0)
        self.assertGreaterEqual(len(report["review_blockers"]), 6)
        self.assertEqual(
            report["candidate_checksum_not_approval"],
            audit._canonical_sha256(self.registry),
        )
        self.assertRegex(report["candidate_checksum_not_approval"], r"^[0-9a-f]{64}$")

    def mutated(self, modify, historical=None):
        registry = copy.deepcopy(self.registry)
        modify(registry)
        with self.assertRaises(audit.CandidateReviewBlocked):
            audit.review(
                registry, historical_raw=self.historical if historical is None else historical,
            )

    def test_historical_table_exact_and_not_a_four_dataset_assumption(self):
        records = audit.historical_inventory(self.historical)
        self.assertEqual(len(records), 20)
        self.assertIn(("Sonar", "sonar"), records)
        self.assertIn(("PageBlocks", "pageblocks"), records)
        self.assertIn(("QSARBiodeg", "qsarbiodeg"), records)
        self.assertIn(("Iris", "iris"), records)
        self.assertIn(("Digits", "digits"), records)
        self.mutated(
            lambda r: None,
            historical=self.historical.replace(b"Wine", b"Wien", 1),
        )

    def test_fabricated_novelty_or_approval_fails_closed(self):
        self.mutated(
            lambda r: r.update(full_history_exposure_audit_complete=True)
        )
        self.mutated(
            lambda r: r["historical_registry"].update(not_exhaustive=False)
        )
        self.mutated(
            lambda r: r["run_permission"].update(execution_authorized=True)
        )
        self.mutated(
            lambda r: r["run_permission"].update(model_fits_executed=1)
        )
        self.mutated(
            lambda r: r["run_permission"].update(datasets_downloaded=1)
        )
        self.mutated(
            lambda r: r.update(selected_dataset_ids=["uci-850-raisin"])
        )
        self.mutated(
            lambda r: r.update(independent_reviewer_approved=True)
        )
        self.mutated(
            lambda r: r.update(source_archive_sha256="0" * 64)
        )

    def test_provenance_and_identity_drift_fails_closed(self):
        self.mutated(
            lambda r: r["historical_registry"].update(
                source_git_blob_sha="f" * 40,
            )
        )
        self.mutated(
            lambda r: r["historical_registry"]["entries"].pop()
        )
        self.mutated(
            lambda r: r["candidates"].__setitem__(
                0, dict(r["candidates"][0], uci_dataset_id=213),
            )
        )
        self.mutated(
            lambda r: r["candidates"][0].update(instances_reported=311)
        )
        self.mutated(
            lambda r: r["candidates"][1].update(classes_reported=3)
        )
        self.mutated(
            lambda r: r["candidates"][2].update(
                source_page="https://example.com/fake",
            )
        )
        self.mutated(
            lambda r: r["candidates"][2].update(license_reported="unknown")
        )
        self.mutated(
            lambda r: r["candidates"][2].update(
                provenance_review="approved",
            )
        )
        self.mutated(
            lambda r: r.update(p1_draft_head="0" * 40)
        )

    def test_historical_alias_cannot_be_promoted_as_new_candidate(self):
        self.mutated(
            lambda r: r["candidates"][0].update(
                name="Wine",
                uci_dataset_id=212,
            )
        )

    def test_inert_audit_imports_no_scientific_runtime_or_fetch(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import numpy", "import numba", "import Cython",
            "import sklearn", "import pcc", "fit_predict(",
            "pcc_propagate(", "requests.get(", "urllib.request",
            "urlopen(", "read_csv(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"execution_authorized": False', source)
        self.assertIn('"datasets_downloaded": 0', source)
        self.assertIn('"model_fits_executed": 0', source)


if __name__ == "__main__":
    unittest.main()
