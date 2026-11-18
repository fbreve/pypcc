"""Zero-data adversarial tests for the UNAPPROVED P1 draft gate.

This suite never imports any PCC backend, samples labels, builds a graph,
downloads a dataset or performs a propagation. Passing is NOT scientific GO.
"""
import copy
import json
from pathlib import Path
import unittest

from experiments import review_p1_proposal as p1


class P1DraftReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = json.loads(p1.PROPOSAL.read_text(encoding="utf-8"))

    def test_actual_draft_is_consistent_but_unapproved(self):
        report = p1.review(self.proposal)
        self.assertEqual(report["design_integrity"], "PASS")
        self.assertEqual(report["phase"], "P1_PROPOSAL_ONLY")
        self.assertEqual(report["prospective_fits_if_later_approved_and_feasible"], 72)
        self.assertEqual(report["datasets_selected"], 0)
        self.assertFalse(report["scientific_protocol_frozen"])
        self.assertFalse(report["execution_authorized"])
        self.assertIsNone(report["execution_preflight_sha256"])
        self.assertFalse(report["P1_ready_to_freeze"])
        self.assertFalse(report["P2_ready"])
        self.assertFalse(report["P3_ready"])
        self.assertEqual(report["model_fits_executed"], 0)
        self.assertGreaterEqual(len(report["blockers"]), 5)
        self.assertRegex(report["draft_content_sha256_not_an_approval"], r"^[0-9a-f]{64}$")

    def assert_rejected(self, modify):
        proposal = copy.deepcopy(self.proposal)
        modify(proposal)
        with self.assertRaises(p1.DesignInvalid):
            p1.review(proposal)

    def test_not_allowed_to_change_unapproved_status(self):
        self.assert_rejected(lambda p: p.update(status="APPROVED"))
        self.assert_rejected(lambda p: p["approval"].update(scientific_protocol_frozen=True))
        self.assert_rejected(lambda p: p["approval"].update(execution_authorized=True))
        self.assert_rejected(lambda p: p["approval"].update(execution_preflight_sha256="0" * 64))
        self.assert_rejected(lambda p: p["approval"].update(scientific_protocol_sha256="a" * 64))
        self.assert_rejected(lambda p: p["approval"].update(real_data_fits_authorized=1))

    def test_no_unreviewed_datasets_can_be_smuggled_into_draft(self):
        self.assert_rejected(
            lambda p: p["dataset_registry"]["selected"].append(
                {"name": "banknote", "version": "unverified"}
            )
        )
        self.assert_rejected(
            lambda p: p["historical_overlap"]["datasets_examined_in_prior_C2_G1"].remove("sonar")
        )

    def test_milestone_schedule_primary_comparison_and_stopping_fail_closed(self):
        self.assert_rejected(lambda p: p["experimental_design"].update(early_stop=True))
        self.assert_rejected(
            lambda p: p["experimental_design"]["milestones_proposed"].append(50000)
        )
        self.assert_rejected(
            lambda p: p["experimental_design"]["milestones_proposed"].remove(5000)
        )
        self.assert_rejected(
            lambda p: p["experimental_design"]["milestones_proposed"].__setitem__(0, 1)
        )
        self.assert_rejected(
            lambda p: p["experimental_design"]["primary_comparison"].update(
                to_iteration=100000
            )
        )
        self.assert_rejected(
            lambda p: p["experimental_design"].update(max_iterations_proposed=100000)
        )
        self.assert_rejected(
            lambda p: p["experimental_design"].update(backend_conditions=["numpy_sequential"])
        )

    def test_denominators_graph_replicates_and_plan_count_fail_closed(self):
        self.assert_rejected(
            lambda p: p["estimands"]["primary"].update(denominator="all nodes")
        )
        self.assert_rejected(
            lambda p: p["experimental_design"].update(number_of_fits_if_approved_and_feasible=70)
        )
        self.assert_rejected(
            lambda p: p["experimental_design"].update(label_mask_replicates_proposed=1)
        )
        self.assert_rejected(
            lambda p: p["experimental_design"].update(no_adaptive_resampling=False)
        )
        self.assert_rejected(
            lambda p: p["experimental_design"].update(graph_policy="convert to CSR")
        )
        self.assert_rejected(
            lambda p: p["p0_reference"].update(source_inventory_is_not_authorization=False)
        )
        self.assert_rejected(
            lambda p: p["p0_reference"].update(source_inventory_sha256="b" * 64)
        )

    def test_seed_and_exploratory_only_inference_cannot_drift_silently(self):
        self.assert_rejected(
            lambda p: p["experimental_design"].update(
                seed_policy="reuse one seed across every backend",
            )
        )
        self.assert_rejected(
            lambda p: p["estimands"]["primary"].update(
                uncertainty="95% CI from naive node-level bootstrap",
            )
        )
        self.assert_rejected(
            lambda p: p["analysis_plan"].update(
                decision_logic="declare significance if p < 0.05",
            )
        )

    def test_digest_is_deterministic_and_detects_edit(self):
        a = copy.deepcopy(self.proposal)
        b = {key: a[key] for key in reversed(list(a))}
        self.assertEqual(p1._canonical_digest(a), p1._canonical_digest(b))
        b["question"] = "changed question"
        self.assertNotEqual(p1._canonical_digest(a), p1._canonical_digest(b))

    def test_script_is_inert_standard_library_only(self):
        source = Path(p1.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import numpy", "import pcc", "import sklearn", "import numba",
            "import Cython", "fit_predict(", "pcc_propagate(",
            "read_csv(", "requests.", "urllib.",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"P1_ready_to_freeze": False', source)
        self.assertIn('"execution_authorized": False', source)
        self.assertIn('"model_fits_executed": 0', source)


if __name__ == "__main__":
    unittest.main()
