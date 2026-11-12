"""Zero-data adversarial checks for P1 scientific decisions — NO model fitting."""
import copy
import json
from pathlib import Path
import unittest

from experiments import review_p1_scientific_decisions as gate


class ScientificDecisionPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = json.loads(gate.PACKET.read_text(encoding="utf-8"))
        cls.draft = json.loads(gate.DRAFT.read_text(encoding="utf-8"))
        cls.graph_source = gate.GRAPH_SOURCE.read_bytes()

    def test_actual_packet_passes_only_as_unapproved_reviewer_draft(self):
        result = gate.review(self.packet, self.draft, self.graph_source)
        self.assertEqual(result["proposal_checks"], "PASS")
        self.assertEqual(result["proposed_candidate_ids_not_selected"],
                         [212, 850, 292])
        self.assertTrue(result["source_provenance_pinned_for_review"])
        self.assertTrue(result["original_graph_source_blob_checked"])
        self.assertFalse(result["P1_scientifically_frozen"])
        self.assertFalse(result["P1_ready_to_freeze"])
        self.assertIsNone(result["P2_execution_preflight_sha256"])
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(result["PCC_fits_executed"], 0)
        self.assertGreaterEqual(result["review_pending_count"], 7)
        self.assertRegex(
            result["packet_sha256_NOT_scientific_protocol_approval"],
            r"^[0-9a-f]{64}$",
        )

    def reject_modified(self, mutate, *, draft=False, graph=False):
        p = copy.deepcopy(self.packet)
        d = copy.deepcopy(self.draft)
        g = self.graph_source
        if draft:
            mutate(d)
        elif graph:
            g = mutate(g)
        else:
            mutate(p)
        with self.assertRaises(gate.ScientificDecisionInvalid):
            gate.review(p, d, g)

    def test_forged_P1_P2_and_dataset_selection_always_rejected(self):
        self.reject_modified(lambda p: p.update(
            status="SCIENTIFICALLY_APPROVED"))
        self.reject_modified(lambda p: p.update(
            scientific_protocol_frozen=True))
        self.reject_modified(lambda p: p.update(
            scientific_protocol_sha256="a" * 64))
        self.reject_modified(lambda p: p.update(
            P2_execution_preflight_sha256="b" * 64))
        self.reject_modified(lambda p: p.update(execution_authorized=True))
        self.reject_modified(lambda p: p.update(real_PCC_fits_executed=1))
        self.reject_modified(lambda p: p["dataset_registry_selected"].append(212))
        self.reject_modified(lambda p: p.update(selected_dataset_count=3))
        self.reject_modified(lambda d: d["approval"].update(
            scientific_protocol_frozen=True), draft=True)

    def test_unreviewed_source_task_or_feature_change_rejected(self):
        self.reject_modified(lambda p: p[
            "candidate_shortlist_proposed_not_selected"].reverse())
        self.reject_modified(lambda p: p[
            "candidate_shortlist_proposed_not_selected"][1].update(
                n_rows=3810))
        self.reject_modified(lambda p: p[
            "candidate_shortlist_proposed_not_selected"][0].update(
                source_zip_sha256="0" * 64))
        self.reject_modified(lambda p: p[
            "candidate_shortlist_proposed_not_selected"][0].update(
                selected_file_name="column_2C_weka.arff"))
        self.reject_modified(lambda p: p[
            "candidate_shortlist_proposed_not_selected"][2].update(
                proposed_target="Region"))
        self.reject_modified(lambda p: p[
            "candidate_shortlist_proposed_not_selected"][2].update(
                excluded_X=["Channel"]))
        self.reject_modified(lambda p: p[
            "candidate_shortlist_proposed_not_selected"][1][
                "class_support"].update(Besni=449))
        self.reject_modified(lambda p: p["not_shortlisted"][1].update(
            reason="continuous numeric graph safe"))
        self.reject_modified(lambda p: p[
            "preprocessing_candidate_rule_not_approved"].update(
                reviewed_and_approved=True))

    def test_graph_and_original_P1_draft_drift_blocked(self):
        self.reject_modified(lambda p: p[
            "graph_candidate_rule_not_approved"].update(
                call="build_knn_graph(X,k_nn=25)"))
        self.reject_modified(lambda p: p[
            "graph_candidate_rule_not_approved"].update(
                dense_padded_minus_one=False))
        self.reject_modified(lambda p: p[
            "graph_candidate_rule_not_approved"].update(
                cross_platform_graph_identity_not_yet_verified=False))
        self.reject_modified(lambda p: p[
            "graph_candidate_rule_not_approved"].update(
                source_git_blob_sha1="b" * 40))
        self.reject_modified(lambda g: g + b"\n# changed\n", graph=True)
        self.reject_modified(lambda d: d["experimental_design"].update(
            max_iterations_proposed=30000), draft=True)
        self.reject_modified(lambda p: p[
            "scientific_design_preserved_from_P1_draft"].update(
                milestones=[0, 100, 50000]))
        self.reject_modified(lambda p: p[
            "scientific_design_preserved_from_P1_draft"].update(
                proposed_fit_count_if_all_approved=70))

    def test_windows_checkout_CRLF_is_same_pinned_git_blob_not_source_change(self):
        # Test the same on either checkout convention: first get Git's LF
        # bytes, then simulate the Windows CRLF working-tree materialization.
        canonical = self.graph_source.replace(b"\r\n", b"\n")
        converted = canonical.replace(b"\n", b"\r\n")
        self.assertNotEqual(
            gate.git_blob_sha1(converted),
            gate.git_blob_sha1(canonical),
        )
        report = gate.review(self.packet, self.draft, converted)
        self.assertTrue(report["original_graph_source_blob_checked"])
        self.assertFalse(report["execution_authorized"])
        changed = converted.replace(b"k_nn=10", b"k_nn=11", 1)
        self.assertNotEqual(changed, converted)
        with self.assertRaises(gate.ScientificDecisionInvalid):
            gate.review(self.packet, self.draft, changed)

    def test_digest_deterministic_and_no_backend_imports(self):
        a = copy.deepcopy(self.packet)
        b = dict(reversed(list(a.items())))
        self.assertEqual(gate.canonical_sha256(a), gate.canonical_sha256(b))
        b["purpose"] += " updated"
        self.assertNotEqual(gate.canonical_sha256(a), gate.canonical_sha256(b))
        src = Path(gate.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import numpy", "import sklearn", "import pcc",
            "import numba", "import Cython",
            "fit_predict(", "pcc_propagate(", "requests.get(",
            "urllib.request", "urlopen(", "np.random.",
        ):
            self.assertNotIn(forbidden, src)
        self.assertIn('"P1_ready_to_freeze": False', src)
        self.assertIn('"execution_authorized": False', src)
        self.assertIn('"PCC_fits_executed": 0', src)


if __name__ == "__main__":
    unittest.main()
