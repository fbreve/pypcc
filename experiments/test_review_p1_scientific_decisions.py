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

    def test_three_paper_public_history_evidence_is_bounded_not_an_approval(self):
        evidence = json.loads(gate.EXPOSURE.read_text(encoding="utf-8"))
        result = gate.review_public_literature_evidence(evidence)
        self.assertEqual(result["public_full_text_papers_scoped_checked"], 3)
        self.assertTrue(
            result["three_candidate_names_absent_from_checked_papers_only"]
        )
        self.assertFalse(result["all_history_or_private_matlab_checked"])
        self.assertFalse(result["paper_20_dataset_inventory_exhaustive"])
        self.assertFalse(result["data_independence_or_ethics_certified"])
        self.assertGreaterEqual(result["unreviewed_categories"], 5)
        self.assertRegex(
            result["public_literature_register_sha256_NOT_novelty_or_approval"],
            "^[0-9a-f]{64}$",
        )
        for field, value in [
            ("historical_exposure_completely_checked", True),
            ("execution_authorized", True),
            ("dataset_independence_verified", True),
            ("P1_scientific_protocol_sha256", "a" * 64),
        ]:
            forged = json.loads(json.dumps(evidence))
            forged[field] = value
            with self.assertRaises(gate.ScientificDecisionInvalid):
                gate.review_public_literature_evidence(forged)
        forged = json.loads(json.dumps(evidence))
        forged["reviewed_full_texts"][0]["experimental_datasets_named"] = [
            "Iris","Wine"
        ]
        with self.assertRaises(gate.ScientificDecisionInvalid):
            gate.review_public_literature_evidence(forged)
        forged = json.loads(json.dumps(evidence))
        forged["source_metadata"][2][
            "channel_as_proposed_derived_target_is_uci_default"
        ] = True
        with self.assertRaises(gate.ScientificDecisionInvalid):
            gate.review_public_literature_evidence(forged)

    def test_scoped_paper_registry_cannot_claim_full_novelty(self):
        self.reject_modified(lambda p: p[
            "publication_exposure_review_scope"].update(
                historical_20_dataset_inventory_is_exhaustive=True
            ))
        self.reject_modified(lambda p: p[
            "publication_exposure_review_scope"].update(
                all_prior_publications_or_unpublished_results_checked=True
            ))
        self.reject_modified(lambda p: p[
            "publication_exposure_review_scope"].update(
                dataset_novelty_or_external_validation_approved=True
            ))
        self.reject_modified(lambda p: p["evidence_files"].remove(
            "docs/current_p1_public_literature_exposure_2026-11-12.json"
        ))

    def test_six_scientific_recommendations_are_review_ready_not_approved(self):
        proposed = json.loads(gate.RESOLUTION.read_text(encoding="utf-8"))
        report = gate.review_scientific_resolution(
            proposed, self.draft, self.packet,
        )
        self.assertEqual(
            report["scientific_resolution_review"],
            "PASS_PENDING_HUMAN_DECISION",
        )
        self.assertEqual(report["proposals_for_explicit_approval"], 6)
        self.assertEqual(
            report["proposed_dataset_ids_not_selected"],
            [212, 850, 292],
        )
        self.assertFalse(report["all_six_recommendations_human_approved"])
        self.assertFalse(report["P1_ready_to_freeze"])
        self.assertFalse(report["execution_authorized"])
        self.assertEqual(report["real_PCC_fits_executed"], 0)
        self.assertRegex(
            report["resolution_checksum_NOT_scientific_protocol_approval"],
            r"^[0-9a-f]{64}$",
        )

    def test_resolution_cannot_forge_science_approval_or_selection(self):
        original = json.loads(gate.RESOLUTION.read_text(encoding="utf-8"))
        changes = [
            ("maintainer_approval", True),
            ("approval_timestamp", "2026-11-12T21:30:00-03:00"),
            ("selected_dataset_ids", [212, 850, 292]),
            ("scientific_protocol_frozen", True),
            ("scientific_protocol_sha256", "0" * 64),
            ("P2_execution_preflight_sha256", "1" * 64),
            ("execution_authorized", True),
            ("real_PCC_fits_performed", 1),
            ("new_generation_of_source_masks_or_graphs_authorized", True),
        ]
        for key, forged_value in changes:
            item = json.loads(json.dumps(original))
            item["human_decision"][key] = forged_value
            with self.subTest(field=key):
                with self.assertRaises(gate.ScientificDecisionInvalid):
                    gate.review_scientific_resolution(item, self.draft, self.packet)

    def test_six_review_gates_and_method_choices_cannot_change_silently(self):
        original = json.loads(gate.RESOLUTION.read_text(encoding="utf-8"))
        changes = [
            lambda p: p["recommendations"].pop(),
            lambda p: p["recommendations"][0].update(
                proposed_uci_ids=[212, 545, 292]),
            lambda p: p["recommendations"][0].update(
                recommend="APPROVED_EXPLORATORY_P1"),
            lambda p: p["recommendations"][1].update(feature_rule="Channel only"),
            lambda p: p["recommendations"][2].update(numeric_rule="raw spending"),
            lambda p: p["recommendations"][3].update(graph="use directed CSR"),
            lambda p: p["recommendations"][4].update(proposed_work="20 fits"),
            lambda p: p["recommendations"][5].update(
                recommend="RECOMMEND_CONFIRMATORY_CI"),
        ]
        for mutate in changes:
            item = json.loads(json.dumps(original))
            mutate(item)
            with self.assertRaises(gate.ScientificDecisionInvalid):
                gate.review_scientific_resolution(item, self.draft, self.packet)

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
