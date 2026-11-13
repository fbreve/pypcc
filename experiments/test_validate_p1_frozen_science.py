"""Approved-science P1 payload validation with synthetic-only adversarial mutations.

This suite never accesses UCI data, samples source labels, builds any graph,
imports the PCC model or compiles native backends.
"""
import copy
import json
import unittest

from experiments import validate_p1_frozen_science as p1


class FrozenP1ScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(p1.FROZEN.read_text(encoding="utf-8"))
        cls.draft = json.loads(p1.DRAFT.read_text(encoding="utf-8"))
        cls.decision = json.loads(p1.DECISIONS.read_text(encoding="utf-8"))
        cls.review = json.loads(p1.REVIEW.read_text(encoding="utf-8"))
        cls.graph = p1.GRAPH.read_bytes()

    def run_review(self, proto=None, graph=None):
        return p1.validate(
            self.protocol if proto is None else proto,
            self.draft, self.decision, self.review,
            self.graph if graph is None else graph,
        )

    def modify_and_reject(self, modifier):
        payload = copy.deepcopy(self.protocol)
        modifier(payload)
        with self.assertRaises(p1.FrozenScienceInvalid):
            self.run_review(payload)

    def test_human_approved_science_still_prohibits_P2_or_real_fit(self):
        report = self.run_review()
        self.assertEqual(report["science_status"],
                         "FROZEN_S1_S6_EXPLORATORY_ONLY")
        self.assertEqual(report["selected_source_uci_ids_as_research_protocol"],
                         [212, 850, 292])
        self.assertEqual(report["trial_count_IF_future_P2_approved"], 72)
        self.assertEqual(report["PCC_fits_executed"], 0)
        self.assertEqual(report["datasets_or_real_labels_processed"], 0)
        self.assertIsNone(report["P2_execution_preflight_sha256"])
        self.assertFalse(report["P2_execution_authorized"])
        self.assertEqual(len(report["scientific_protocol_sha256"]), 64)
        self.assertFalse(self.review["human_decision"]["maintainer_approval"])
        self.assertEqual(
            self.protocol["approval_origin"]["local_unpublished_experiment_history"],
            "UNKNOWN_NOT_DECLARED: approval to proceed with exploratory methodology is not a claim of no prior dataset use",
        )

    def test_no_data_novelty_forgery(self):
        self.modify_and_reject(lambda p:p["approval_origin"].update(
            local_unpublished_experiment_history="NONE"))
        self.modify_and_reject(lambda p:p["approval_origin"].update(
            does_NOT_approve_P2_execution=False))
        self.modify_and_reject(lambda p:p["ethics_and_limitations"].update(
            previously_used_locally="NEVER"))
        self.modify_and_reject(lambda p:p["selected_datasets"][0].update(
            clinical_subject_or_acquisition_batch_independence_established=True))
        self.modify_and_reject(lambda p:p["selected_datasets"][1].update(
            documented_prior_local_PCC_use="NEVER"))
        self.modify_and_reject(lambda p:p["selected_datasets"][2].update(
            target_is_derived=False))

    def test_exact_source_and_graph_choice_fail_closed(self):
        self.modify_and_reject(lambda p:p["selected_datasets"].reverse())
        self.modify_and_reject(lambda p:p["selected_datasets"][0].update(
            zip_sha256="0"*64))
        self.modify_and_reject(lambda p:p["selected_datasets"][1].update(
            class_support={"Besni": 449, "Kecimen": 451}))
        self.modify_and_reject(lambda p:p["selected_datasets"][2].update(
            excluded_predictors=["Channel"]))
        self.modify_and_reject(lambda p:p["selected_datasets"][2].update(
            actual_target="Region"))
        self.modify_and_reject(lambda p:p["graph"].update(k_nn=11))
        self.modify_and_reject(lambda p:p["graph"].update(
            graph_directed=True))
        self.modify_and_reject(lambda p:p["graph"].update(
            prohibit_CSR_propagation_replacement=False))
        self.modify_and_reject(lambda p:p["graph"].update(
            installation_route="set_graph"))
        with self.assertRaises(p1.FrozenScienceInvalid):
            self.run_review(graph=self.graph+b"# changed\n")

    def test_science_choices_are_frozen(self):
        self.modify_and_reject(lambda p:p["preprocessing"].update(
            no_row_exclusion=False))
        self.modify_and_reject(lambda p:p["preprocessing"].update(
            population_zscore_rule="zscore(ddof=1)"))
        self.modify_and_reject(lambda p:p["fit"].update(
            max_iter=30000))
        self.modify_and_reject(lambda p:p["fit"].update(
            early_stop=True))
        self.modify_and_reject(lambda p:p["fit"].update(
            p_grd=0.8))
        self.modify_and_reject(lambda p:p["fit"].update(
            milestones=[0,100,50000]))
        self.modify_and_reject(lambda p:p["fit"].update(
            total_proposed_fits=70))
        self.modify_and_reject(lambda p:p["rng_and_masks"].update(
            mask_seed_recipe="np.random.seed(0)"))
        self.modify_and_reject(lambda p:p["analysis"].update(
            inference="CONFIRMATORY_WITH_BOOTSTRAP_PVALUE"))
        self.modify_and_reject(lambda p:p["analysis"].update(
            primary_checkpoints=[0,50000]))

    def test_no_implicit_P2_approval(self):
        for key, value in (
            ("P2_preflight_status","APPROVED"),
            ("P2_preflight_sha256","1"*64),
            ("P2_execution_approved",True),
            ("real_data_fits_authorized",72),
            ("real_data_fits_executed",1),
            ("labels_sampling_or_graph_building_for_real_datasets_authorized_at_P1",True),
        ):
            self.modify_and_reject(lambda p,k=key,v=value:
                p["approvals_and_execution"].update({k:v}))

    def test_CRLf_checkout_allowed_only_for_identical_original_Git_blob(self):
        canonical=self.graph.replace(b"\r\n",b"\n")
        converted=canonical.replace(b"\n",b"\r\n")
        r=self.run_review(graph=converted)
        self.assertFalse(r["P2_execution_authorized"])
        with self.assertRaises(p1.FrozenScienceInvalid):
            self.run_review(graph=converted+b"# accidental change\r\n")

    def test_manifest_matches_exact_human_approved_scientific_SHA(self):
        manifest = json.loads(p1.MANIFEST.read_text(encoding="utf-8"))
        r = p1.verify_scientific_manifest(manifest, self.run_review())
        self.assertTrue(
            r["scientific_P1_SHA_manifest_matches_exact_approved_digest"]
        )
        self.assertFalse(r["P2_execution_authorized"])
        self.assertEqual(
            r["scientific_protocol_sha256"],
            "f0111f153ade3d72191c107ace039e1835e5c27db2cf4b3df8d610b41b2f4166",
        )

    def test_manifest_forgery_must_not_authorize_unreviewed_change(self):
        source = json.loads(p1.MANIFEST.read_text(encoding="utf-8"))
        result = self.run_review()
        bad_cases = [
            ("approved_P1_scientific_protocol_sha256", "0"*64),
            ("P2_execution_authorized", True),
            ("P2_preflight_sha256", "f"*64),
            ("any_real_PCC_fit_authorized", True),
            ("local_prior_PCC_use_on_selected_sets", "NEVER_USED"),
            ("real_PCC_fits_executed", 1),
        ]
        for key, value in bad_cases:
            modified = copy.deepcopy(source)
            modified[key] = value
            with self.subTest(key=key):
                with self.assertRaises(p1.FrozenScienceInvalid):
                    p1.verify_scientific_manifest(modified, result)
        changed = copy.deepcopy(result)
        changed["scientific_protocol_sha256"] = "1" * 64
        with self.assertRaises(p1.FrozenScienceInvalid):
            p1.verify_scientific_manifest(source, changed)

    def test_canonical_digest_is_deterministic_and_sensitive(self):
        x=copy.deepcopy(self.protocol)
        y=dict(reversed(list(x.items())))
        self.assertEqual(p1.canonical_sha256(x),p1.canonical_sha256(y))
        y["scientific_question"]+=" updated"
        self.assertNotEqual(p1.canonical_sha256(x),p1.canonical_sha256(y))


if __name__=="__main__":
    unittest.main()
