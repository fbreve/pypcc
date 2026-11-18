"""Adversarial, synthetic-only checks of bounded PCC literature evidence."""
import copy
from pathlib import Path
import json
import unittest

from experiments import review_p1_additional_public_papers as audit


class AdditionalPublicPaperEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(audit.REGISTER.read_text(encoding="utf-8"))

    def check_forgery_rejected(self, mutate):
        item = copy.deepcopy(self.data)
        mutate(item)
        with self.assertRaises(audit.ExposureScopeInvalid):
            audit.review(item)

    def test_original_seven_source_record_passes_as_limited_not_approved(self):
        result = audit.review(self.data)
        self.assertEqual(result["scope_integrity"], "PASS")
        self.assertEqual(result["additional_published_sources"], 6)
        self.assertEqual(result["additional_arxiv_preprints"], 1)
        self.assertEqual(result["total_scoped_sources_with_prior_three"], 10)
        self.assertTrue(result["ten_scoped_experimental_sections_without_candidate_names"])
        self.assertFalse(result["all_prior_research_or_aliases_checked"])
        self.assertFalse(result["candidate_dataset_novelty_proved"])
        self.assertFalse(result["patient_customer_grain_independence_proved"])
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(result["PCC_fits_executed"], 0)
        self.assertRegex(
            result["source_register_sha256_NOT_scientific_P1_approval"],
            "^[0-9a-f]{64}$",
        )

    def test_source_list_forgery_and_title_or_type_drift_rejected(self):
        self.check_forgery_rejected(lambda x:
            x["reviewed_experimental_sources"][2]["experimental_datasets"].remove("g241d"))
        self.check_forgery_rejected(lambda x:
            x["reviewed_experimental_sources"][3].update(year=2013))
        self.check_forgery_rejected(lambda x:
            x["reviewed_experimental_sources"][6].update(
                type="peer_reviewed_journal"))
        self.check_forgery_rejected(lambda x:
            x["reviewed_experimental_sources"][0].update(
                candidate_names_present=["Raisin"]))
        self.check_forgery_rejected(lambda x:
            x["reviewed_experimental_sources"].pop())
        self.check_forgery_rejected(lambda x:
            x.update(checked_extra_source_count=8))

    def test_invented_novelty_or_protocol_approval_rejected(self):
        for key, value in [
            ("historical_20_dataset_campaign_exhaustive", True),
            ("all_prior_research_checked", True),
            ("all_candidate_aliases_checked", True),
            ("clinical_or_subject_group_independence_proved", True),
            ("scientific_P1_approved", True),
            ("P1_scientific_sha256", "a" * 64),
            ("P2_execution_sha256", "b" * 64),
            ("dataset_selected_ids", [212,850,292]),
            ("PCC_real_data_fits_executed", 1),
            ("execution_authorized", True),
        ]:
            self.check_forgery_rejected(lambda x, k=key, v=value:
                                         x.update({k:v}))
        self.check_forgery_rejected(lambda x:x.update(
            negative_claim="NONE_OF_THE_THREE_CANDIDATES_EVER_USED_IN_PCC"))

    def test_no_model_graph_or_downloads(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import pcc", "import numpy", "import numba", "import sklearn",
            "fit_predict(", "PCC.fit(", "build_knn_graph(", "urlopen(",
            "requests.get(", "SeedSequence(", "np.random.",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
