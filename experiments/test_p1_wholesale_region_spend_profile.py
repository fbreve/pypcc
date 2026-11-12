"""Synthetic-only source-quality checks for derived Channel and Region effects."""
import copy
import io
import json
from pathlib import Path
import unittest
import zipfile

from experiments import p1_alternative_domain_source_quality as alt
from experiments import p1_wholesale_region_spend_profile as profile
from experiments.test_p1_alternative_domain_source_quality import make_archive


def synthetic_rows():
    raw = make_archive(292)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member = z.read(alt.DATASETS[292]["source_member_basename"])
    return alt.parse_wholesale(member)


class WholesaleDescriptiveSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.columns, cls.rows = synthetic_rows()

    def test_complete_region_channel_counts_and_no_model_changes(self):
        r = profile.describe_wholesale_rows(self.rows, self.columns)
        self.assertEqual(r["n"], 440)
        self.assertEqual(r["channel_totals"], {"1": 298, "2": 142})
        self.assertEqual(sum(r["region_totals"].values()), 440)
        self.assertEqual(len(r["features"]), 6)
        self.assertNotIn("Region", r["features"])
        self.assertNotIn("Channel", r["features"])
        self.assertTrue(r["region_is_excluded_from_model_features"])
        self.assertTrue(r["no_p_value_or_automatic_exclusion"])
        self.assertTrue(r["does_not_establish_independent_customers_or_batches"])
        self.assertTrue(r["does_not_prescribe_scaling_or_PCC_graph"])
        self.assertFalse(r["P1_dataset_selected"])
        self.assertFalse(r["P1_scientific_approval"])
        self.assertFalse(r["execution_authorized"])
        self.assertEqual(r["PCC_fits_executed"], 0)
        self.assertIsNone(r["P2_execution_preflight_sha256"])
        for name, d in r["features"].items():
            self.assertEqual(d["values_modified_or_excluded"], 0)
            self.assertLessEqual(d["min"], d["median_q50"])
            self.assertLessEqual(d["median_q50"], d["q95"])
            self.assertLessEqual(d["q95"], d["max"])

    def test_descriptive_contingency_detects_region_channel_association(self):
        r = profile.describe_wholesale_rows(self.rows, self.columns)
        self.assertGreater(r["cramers_v_descriptive_not_causal"], 0)
        self.assertLessEqual(r["cramers_v_descriptive_not_causal"], 1)
        self.assertTrue(all(0 < share < 1 for share in
                            r["channel_1_share_by_region"].values()))
        self.assertNotIn("p_value", r)

    def test_quantile_is_defined_before_source_results(self):
        from decimal import Decimal
        x = list(map(Decimal, ("1", "2", "3", "4", "5", "6", "7", "8")))
        self.assertEqual(profile.nearest_rank(x, Decimal("0.25")), 2)
        self.assertEqual(profile.nearest_rank(x, Decimal("0.5")), 4)
        self.assertEqual(profile.nearest_rank(x, Decimal("0.95")), 8)
        self.assertEqual(profile.P_QUANTILES,
                         tuple(Decimal(t) for t in
                               ("0.25", "0.50", "0.75", "0.95", "0.99")))

    def test_source_quality_rejects_bad_schemas_and_source_values(self):
        bad_cols = list(self.columns)
        bad_cols[1] = "Region-not-real"
        with self.assertRaises(profile.DiagnosticBlocked):
            profile.describe_wholesale_rows(self.rows, bad_cols)
        cases = [
            ("Region", "99"), ("Channel", "3"),
            ("Fresh", "-1"), ("Grocery", "NaN"),
        ]
        for col, value in cases:
            rows = [list(row) for row in self.rows]
            rows[0][self.columns.index(col)] = value
            with self.assertRaises((profile.DiagnosticBlocked,
                                    alt.CandidateReviewBlocked)):
                profile.describe_wholesale_rows(rows, self.columns)

    def test_pinned_drift_validator_rejects_unapproved_and_changed_values(self):
        r = profile.describe_wholesale_rows(self.rows, self.columns)
        r["source_url"] = alt.DATASETS[292]["source_url"]
        r["archive_sha256"] = "a" * 64
        r["selected_member_sha256"] = "b" * 64
        ref = copy.deepcopy(r)
        self.assertTrue(
            profile.validate_pinned_diagnostics(copy.deepcopy(r), ref)[
                "reproduced_pinned_quality_only"]
        )
        ref["features"]["Fresh"]["q95"] += 1
        with self.assertRaisesRegex(profile.DiagnosticBlocked, "features"):
            profile.validate_pinned_diagnostics(r, ref)
        ref = copy.deepcopy(r)
        ref["P1_scientific_approval"] = True
        with self.assertRaises(profile.DiagnosticBlocked):
            profile.validate_pinned_diagnostics(r, ref)

    def test_no_PCC_graph_or_model_fit_or_individual_records_export(self):
        source = Path(profile.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import pcc", "import sklearn", "import numba",
            "fit_predict(", "pcc_propagate(", "kneighbors_graph(",
            "NearestNeighbors(", "train_test_split(", "k_neighbors(",
        ):
            self.assertNotIn(forbidden, source)
        r = profile.describe_wholesale_rows(self.rows, self.columns)
        self.assertTrue(r["source_rows_not_exported"])
        self.assertNotIn("data_rows", r)


if __name__ == "__main__":
    unittest.main()
