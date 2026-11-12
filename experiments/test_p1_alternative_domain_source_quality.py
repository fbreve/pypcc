"""Synthetic-only tests for unselected UCI 292 and 379 source-data candidates."""
import io
import json
from pathlib import Path
import unittest
import zipfile

from experiments import p1_alternative_domain_source_quality as a


def make_archive(dataset_id, *,
                 bad_header=False, invalid_code=False,
                 duplicate_across_classes=False,
                 path=None):
    spec = a.DATASETS[dataset_id]
    cols = spec["all_columns"]
    names = list(spec["class_counts"])
    labels = [v for name in names for v in
              [name] * spec["class_counts"][name]]
    if dataset_id == 292:
        head = list(cols)
        if bad_header:
            head[0] = "Region"
        lines = [",".join(head)]
        for i, label in enumerate(labels):
            vals = [
                label, str(i % 3 + 1),
                *[str((i + 1) * (j + 2) + j) for j in range(6)],
            ]
            if duplicate_across_classes and i == spec["class_counts"][names[0]]:
                # Same 6-dimensional features as first item but other class.
                vals[2:] = [str(j + 2 + j) for j in range(6)]
            if invalid_code and i == 0:
                vals[2] = "NaN"
            lines.append(",".join(vals))
    else:
        head = list(cols)
        if bad_header:
            head[3] = "UnexpectedFeature"
        lines = ["@relation synthetic_no_fits"]
        for name in head:
            lines.append("@attribute " + name + " numeric")
        lines.append("@data")
        for i, label in enumerate(labels):
            k = i
            if duplicate_across_classes and i == spec["class_counts"][names[0]]:
                k = 0
            code = []
            for j in range(9):
                code.append(str(k % 3 - 1))
                k //= 3
            if invalid_code and i == 0:
                code[0] = "2"
            lines.append(",".join(code + [label]))
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            path if path is not None else spec["source_member_basename"],
            "\n".join(lines) + "\n",
        )
    return result.getvalue()


class AlternativesTest(unittest.TestCase):
    def test_wholesale_channel_target_excludes_region_from_features(self):
        report = a.inspect_source(292, payload_override=make_archive(292))
        self.assertEqual(report["rows"], 440)
        self.assertEqual(report["proposed_target"], "Channel")
        self.assertEqual(report["features_excluded"], ["Region"])
        self.assertEqual(len(report["features_used"]), 6)
        self.assertEqual(report["class_support_by_uci_code"], {"1": 298, "2": 142})
        self.assertEqual(sum(report["region_support_if_wholesale"].values()), 440)
        self.assertEqual(report["exact_duplicate_full_row_surplus"], 0)
        self.assertEqual(report["conflicting_label_feature_patterns"], 0)
        self.assertEqual(report["datasets_selected"], 0)
        self.assertFalse(report["execution_authorized"])
        self.assertEqual(report["PCC_fits_executed"], 0)
        self.assertFalse(report["target_task_independently_reviewed"])

    def test_phishing_three_class_integer_codes_not_distance_approved(self):
        report = a.inspect_source(379, payload_override=make_archive(379))
        self.assertEqual(report["rows"], 1353)
        self.assertEqual(report["class_support_by_uci_code"],
                         {"-1": 702, "0": 103, "1": 548})
        self.assertEqual(len(report["features_used"]), 9)
        self.assertIsNone(report["region_support_if_wholesale"])
        self.assertFalse(report["categorical_features_euclidean_ready"])
        self.assertEqual(report["exact_repeated_feature_vector_surplus_rows"], 0)
        self.assertFalse(report["execution_authorized"])

    def test_repeated_features_across_labels_flagged_but_not_silently_removed(self):
        for dataset_id in (292, 379):
            report = a.inspect_source(
                dataset_id, payload_override=make_archive(
                    dataset_id, duplicate_across_classes=True,
                ),
            )
            self.assertGreater(report["conflicting_label_feature_patterns"], 0)
            self.assertGreater(report["exact_repeated_feature_vector_surplus_rows"], 0)
            self.assertEqual(report["rows"], a.DATASETS[dataset_id]["n"])
            self.assertEqual(report["datasets_selected"], 0)

    def test_invalid_schema_and_numerical_data_fail(self):
        for dataset_id in (292, 379):
            with self.assertRaises(a.CandidateReviewBlocked):
                a.inspect_source(dataset_id, payload_override=make_archive(
                    dataset_id, bad_header=True,
                ))
            with self.assertRaises(a.CandidateReviewBlocked):
                a.inspect_source(dataset_id, payload_override=make_archive(
                    dataset_id, invalid_code=True,
                ))

    def test_zip_traversal_wrong_task_and_extra_member_rejected(self):
        for dataset_id in (292, 379):
            with self.assertRaises(a.CandidateReviewBlocked):
                a.inspect_source(dataset_id, payload_override=make_archive(
                    dataset_id,
                    path="../" + a.DATASETS[dataset_id]["source_member_basename"],
                ))
            with self.assertRaises(a.CandidateReviewBlocked):
                a.inspect_source(dataset_id, payload_override=b"not a zip")
            with self.assertRaises(a.CandidateReviewBlocked):
                a.inspect_source(
                    dataset_id, payload_override=make_archive(
                        dataset_id,
                        path="subfolder/other_unexpected.csv",
                    ),
                )

    def test_only_known_https_uci_endpoints_allowed(self):
        for url in (
            "http://archive.ics.uci.edu/static/public/292/x.zip",
            "https://example.com/static/public/292/x.zip",
            "https://user:password@archive.ics.uci.edu/static/public/292/x.zip",
            "https://archive.ics.uci.edu/other/292/x.zip",
        ):
            with self.assertRaises(a.CandidateReviewBlocked):
                a.allowed_url(url)
        for d in a.DATASETS.values():
            self.assertEqual(a.allowed_url(d["source_url"]), d["source_url"])

    def test_pinned_source_quality_rejects_modified_duplicate_metrics_and_approval(self):
        reference = json.loads((
            Path(__file__).resolve().parents[1] / "docs" /
            "current_p1_alternative_domain_quality_results_2026-11-12.json"
        ).read_text(encoding="utf-8"))
        for case in reference["datasets"]:
            sample = {
                "uci_id": case["id"],
                "source": {
                    "source_url": case["source_url"],
                    "archive_sha256": case["archive_sha256"],
                    "data_member": case["member_name"],
                    "data_member_sha256": case["member_sha256"],
                },
                "rows": case["instances"],
                "features_used": case.get(
                    "feature_names",
                    a.DATASETS[case["id"]]["numeric_features"],
                ),
                "features_excluded": (
                    [case["excluded_feature"]] if case["id"] == 292 else []
                ),
                "proposed_target": case["target"],
                "exact_repeated_feature_vector_surplus_rows":
                    case["exact_repeated_feature_surplus_rows"],
                "exact_duplicate_full_row_surplus":
                    case["exact_full_row_duplicate_surplus"],
                "conflicting_label_feature_patterns":
                    case["conflicting_class_feature_patterns"],
                "distinct_feature_patterns": case["unique_feature_patterns"],
                "class_support_by_uci_code": (
                    case["channel_counts_by_official_code"] if case["id"] == 292
                    else case["class_support_by_official_code"]
                ),
                "categorical_features_euclidean_ready": False,
            }
            ok = a.enforce_published_source_quality(sample, reference)
            self.assertTrue(ok["matches_prior_source_only_quality"])
            self.assertTrue(ok["source_quality_record_is_not_P1_or_P2_approval"])
            bad = json.loads(json.dumps(sample))
            bad["source"]["archive_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                a.CandidateReviewBlocked, "archive_sha256"
            ):
                a.enforce_published_source_quality(bad, reference)
            bad = json.loads(json.dumps(sample))
            bad["conflicting_label_feature_patterns"] += 1
            with self.assertRaisesRegex(
                a.CandidateReviewBlocked, "conflicting_class_feature_patterns"
            ):
                a.enforce_published_source_quality(bad, reference)
        approved = json.loads(json.dumps(reference))
        approved["execution_authorized"] = True
        with self.assertRaises(a.CandidateReviewBlocked):
            a.enforce_published_source_quality(sample, approved)

    def test_no_production_code_import_or_model_fit(self):
        source = Path(a.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import pcc", "from pypcc", "import numba", "import sklearn",
            "fit_predict(", "pcc_propagate(", "build_graph(", "k_neighbors(",
        ):
            self.assertNotIn(forbidden, source)
        for dataset_id in a.DATASETS:
            self.assertTrue(a.DATASETS[dataset_id]["target_approval_required"])


if __name__ == "__main__":
    unittest.main()
