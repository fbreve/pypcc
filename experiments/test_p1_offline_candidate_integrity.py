"""Synthetic-only tests for the OFFLINE, no-fit candidate-quality reader.

All rows are generated inside temporary test fixtures and are NOT real UCI data.
"""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments import p1_offline_candidate_integrity as checker


def synthetic_arff(task_id, *, repeated=False, conflicting=False):
    task = checker.TASKS[task_id]
    keys = list(task["labels"])
    counts = task["labels"]
    if all(x is not None for x in counts.values()):
        labels = [k for k in keys for _ in range(counts[k])]
    else:
        labels = [keys[i % len(keys)] for i in range(task["n"])]
    lines = ["@relation synthetic_only"]
    for i in range(task["d"]):
        lines.append(f"@attribute feature{i} numeric")
    lines.append("@attribute Class {" + ",".join(keys) + "}")
    lines.append("@data")
    for i, label in enumerate(labels):
        feat_index = i
        if repeated and i == 1:
            feat_index = 0
        if conflicting and i == counts[keys[0]]:
            feat_index = 0
        values = [str(feat_index * 31 + k / 10) for k in range(task["d"])]
        lines.append(",".join(values + [label]))
    return ("\n".join(lines) + "\n").encode()


class OfflineCandidateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def write(self, name, body):
        p = self.directory / name
        p.write_bytes(body)
        return p

    def test_vertebral_three_class_exact_row_and_class_counts(self):
        f = self.write("column_3C_weka.arff", synthetic_arff(212))
        report = checker.review(212, f)
        self.assertEqual(report["rows"], 310)
        self.assertEqual(report["numeric_features"], 6)
        self.assertEqual(report["class_support"], {"Hernia": 60, "Normal": 100, "Spondylolisthesis": 150})
        self.assertEqual(report["exact_repeated_feature_rows"], 0)
        self.assertEqual(report["conflicting_label_feature_patterns"], 0)
        self.assertEqual(report["archive_sha256"], checker.sha(f.read_bytes()))
        self.assertEqual(report["datasets_selected"], 0)
        self.assertFalse(report["execution_authorized"])
        self.assertEqual(report["model_fits_executed"], 0)
        self.assertFalse(report["source_authenticity_independently_verified"])

    def test_rice_binary_arff_nonparametric_quality_counts(self):
        f = self.write("Rice_Cammeo_Osmancik.arff", synthetic_arff(545))
        report = checker.review(545, f)
        self.assertEqual(report["rows"], 3810)
        self.assertEqual(report["class_support"], {"Cammeo": 1905, "Osmancik": 1905})
        self.assertEqual(report["numeric_features"], 7)
        self.assertFalse(report["near_duplicates_checked"])

    def test_duplicate_and_conflicting_labels_report_not_silenced(self):
        f = self.write("column_3C_weka.arff",
                       synthetic_arff(212, repeated=True, conflicting=True))
        report = checker.review(212, f)
        self.assertEqual(report["exact_repeated_feature_rows"], 2)
        self.assertEqual(report["exact_duplicate_full_rows"], 1)
        self.assertEqual(report["conflicting_label_feature_patterns"], 1)

    def test_zip_archive_requires_unique_member_and_hashes(self):
        source = synthetic_arff(212)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("dataset/column_3C_weka.arff", source)
            z.writestr("notes.txt", "SYNTHETIC ONLY")
        path = self.write("test.zip", output.getvalue())
        report = checker.review(212, path)
        self.assertEqual(report["source_member"], "dataset/column_3C_weka.arff")
        self.assertEqual(report["source_member_sha256"], checker.sha(source))
        self.assertEqual(report["archive_sha256"], checker.sha(output.getvalue()))

    def test_wrong_binary_task_rejected(self):
        f = self.write("column_2C_weka.arff", synthetic_arff(212))
        with self.assertRaisesRegex(checker.FixtureRejected, "Wrong task"):
            checker.review(212, f)

    def test_malformed_and_insufficient_data_rejected(self):
        good = synthetic_arff(212)
        for bad in (
            good.replace(b"0.0", b"NaN", 1),
            good.replace(b"0.0", b"Infinity", 1),
            good.replace(b"@data", b"@wrong"),
            good.replace(b"@attribute feature0 numeric", b"@attribute feature0 string"),
            good[:-30],
        ):
            f = self.write("column_3C_weka.arff", bad)
            with self.assertRaises(checker.FixtureRejected):
                checker.review(212, f)
        with self.assertRaises(checker.FixtureRejected):
            checker.review(850, f)

    def test_unsafe_zip_path_rejected(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("../column_3C_weka.arff", synthetic_arff(212))
        f = self.write("unsafe.zip", out.getvalue())
        with self.assertRaisesRegex(checker.FixtureRejected, "Unsafe ZIP"):
            checker.review(212, f)

    def test_ambiguous_zip_file_selection_rejected(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("column_3C_weka.arff", synthetic_arff(212))
            z.writestr("column_2C_weka.arff", synthetic_arff(212))
        f = self.write("multiple.zip", out.getvalue())
        with self.assertRaisesRegex(checker.FixtureRejected, "Select exactly one"):
            checker.review(212, f)
        self.assertEqual(
            checker.review(212, f, member="column_3C_weka.arff")["rows"], 310,
        )

    def test_exact_official_raisin_two_sheet_layout_is_allowed(self):
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl not installed")
        wb = Workbook()
        ws = wb.active
        ws.title = "Raisin_Grains_Dataset"
        ws.append(["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity",
                   "ConvexArea", "Extent", "Perimeter", "Class"])
        for i in range(900):
            ws.append([i + 1 + k / 100 for k in range(7)] +
                      ["Kecimen" if i < 450 else "Besni"])
        cite = wb.create_sheet("Citation_Request")
        cite["A1"] = "Citation Request : source paper"
        cite["A2"] = "Synthetic fixture for verifier tests only"
        out = io.BytesIO()
        wb.save(out)
        f = self.write("Raisin_Dataset.xlsx", out.getvalue())
        result = checker.review(850, f)
        self.assertEqual(result["class_support"], {"Besni": 450, "Kecimen": 450})

        cite.title = "Unexpected"
        out = io.BytesIO()
        wb.save(out)
        bad = self.write("Raisin_Dataset.xlsx", out.getvalue())
        with self.assertRaisesRegex(checker.FixtureRejected, "exact UCI two-sheet"):
            checker.review(850, bad)

    def test_raisin_xlsx_read_only_and_optional_dependency(self):
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl not installed; dedicated CI must install it")
        wb = Workbook()
        ws = wb.active
        ws.append(["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity",
                   "ConvexArea", "Extent", "Perimeter", "Class"])
        for i in range(900):
            ws.append([i + 1 + k / 100 for k in range(7)] +
                      ["Kecimen" if i < 450 else "Besni"])
        out = io.BytesIO()
        wb.save(out)
        f = self.write("Raisin_Dataset.xlsx", out.getvalue())
        report = checker.review(850, f)
        self.assertEqual(report["rows"], 900)
        self.assertEqual(report["class_support"], {"Besni": 450, "Kecimen": 450})
        self.assertEqual(report["exact_repeated_feature_rows"], 0)
        self.assertEqual(report["model_fits_executed"], 0)

        ws.cell(2, 1, "=1+2")
        out = io.BytesIO()
        wb.save(out)
        bad = self.write("Raisin_Dataset.xlsx", out.getvalue())
        with self.assertRaisesRegex(checker.FixtureRejected, "Formulas forbidden"):
            checker.review(850, bad)

    def test_prior_official_source_record_rejects_byte_and_result_drift(self):
        from experiments import p1_official_uci_no_fit as source
        prior = json.loads((
            Path(__file__).resolve().parents[1] /
            "docs/current_p1_official_uci_quality_results_2026-11-12.json"
        ).read_text(encoding="utf-8"))
        for case in prior["datasets"]:
            fake = {
                "quality": {
                    "dataset_id": case["uci_id"],
                    "source_member_sha256": case["selected_member_sha256"],
                    "rows": case["observed_rows"],
                    "numeric_features": case["numeric_features"],
                    "class_support": case["class_support"],
                    "exact_duplicate_full_rows": case["exact_duplicate_full_rows"],
                    "exact_repeated_feature_rows": case["exact_repeated_feature_rows"],
                    "conflicting_label_feature_patterns":
                        case["conflicting_label_feature_patterns"],
                },
                "source": {
                    "download_sha256": case["source_zip_sha256"],
                    "requested_url": case["url"],
                },
                "execution_authorized": False, "PCC_fits_executed": 0,
            }
            ok = source.verify_against_previous_observation(fake, prior)
            self.assertTrue(ok["matches_previous_source_only_audit"])
            self.assertTrue(ok["previous_source_only_audit_is_not_scientific_approval"])
            corrupt = json.loads(json.dumps(fake))
            corrupt["source"]["download_sha256"] = "f" * 64
            with self.assertRaisesRegex(source.ExternalDataBlocked, "source_zip_sha256"):
                source.verify_against_previous_observation(corrupt, prior)
            corrupt = json.loads(json.dumps(fake))
            corrupt["quality"]["class_support"] = {"tampered": 1}
            with self.assertRaisesRegex(source.ExternalDataBlocked, "class_support"):
                source.verify_against_previous_observation(corrupt, prior)
        self.assertEqual(prior["selected_dataset_ids"], [])
        self.assertFalse(prior["execution_authorized"])
        self.assertEqual(prior["PCC_real_data_fits_executed"], 0)

    def test_predeclared_near_duplicate_grid_and_cross_class_pairs(self):
        from experiments import p1_near_duplicate_audit as near
        matrix = [
            [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.00001], [10.0, 10.0],
        ]
        result = near.analyze_features(matrix, ["A", "A", "B", "B"])
        self.assertEqual(result["unordered_pairs_inspected"], 6)
        self.assertEqual(
            [row["rms_std_euclidean_threshold_le"]
             for row in result["thresholds"]],
            [1e-6, 1e-4, 1e-3, 1e-2],
        )
        self.assertEqual(result["thresholds"][0]["unordered_pair_count"], 1)
        self.assertEqual(result["thresholds"][0]["cross_class_pair_count"], 0)
        self.assertEqual(result["thresholds"][1]["unordered_pair_count"], 3)
        self.assertEqual(result["thresholds"][1]["cross_class_pair_count"], 2)
        self.assertEqual(result["thresholds"][1]["distinct_rows_involved"], 3)
        self.assertEqual(result["minimum_distinct_row_rms_standardized_distance"], 0)
        self.assertFalse(result["dataset_selected_for_P1"])
        self.assertEqual(result["model_fits_executed"], 0)
        self.assertFalse(result["execution_authorized"])
        self.assertTrue(
            all(a["unordered_pair_count"] <= b["unordered_pair_count"]
                for a, b in zip(result["thresholds"], result["thresholds"][1:]))
        )

    def test_near_duplicate_screen_rejects_constant_and_nonfinite_features(self):
        from experiments import p1_near_duplicate_audit as near
        with self.assertRaises(near.NearDuplicateAuditBlocked):
            near.analyze_features([[1, 0], [1, 1]], ["A", "B"])
        with self.assertRaises(near.NearDuplicateAuditBlocked):
            near.analyze_features([[0, 0], [1, float("nan")]], ["A", "B"])
        with self.assertRaises(near.NearDuplicateAuditBlocked):
            near.analyze_features([[0, 0], [1, 1]], ["A"])
        with self.assertRaises(near.NearDuplicateAuditBlocked):
            near.analyze_features([[0, 0], [1, 1]], ["A", "B"])

    def test_near_duplicate_archive_is_only_local_and_no_graph(self):
        from experiments import p1_near_duplicate_audit as near
        f = self.write("column_3C_weka.arff", synthetic_arff(212))
        result = near.audit_local_archive(
            f, dataset_id=212, member=None,
        )
        self.assertEqual(result["n_source_rows"], 310)
        self.assertEqual(result["features"], 6)
        self.assertEqual(result["unordered_pairs_inspected"], 310 * 309 // 2)
        self.assertFalse(result["subject_or_batch_identity_verified"])
        self.assertEqual(result["model_fits_executed"], 0)
        source = Path(near.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import pcc", "import sklearn", "import numba", "pcc_propagate(",
            "fit_predict(", "requests.get(", "urllib.request", "urlopen(",
            "kneighbors_graph(", "NearestNeighbors(",
        ):
            self.assertNotIn(forbidden, source)

    def test_no_runtime_imports_or_network_calls(self):
        source = Path(checker.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import pcc", "import numpy", "import sklearn", "import numba",
            "fit_predict(", "pcc_propagate(", "requests.get(",
            "urllib.request", "urlopen(", "http.client",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"execution_authorized": False', source)
        self.assertIn('"model_fits_executed": 0', source)


if __name__ == "__main__":
    unittest.main()
