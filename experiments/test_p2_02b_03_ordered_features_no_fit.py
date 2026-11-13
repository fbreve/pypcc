"""Synthetic-only adversarial tests; no official downloads, PCC, masks or graphs."""
import io
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from experiments import p2_02b_03_ordered_features_no_fit as target


def sample_wholesale():
    columns = list(target.WHOLESALE_COLUMNS)
    rows = [
        ["1", "3", "1", "2", "4", "5", "8", "13"],
        ["2", "1", "2", "4", "8", "10", "16", "26"],
        ["1", "2", "4", "8", "16", "20", "32", "52"],
    ]
    spec = {"uci_id": 292, "original_n_rows": 3, "n_features": 6,
            "class_support": {"1": 2, "2": 1}}
    return spec, columns, rows


class OrderedFeatureTests(unittest.TestCase):
    def test_derived_target_and_original_row_order(self):
        spec, columns, rows = sample_wholesale()
        X, labels, support = target.numeric_rows_and_labels(
            spec, columns, rows, list(range(2, 8)))
        self.assertEqual(X.shape, (3, 6))
        self.assertEqual(labels, ("1", "2", "1"))
        self.assertEqual(support, spec["class_support"])
        self.assertEqual(X[0, 0], 1.0)
        self.assertEqual(X[2, 0], 4.0)
        first = target.array_fingerprint(X, target.WHOLESALE_X)
        permuted = target.array_fingerprint(X[[1, 0, 2], :], target.WHOLESALE_X)
        self.assertNotEqual(
            first["shape_dtype_feature_order_and_numeric_bytes_sha256"],
            permuted["shape_dtype_feature_order_and_numeric_bytes_sha256"])
        names = list(target.WHOLESALE_X)
        names[0], names[1] = names[1], names[0]
        renamed = target.array_fingerprint(X, names)
        self.assertNotEqual(
            first["shape_dtype_feature_order_and_numeric_bytes_sha256"],
            renamed["shape_dtype_feature_order_and_numeric_bytes_sha256"])

    def test_exact_approved_population_ddof0_and_log1p(self):
        spec, columns, rows = sample_wholesale()
        X, _, _ = target.numeric_rows_and_labels(spec, columns, rows,
                                                  list(range(2, 8)))
        baseline = X.copy()
        Z, evidence = target.approved_transform(X, wholesale=True)
        expected_input = np.log1p(baseline)
        expected = (expected_input -
                    np.mean(expected_input, axis=0, dtype=np.float64)) / np.std(
                        expected_input, axis=0, ddof=0, dtype=np.float64)
        np.testing.assert_array_equal(Z, expected)
        np.testing.assert_array_equal(X, baseline)
        self.assertTrue(evidence["labels_used_to_select_or_scale_features"] is False)
        self.assertEqual(evidence["pop_std_ddof"], 0)
        self.assertLess(evidence["mean_abs_max_after_zscore"], 1e-12)
        Z2, plain = target.approved_transform(X, wholesale=False)
        expected_no_log = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=0)
        np.testing.assert_array_equal(Z2, expected_no_log)
        self.assertIn("population_zscore", plain["method"])

    def test_reject_nonfinite_missing_negative_and_constant_features(self):
        spec, columns, rows = sample_wholesale()
        bad = [r.copy() for r in rows]
        bad[1][2] = ""
        with self.assertRaises(target.OrderedSourceBlocked):
            target.numeric_rows_and_labels(spec, columns, bad, list(range(2, 8)))
        bad = [r.copy() for r in rows]
        bad[1][2] = "NaN"
        with self.assertRaises(target.OrderedSourceBlocked):
            target.numeric_rows_and_labels(spec, columns, bad, list(range(2, 8)))
        X, _, _ = target.numeric_rows_and_labels(spec, columns, rows,
                                                  list(range(2, 8)))
        negative = X.copy()
        negative[0, 0] = -1
        with self.assertRaises(target.OrderedSourceBlocked):
            target.approved_transform(negative, wholesale=True)
        constant = X.copy()
        constant[:, 0] = 5
        with self.assertRaises(target.OrderedSourceBlocked):
            target.approved_transform(constant, wholesale=False)

    def test_reject_class_leak_and_class_support_drift(self):
        spec, columns, rows = sample_wholesale()
        with self.assertRaises(target.OrderedSourceBlocked):
            target.numeric_rows_and_labels(spec, columns, rows,
                                            [0, 3, 4, 5, 6, 7])
        wrong = [r.copy() for r in rows]
        wrong[1][0] = "1"
        with self.assertRaises(target.OrderedSourceBlocked):
            target.numeric_rows_and_labels(spec, columns, wrong,
                                            list(range(2, 8)))

    def test_fingerprint_rejects_wrong_shape_and_dtype(self):
        with self.assertRaises(target.OrderedSourceBlocked):
            target.array_fingerprint(np.array([[1, 2]], dtype=np.int64), ["a", "b"])
        with self.assertRaises(target.OrderedSourceBlocked):
            target.array_fingerprint(np.array([[np.inf]], dtype=np.float64), ["a"])

    def test_P1_original_member_hash_reverified_before_parsing(self):
        def zf(rows):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as archive:
                archive.writestr("Wholesale customers data.csv", rows)
            return buf.getvalue()
        sample = b"Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen\n"
        raw = zf(sample)
        spec = {"uci_id": 292, "file_name": "Wholesale customers data.csv",
                "file_sha256": target.sha256(sample)}
        self.assertEqual(target.load_member_bytes(raw, spec), sample)
        bad = {**spec, "file_sha256": "0" * 64}
        with self.assertRaises(target.OrderedSourceBlocked):
            target.load_member_bytes(raw, bad)

    def test_label_encoding_and_order_changes_change_hash(self):
        a = target.label_fingerprint(("Besni", "Kecimen", "Besni"),
                                     ("Besni", "Kecimen"))
        b = target.label_fingerprint(("Kecimen", "Besni", "Besni"),
                                     ("Besni", "Kecimen"))
        self.assertNotEqual(a["original_target_order_sha256"],
                            b["original_target_order_sha256"])
        with self.assertRaises(target.OrderedSourceBlocked):
            target.label_fingerprint(("Besni", "Kecimen"),
                                     ("Kecimen", "Besni"))


if __name__ == "__main__":
    unittest.main()
