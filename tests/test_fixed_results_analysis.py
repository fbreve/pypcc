"""Raw-file validation with synthetic records; no benchmark fits."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from tools.analyze_fixed_results import BACKENDS, audit_run, describe


class RawAnalysisTests(unittest.TestCase):
    def make_run(self, root):
        fixture = {"fixture_sha256": "fixture", "source": {"frozen_fingerprints": {
            "graph": {"matches": True, "actual": "graph", "expected": "graph"}}},
            "labeled_class_counts": {"0": 3, "1": 3}}
        directory = root / "fixtures" / "wine-labels10"
        directory.mkdir(parents=True)
        (directory / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
        (root / "completion.json").write_text(json.dumps({"status": "complete",
            "measured_executions": 18, "fixtures": [{"case_id": "wine-labels10",
                                                     "fixture_sha256": "fixture"}]}), encoding="utf-8")
        for filename, budget, calls in [("measurements.jsonl", 10000, range(6)),
                                       ("warmups.jsonl", 2, ("first", "warm"))]:
            rows = []
            for backend, call in itertools.product(BACKENDS, calls):
                repeat = call if isinstance(call, int) else 0
                order = BACKENDS[repeat % 3:] + BACKENDS[:repeat % 3]
                row = {"case_id": "wine-labels10", "backend": backend,
                       "repeat": repeat, "call": call, "mode": "fixed",
                       "effective_iterations": budget, "max_iter": budget, "checks": 0,
                       "fixture_sha256": "fixture", "order_index": order.index(backend),
                       "backend_identity": {"numpy": "pcc_numpy.kernel", "numba": "pcc_numba.kernel",
                                            "cython": "pcc_step.kernel"}[backend],
                       "state_hashes": {"state": backend}, "accuracy_unlabeled": .5,
                       "initialization_and_dispatch_s": .01, "postprocessing_s": .01,
                       "fit_predict_s": 1.02, "propagation_s": 1000. if repeat == 5 else 1.}
                rows.append(row)
            (root / filename).write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")

    def test_outlier_is_retained_and_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_run(root)
            _, report = audit_run(root, ("wine",), (10,))
            self.assertEqual(report["measurements"], 18)
            self.assertEqual(report["warmups"], 6)
            self.assertEqual(report["outliers_excluded"], 0)
            self.assertEqual(report["tables"][0]["max"], 1000.)
            raw = root / "measurements.jsonl"
            lines = raw.read_text(encoding="utf-8").splitlines()
            raw.write_text("\n".join(lines[:-1] + [lines[0]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
                audit_run(root, ("wine",), (10,))

    def test_quartiles_match_linear_interpolation(self):
        stats = describe([1, 2, 3, 1000])
        self.assertEqual(stats["median"], 2.5)
        self.assertEqual(stats["q1"], 1.75)
        self.assertEqual(stats["q3"], 252.25)


if __name__ == "__main__":
    unittest.main()
