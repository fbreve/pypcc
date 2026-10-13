"""Small fixtures and short loops only; no downloads or performance campaigns."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.fixtures import (Case, array_hash, mnist_indices, prepare_fixture,
                                 prepare_mnist, save_fixture, select_labels)
from benchmarks.run import execution_order, main, plan
from benchmarks.runner import (PARAMETERS, PCC, environment_manifest, expected_backend,
                               measure, seed_backend, summarize, verify_backend)


class FixtureTests(unittest.TestCase):
    def test_mnist_subset_prefix_and_explicit_preparation_overwrite(self):
        small = mnist_indices(5000, 42)
        large = mnist_indices(10000, 42)
        np.testing.assert_array_equal(small, large[:5000])
        np.testing.assert_array_equal(small, mnist_indices(5000, 42))
        self.assertEqual(len(set(large)), 10000)
        with tempfile.TemporaryDirectory() as temp, \
                patch("benchmarks.fixtures.datasets.fetch_openml", side_effect=AssertionError("no download")):
            with self.assertRaises(FileExistsError):
                prepare_mnist(temp)

    def test_builtin_and_synthetic_determinism(self):
        for case in (Case("iris", 150, 4, 3),
                     Case("tiny", 24, 3, 3, k_nn=3, labels_per_class=1, synthetic=True)):
            arrays, metadata = prepare_fixture(case)
            again, repeated = prepare_fixture(case)
            self.assertEqual(metadata["hashes"], repeated["hashes"])
            self.assertEqual(metadata["fixture_sha256"], repeated["fixture_sha256"])
            self.assertEqual(metadata["labeled_nodes"], case.particles)
            self.assertEqual(set(arrays["labels"]) - {-1}, set(arrays["truth"]))
            for node, degree in enumerate(arrays["degrees"]):
                neighbors = arrays["neighbors"][node, :degree]
                self.assertNotIn(node, neighbors)
                self.assertEqual(len(neighbors), len(set(neighbors)))
                for target in neighbors:
                    self.assertIn(node, arrays["neighbors"][target, :arrays["degrees"][target]])
            np.testing.assert_array_equal(arrays["x"], again["x"])

    def test_fraction_labels_and_seed(self):
        truth = np.repeat(np.arange(3), 10)
        case = Case("tiny", 30, 2, 3, labels_per_class=None, labeled_fraction=.4)
        first = select_labels(truth, case)
        np.testing.assert_array_equal(first, select_labels(truth, case))
        self.assertEqual(np.sum(first != -1), 12)
        self.assertEqual(set(first)-{-1}, {0, 1, 2})
        self.assertFalse(np.array_equal(first, select_labels(truth, replace(case, label_seed=1))))
        with self.assertRaises(ValueError):
            prepare_fixture(Case("bad", 6, 2, 3, k_nn=2, labels_per_class=2, synthetic=True))

    def test_portable_hash_and_fixture_storage(self):
        values = np.arange(12, dtype=np.int64).reshape(3, 4)
        self.assertEqual(array_hash(values), array_hash(np.asfortranarray(values)))
        self.assertEqual(array_hash(values), array_hash(values.astype(">i8")))
        arrays, metadata = prepare_fixture(Case("tiny", 12, 2, 2, k_nn=2, labels_per_class=1, synthetic=True))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fixture"
            save_fixture(output, arrays, metadata)
            original = (output / "arrays.npz").read_bytes()
            with self.assertRaises(FileExistsError):
                save_fixture(output, arrays, metadata)
            self.assertEqual(original, (output / "arrays.npz").read_bytes())
            with np.load(output / "arrays.npz", allow_pickle=False) as stored:
                for name, value in arrays.items():
                    self.assertEqual(array_hash(stored[name]), array_hash(value))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.case = Case("tiny", 12, 2, 2, k_nn=2, labels_per_class=1, synthetic=True)
        self.arrays, self.metadata = prepare_fixture(self.case)

    def test_identity_and_fallback_rejection(self):
        model = PCC(impl="numpy")
        self.assertIs(verify_backend(model, "numpy"), expected_backend("numpy"))
        with patch.object(model, "_get_backend_fn", return_value=lambda: None):
            with self.assertRaisesRegex(RuntimeError, "fallback rejected"):
                verify_backend(model, "numpy")
        with patch("benchmarks.runner.importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                expected_backend("cython")

    def test_fresh_state_and_repeatability_all_available_backends(self):
        tested = []
        for name in ("numpy", "numba", "cython"):
            try:
                expected_backend(name)
            except RuntimeError:
                continue
            tested.append(name)
            measure(self.arrays, self.case, name, 2, False, 7, steady_state=False)
            first = measure(self.arrays, self.case, name, 5, False, 7)
            next_numpy = np.random.random()
            second = measure(self.arrays, self.case, name, 5, False, 7)
            self.assertEqual(next_numpy, np.random.random())
            self.assertEqual(first["state_hashes"], second["state_hashes"])
            self.assertEqual(first["effective_iterations"], 5)
            self.assertEqual(first["stop_reason"], "max_iter")
            self.assertTrue(first["reached_max_iter"])
            self.assertFalse(first["stopped_before_cap"])
            self.assertAlmostEqual(first["fit_predict_s"], first["initialization_and_dispatch_s"] +
                                   first["propagation_s"] + first["postprocessing_s"])
            # Compare against an ordinary fit: wrapping must preserve state and RNG.
            model = PCC(impl=name)
            model.set_graph(self.arrays["neighbors"], self.arrays["degrees"], k_nn=self.case.k_nn)
            seed_backend(name, 7)
            predicted = model.fit_predict(self.arrays["labels"], max_iter=5, early_stop=False, **PARAMETERS)
            self.assertEqual(next_numpy, np.random.random())
            self.assertEqual(array_hash(predicted), first["state_hashes"]["predictions"])
            self.assertEqual(array_hash(model.node.dominance), first["state_hashes"]["dominance"])
            for field, value in (("positions", model.part.curnode), ("strengths", model.part.strength),
                                 ("distances", model.part.dist_table), ("owndeg", model.owndeg)):
                self.assertEqual(array_hash(value), first["state_hashes"][field])
        self.assertIn("numpy", tested)
        self.assertEqual(self.metadata["hashes"], {name: array_hash(value) for name, value in self.arrays.items()})

    def test_early_stop_and_cap_reporting(self):
        case = Case("isolated", 6, 1, 2, k_nn=1, labels_per_class=1, synthetic=True)
        arrays = dict(neighbors=np.full((6, 1), -1, dtype=np.int64), degrees=np.zeros(6, dtype=np.int64),
                      labels=np.array([0, -1, -1, 1, -1, -1]), truth=np.array([0, 0, 0, 1, 1, 1]))
        for name in ("numpy", "numba", "cython"):
            try:
                expected_backend(name)
            except RuntimeError:
                continue
            measure(arrays, case, name, 2, True, 0, steady_state=False)
            stopped = measure(arrays, case, name, 30, True, 0)
            capped = measure(arrays, case, name, 5, True, 0)
            self.assertEqual(stopped["effective_iterations"], 21)
            self.assertEqual(stopped["stop_reason"], "early_stop")
            self.assertTrue(stopped["stopped_before_cap"])
            self.assertFalse(stopped["reached_max_iter"])
            self.assertEqual(capped["effective_iterations"], 5)
            self.assertEqual(capped["stop_reason"], "max_iter")
            at_cap = measure(arrays, case, name, 21, True, 0)
            self.assertEqual(at_cap["stop_reason"], "early_stop")
            self.assertTrue(at_cap["reached_max_iter"])
            self.assertFalse(at_cap["stopped_before_cap"])

    def test_numba_unexpected_compilation_rejected(self):
        try:
            expected_backend("numba")
        except RuntimeError:
            self.skipTest("Numba unavailable")
        measure(self.arrays, self.case, "numba", 2, False, 0, steady_state=False)
        with patch("benchmarks.runner.signatures", side_effect=[("old",), ("new",)]):
            with self.assertRaisesRegex(RuntimeError, "unexpected Numba compilation"):
                measure(self.arrays, self.case, "numba", 2, False, 0)

    def test_summary_statistics_and_no_case_pooling(self):
        rows = []
        for value in (1., 2., 3., 4., 100.):
            row = dict(case_id="a", fixture_sha256="hash", backend="numpy", mode="fixed", max_iter=10,
                       initialization_and_dispatch_s=value, propagation_s=value, postprocessing_s=value,
                       fit_predict_s=value*3, effective_iterations=10, propagation_s_per_iteration=value/10,
                       reached_max_iter=True, stop_reason="max_iter")
            rows.append(row)
        rows.append(dict(rows[0], case_id="b"))
        summaries = summarize(rows)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["propagation_s"]["median"], 3.)
        self.assertEqual(summaries[0]["propagation_s"]["iqr"], 2.)
        self.assertEqual(summaries[0]["cap_fraction"], 1.)

    def test_manifest_and_order(self):
        manifest = environment_manifest({"test": True}, ["numpy"])
        json.dumps(manifest, allow_nan=False)
        self.assertIn("commit", manifest["repository"])
        self.assertIn("numpy", manifest["environment"]["versions"])
        self.assertIn("timestamp_utc", manifest)
        self.assertIsNone(manifest["cython_extension"])
        backends = ["numpy", "numba", "cython"]
        self.assertEqual([execution_order(backends, 0, r)[0] for r in range(3)], backends)

    def test_dry_run_no_loading_and_expensive_gate(self):
        with patch("benchmarks.run.prepare_fixture", side_effect=AssertionError("must not load")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--datasets", "mnist_10000", "--dry-run"])
        workload = json.loads(output.getvalue())
        self.assertTrue(workload["requires_expensive_opt_in"])
        self.assertEqual(workload["measured_executions"], 15)
        self.assertEqual(workload["particle_update_cap"], 45000)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            destination = Path(temp) / "untouched"
            with self.assertRaises(SystemExit):
                main(["--datasets", "mnist_10000", "--execute", "--output", str(destination)])
            self.assertFalse(destination.exists())

    def test_cli_storage_overwrite_and_optional_missing(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                patch("benchmarks.fixtures.datasets.fetch_openml", side_effect=AssertionError("no download")):
            destination = Path(temp) / "run"
            args = ["--datasets", "iris", "--backends", "numpy", "--iterations", "2", "--repeats", "2",
                    "--execute", "--output", str(destination)]
            main(args)
            rows = [json.loads(line) for line in (destination / "measurements.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["state_hashes"], rows[1]["state_hashes"])
            original = (destination / "manifest.json").read_bytes()
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(original, (destination / "manifest.json").read_bytes())
            optional = Path(temp) / "optional"
            main(["--datasets", "mnist_5000", "--backends", "numpy", "--skip-missing-optional",
                  "--allow-expensive", "--execute", "--output", str(optional)])
            status = json.loads((optional / "completion.json").read_text())
            self.assertEqual(status["status"], "no_measurements")
            self.assertEqual(len(status["skipped"]), 1)


if __name__ == "__main__":
    unittest.main()
