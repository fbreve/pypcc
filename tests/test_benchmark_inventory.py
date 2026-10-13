"""Dataset and planning checks only; network and campaign execution prohibited."""
import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from benchmarks import fixtures as fixture_module
from benchmarks.historical import ENTRIES, FROZEN, PROTOCOL, fingerprint, load_historical
from benchmarks.fixtures import Case, REAL_DATASETS, FixturePreparer, prepare_fixture, select_labels
from benchmarks.run import main
from experiments.patience_data import load


class InventoryTests(unittest.TestCase):
    def test_exact_frozen_inventory(self):
        self.assertEqual(len(ENTRIES), 20)
        self.assertEqual(sum("openml_id" in e for e in ENTRIES.values()), 16)
        self.assertTrue(all(e["version"] == 1 for e in ENTRIES.values() if "openml_id" in e))
        self.assertFalse(any(name.startswith("mnist") for name in ENTRIES))
        for name, frozen in FROZEN.items():
            self.assertEqual(REAL_DATASETS[name][:3],
                             (frozen["nodes"], frozen["processed_features"], frozen["classes"]))

    def test_builtins_and_historical_label_masks(self):
        for name in ("iris", "wine", "digits", "breast_cancer"):
            raw, truth, indices, source = load_historical(name)
            frozen = FROZEN[name]
            self.assertEqual(fingerprint(raw, truth), frozen["raw_data_fingerprint"])
            np.testing.assert_array_equal(indices, np.arange(len(truth)))
            case = Case(name, *REAL_DATASETS[name][:3])
            for seed in PROTOCOL["seeds"]:
                labels = select_labels(truth, replace(case, label_seed=seed))
                self.assertEqual(fingerprint(labels), frozen["label_fingerprints"][str(seed)])
            self.assertEqual(source["historical_name"], ENTRIES[name]["name"])
        a, meta = prepare_fixture(Case("iris", 150, 4, 3, label_seed=10))
        b, again = prepare_fixture(Case("iris", 150, 4, 3, label_seed=10))
        self.assertEqual(meta["fixture_sha256"], again["fixture_sha256"])
        self.assertTrue(meta["source"]["label_fingerprint_matches"])
        self.assertTrue(np.isfinite(a["x"]).all())
        np.testing.assert_allclose(a["x"].mean(axis=0), 0, atol=1e-14)

    def test_missing_and_corrupt_cache_never_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(FileNotFoundError):
                load_historical("germancredit", tmp)
            row = FROZEN["germancredit"]
            meta = dict(version=1, name=row["source_name"], url="https://invalid.example/source",
                        md5_checksum="wrong", default_target_attribute="class")
            Path(tmp, "31.json").write_text(json.dumps(dict(data_set_description=meta)))
            Path(tmp, "31.arff").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_historical("germancredit", tmp)
            meta["version"] = 2
            Path(tmp, "31.json").write_text(json.dumps(dict(data_set_description=meta)))
            with self.assertRaisesRegex(ValueError, "version"):
                load_historical("germancredit", tmp)

    def test_original_encoding_and_imputation(self):
        rows = ["1,a,no", "?,b,no", "3,?,no", "5,a,no",
                "7,b,yes", "9,?,yes", "11,a,yes", "13,b,yes"]
        raw = ("@relation test\n@attribute numeric numeric\n@attribute category {a,b,?}\n"
               "@attribute class {no,yes}\n@data\n" + "\n".join(rows)).encode()
        meta = dict(version=1, name="test", url="unused", md5_checksum=hashlib.md5(raw).hexdigest(),
                    default_target_attribute="class")
        def fetch(url, path):
            return json.dumps(dict(data_set_description=meta)).encode() if path.suffix == ".json" else raw
        x, scaled, truth, info = load(dict(openml_id=1, version=1), Path("unused"), fetch=fetch)
        self.assertEqual(x.shape, (8, 4))
        self.assertEqual(x[1, 0], 7.)
        np.testing.assert_array_equal(x[:, 1:].sum(axis=1), np.ones(8))
        np.testing.assert_array_equal(x[:, 1], [0, 0, 1, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(truth, [0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-14)
        self.assertEqual(info["nominal_features"], 1)

    def test_planning_without_loading(self):
        with patch("benchmarks.run.prepare_fixture", side_effect=AssertionError("no fixtures")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("no download")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--historical", "--iterations", "10000", "--label-seeds",
                  *map(str, PROTOCOL["seeds"]), "--dry-run"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["case_count"], 200)
        self.assertEqual(result["preparation_groups"], 20)
        self.assertEqual(result["graph_builds"], 20)
        self.assertEqual(result["graph_pair_feature_upper_work_total"], 5406433612)
        self.assertEqual(result["measured_executions"], 3600)
        self.assertEqual(result["warmup_executions"], 1200)
        self.assertEqual(result["measured_iteration_cap"], 36000000)
        self.assertEqual(result["particle_update_cap"], sum(r["classes"] for r in FROZEN.values()) * 3 * 10 * 18 * 10000)
        self.assertTrue(result["requires_expensive_opt_in"])
        self.assertFalse(any("mnist" in row["case_id"] for row in result["cases"]))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--historical", "--dry-run"])
        self.assertTrue(all(row["case_id"].endswith("labels10") for row in json.loads(output.getvalue())["cases"]))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--historical", "--datasets", "mnist_5000"])

    def test_modes_and_explicit_preparation_are_separate(self):
        for mode in ("fixed", "early_stopping"):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(["--datasets", "iris", "--mode", mode, "--dry-run"])
            result = json.loads(output.getvalue())
            self.assertEqual(result["mode"], mode)
            self.assertEqual(result["cases"][0]["feature_arrays_bytes"], 9600)
        with patch("benchmarks.run.load_historical") as loader, \
                patch("benchmarks.run.execute", side_effect=AssertionError("no execution")):
            main(["--datasets", "germancredit", "--prepare-historical", "unused"])
            loader.assert_called_once_with("germancredit", Path("unused"), download=True)
        with patch("benchmarks.run.load_historical", side_effect=AssertionError("no download")), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--historical", "--prepare-historical", "unused", "--execute"])

    def test_reuse_preserves_all_arrays_and_fixture_hashes(self):
        first = Case("iris", 150, 4, 3, label_seed=10)
        cases = [first, replace(first, label_seed=11), replace(first, label_seed=12, labels_per_class=2)]
        preparer = FixturePreparer()
        with patch("benchmarks.fixtures.load_data", wraps=fixture_module.load_data) as loading, \
                patch("benchmarks.fixtures.build_knn_graph", wraps=fixture_module.build_knn_graph) as graph:
            prepared = [preparer.prepare(case) for case in cases]
            self.assertEqual(loading.call_count, 1)
            self.assertEqual(graph.call_count, 1)
        for case, (arrays, meta) in zip(cases, prepared):
            independent, expected = prepare_fixture(case)
            self.assertEqual(meta["hashes"], expected["hashes"])
            self.assertEqual(meta["fixture_sha256"], expected["fixture_sha256"])
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], independent[key])
            self.assertEqual(meta["preparation"]["source_case_id"], first.case_id)
            self.assertTrue(meta["source"]["graph_fingerprint_matches"])
        self.assertIs(prepared[0][0]["neighbors"], prepared[1][0]["neighbors"])
        self.assertIsNot(prepared[0][0]["labels"], prepared[1][0]["labels"])
        with self.assertRaises(ValueError):
            prepared[1][0]["neighbors"][0, 0] = 99
        prepared[1][1]["source"]["historical_name"] = "modified"
        self.assertEqual(preparer.prepare(first)[1]["source"]["historical_name"], "Iris")

    def test_reuse_key_invalidation_and_failure(self):
        case = Case("small", 24, 2, 2, k_nn=3, labels_per_class=1, synthetic=True)
        preparer = FixturePreparer()
        with patch("benchmarks.fixtures.prepare_fixture", wraps=prepare_fixture) as prepare:
            preparer = FixturePreparer(prepare)
            for item, mnist, historical in ((case, None, None), (replace(case, label_seed=1), None, None),
                                           (replace(case, data_seed=1), None, None),
                                           (replace(case, k_nn=4), None, None), (case, Path("a"), None),
                                           (case, Path("a"), Path("b"))):
                preparer.prepare(item, mnist, historical)
            self.assertEqual(prepare.call_count, 5)
            with patch("benchmarks.fixtures.load_data", side_effect=ValueError("failed")), self.assertRaises(ValueError):
                preparer.prepare(replace(case, data_seed=2))
            self.assertFalse(preparer.prepare(case)[1]["preparation"]["reused"])

    def test_reuse_timing_boundaries(self):
        case = Case("iris", 150, 4, 3, label_seed=10)
        preparer = FixturePreparer()
        with patch("benchmarks.fixtures.time.perf_counter", side_effect=[10., 12., 15., 19., 20., 23.]):
            _, first = preparer.prepare(case)
            _, reused = preparer.prepare(replace(case, label_seed=11))
        self.assertEqual(first["timings"], dict(dataset_load_s=2., preprocessing_and_labels_s=3., graph_build_s=4.))
        self.assertEqual(reused["timings"], dict(dataset_load_s=0., preprocessing_and_labels_s=3., graph_build_s=0.))

    def test_processed_and_ordered_graph_mismatches_fail(self):
        from benchmarks.historical import load as historical_load
        def changed_load(*args, **kwargs):
            raw, scaled, truth, info = historical_load(*args, **kwargs)
            scaled[0, 0] += 1e-15
            return raw, scaled, truth, info
        with patch("benchmarks.historical.load", side_effect=changed_load), self.assertRaisesRegex(ValueError, "unexplained frozen processed-data"):
            load_historical("iris")
        from benchmarks.fixtures import build_knn_graph
        def reordered(*args, **kwargs):
            neighbors, degrees = build_knn_graph(*args, **kwargs)
            neighbors[0, :2] = neighbors[0, :2][::-1]
            return neighbors, degrees
        with patch("benchmarks.fixtures.build_knn_graph", side_effect=reordered), self.assertRaisesRegex(ValueError, "unexplained frozen ordered graph"):
            FixturePreparer().prepare(Case("iris", 150, 4, 3))
        _, custom = prepare_fixture(Case("iris", 150, 4, 3, k_nn=3))
        self.assertFalse(custom["source"]["frozen_fingerprints"]["graph"]["applicable"])

    def test_both_pilot_presets(self):
        for stage, updates in (("100", 162000), ("1000", 1620000)):
            with patch("benchmarks.run.execute", side_effect=AssertionError("no measurements")), \
                    patch("benchmarks.run.prepare_fixture", side_effect=AssertionError("no preparation")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                main(["--pilot-stage", stage, "--dry-run"])
            result = json.loads(output.getvalue())
            self.assertEqual(result["mode"], "fixed")
            self.assertEqual(result["repetitions"], 6)
            self.assertEqual(result["backends"], ["numpy", "numba", "cython"])
            self.assertEqual(result["case_count"], 6)
            self.assertEqual(result["graph_builds"], 3)
            self.assertEqual(result["measured_executions"], 108)
            self.assertEqual(result["warmup_executions"], 36)
            self.assertEqual(result["particle_update_cap"], updates)
            self.assertFalse(result["requires_expensive_opt_in"])
        for extra in (["--iterations", "50"], ["--mode", "early_stopping"], ["--repeats", "3"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--pilot-stage", "100", *extra])

    def test_preparation_report_no_duplicate_graph_charge(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            main(["--datasets", "iris", "--label-seeds", "10", "11", "--iterations", "2", "--backends", "numpy",
                  "--execute", "--output", str(Path(tmp) / "run")])
            report = json.loads(Path(tmp, "run", "preparation.json").read_text())
            self.assertEqual(report["graph_builds"], 1)
            self.assertFalse(report["fixtures"][0]["reused"])
            self.assertTrue(report["fixtures"][1]["reused"])
            self.assertEqual(report["fixtures"][1]["dataset_load_s"], 0.)
            self.assertEqual(report["fixtures"][1]["graph_build_s"], 0.)
            self.assertEqual(report["totals"]["graph_build_s"], report["fixtures"][0]["graph_build_s"])
            summary = json.loads(Path(tmp, "run", "summary.json").read_text())
            self.assertEqual(len(summary), 2)
            self.assertTrue(all("initialization_and_dispatch_s" in r and "propagation_s" in r for r in summary))

    def test_cache_only_audit_failure_and_report_overwrite(self):
        from benchmarks.validate_inventory import audit, main as audit_main
        with patch("urllib.request.urlopen", side_effect=AssertionError("no network")):
            report = audit(["iris", "wine"], None)
            self.assertEqual(report["status"], "match")
            self.assertTrue(all(check["expected"] == check["actual"] for row in report["datasets"] for check in row["checks"].values()))
        with patch("benchmarks.validate_inventory.prepare_fixture", side_effect=ValueError("unexplained mismatch")):
            self.assertEqual(audit(["iris"], None)["status"], "failed")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                audit_main(["--datasets", "iris"])
        with tempfile.TemporaryDirectory() as tmp, patch("benchmarks.validate_inventory.audit", side_effect=AssertionError("no preparation")):
            destination = Path(tmp, "existing.json")
            destination.write_text("preserve")
            with self.assertRaises(FileExistsError):
                audit_main(["--output", str(destination)])
            self.assertEqual(destination.read_text(), "preserve")

    def test_reused_inputs_keep_backend_state_and_rng_reproducible(self):
        from benchmarks.runner import expected_backend, measure
        case = Case("iris", 150, 4, 3, label_seed=11)
        preparer = FixturePreparer()
        preparer.prepare(replace(case, label_seed=10))
        reused, metadata = preparer.prepare(case)
        independent, expected = prepare_fixture(case)
        for backend in ("numpy", "numba", "cython"):
            try:
                expected_backend(backend)
            except RuntimeError:
                continue
            measure(reused, case, backend, 2, False, 0, steady_state=False)
            first = measure(reused, case, backend, 5, False, 0)
            next_random = np.random.random()
            baseline = measure(independent, case, backend, 5, False, 0)
            self.assertEqual(first["state_hashes"], baseline["state_hashes"])
            self.assertEqual(next_random, np.random.random())
        self.assertEqual(metadata["hashes"], {key: fixture_module.array_hash(array) for key, array in reused.items()})
