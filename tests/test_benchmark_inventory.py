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
from benchmarks.historical import ENTRIES, FROZEN, PROTOCOL, fingerprint, load_historical
from benchmarks.fixtures import Case, REAL_DATASETS, prepare_fixture, select_labels
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
