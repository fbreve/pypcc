"""Audit gates for the independent, Git-only three-backend source inventory.

Runs only in the dedicated research workflow after fetching the three pinned
experimental remote-tracking refs. Never imports a PCC backend or fits data.
"""
import copy
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from experiments import preflight_three_backend_sources as preflight


ROOT = Path(__file__).resolve().parents[1]


class ThreeBackendSourceInventoryTests(unittest.TestCase):
    def test_actual_source_only_inventory_is_exact_and_deterministic(self):
        first = preflight.audit(ROOT)
        second = preflight.audit(ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first["source_validation"], "pass")
        self.assertEqual(first["model_fits_executed"], 0)
        self.assertFalse(first["execution_authorized"])
        self.assertFalse(first["scientific_protocol_frozen"])
        self.assertFalse(first["real_dataset_inspected"])
        self.assertIsNone(first["execution_preflight_sha256"])
        self.assertGreater(len(first["blockers"]), 0)
        self.assertRegex(first["source_inventory_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            first["source_inventory_sha256"],
            preflight._sha_json(first["source_inventory"]),
        )
        self.assertEqual(
            set(first["source_inventory"]["research_branches"]),
            {"numpy", "numba", "cython"},
        )
        for name, config in preflight.BACKENDS.items():
            info = first["source_inventory"]["research_branches"][name]
            self.assertEqual(info["head_commit"], config["head"])
            self.assertEqual(info["base_commit"], preflight.BASE)
            self.assertEqual(set(info["research_file_blobs"]), set(config["files"]))
            self.assertEqual(len(info["research_file_blobs"]), len(config["files"]))
            for blob in info["research_file_blobs"].values():
                self.assertRegex(blob, r"^[0-9a-f]{40}$")

    def test_canonical_json_hash_ignores_dictionary_key_order(self):
        self.assertEqual(
            preflight._sha_json({"a": 1, "b": {"y": 2, "x": 3}}),
            preflight._sha_json({"b": {"x": 3, "y": 2}, "a": 1}),
        )
        self.assertNotEqual(
            preflight._sha_json({"a": 1}),
            preflight._sha_json({"a": 2}),
        )

    def test_changed_head_fails_closed(self):
        mutated = copy.deepcopy(preflight.BACKENDS)
        mutated["numba"]["head"] = "0" * 40
        with patch.object(preflight, "BACKENDS", mutated):
            with self.assertRaises(preflight.PreflightBlocked):
                preflight.audit(ROOT)

    def test_changed_scope_fails_closed(self):
        mutated = copy.deepcopy(preflight.BACKENDS)
        mutated["cython"]["files"] = tuple(
            f for f in mutated["cython"]["files"]
            if f != "experiments/live_cython_milestones.py"
        )
        with patch.object(preflight, "BACKENDS", mutated):
            with self.assertRaisesRegex(
                preflight.PreflightBlocked, "changed-file set differs"
            ):
                preflight.audit(ROOT)

    def test_changed_production_blob_pin_fails_closed(self):
        with patch.object(preflight, "CYTHON_PRODUCTION_BLOB", "f" * 40):
            with self.assertRaisesRegex(
                preflight.PreflightBlocked, "production blob"
            ):
                preflight.audit(ROOT)

    def test_moved_remote_tracking_head_fails_closed(self):
        with patch.object(
            preflight, "_verify_remote_head",
            side_effect=preflight.PreflightBlocked("remote branch moved"),
        ):
            with self.assertRaisesRegex(
                preflight.PreflightBlocked, "remote branch moved"
            ):
                preflight.audit(ROOT)

    def test_no_runtime_or_network_imports(self):
        source = (ROOT / "experiments" / "preflight_three_backend_sources.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "import pcc", "import numpy", "import numba", "import Cython",
            "fit_predict(", "pcc_propagate(", "urllib", "requests",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("execution_authorized\": False", source)
        self.assertIn("model_fits_executed\": 0", source)


if __name__ == "__main__":
    unittest.main()
