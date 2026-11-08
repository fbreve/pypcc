"""Regression: benchmark timing instrumentation must preserve native backend layout.

The production model selects C-order dominance for Numba with >=8 classes by
checking the resolved backend's function name. Timing wrappers must not change
that decision.
"""
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.fixtures import Case, prepare_fixture
from benchmarks.runner import measure
from pcc import ParticleCompetitionAndCooperation as PCC


class BenchmarkNativeLayoutParityTests(unittest.TestCase):
    def exercise(self, classes, backend, expected_c):
        case = Case("layout_parity", nodes=64, features=4, classes=classes,
                    labels_per_class=1, k_nn=3, label_seed=36113,
                    data_seed=36119, synthetic=True)
        arrays, _ = prepare_fixture(case)
        seen = []
        legacy_method = PCC._ParticleCompetitionAndCooperation__genNodes

        def record(self):
            nodes = legacy_method(self)
            seen.append((nodes.dominance.flags.c_contiguous,
                         nodes.dominance.flags.f_contiguous))
            return nodes

        with patch.object(PCC, "_ParticleCompetitionAndCooperation__genNodes", record):
            result = measure(arrays, case, backend, iterations=3, early_stop=False,
                             seed=36131, steady_state=False)
        self.assertEqual(result["effective_iterations"], 3)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0], (expected_c, not expected_c))

    def test_numba_eight_or_more_classes_has_production_c_order(self):
        for classes in (8, 10, 12):
            with self.subTest(classes=classes):
                self.exercise(classes, "numba", True)

    def test_numba_few_classes_keeps_production_f_order(self):
        for classes in (2, 3, 7):
            with self.subTest(classes=classes):
                self.exercise(classes, "numba", False)

    def test_other_backends_keep_fortran_order(self):
        for backend in ("numpy", "cython"):
            for classes in (3, 10):
                with self.subTest(backend=backend, classes=classes):
                    self.exercise(classes, backend, False)


if __name__ == "__main__":
    unittest.main()
