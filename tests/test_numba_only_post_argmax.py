"""Regression: only the actual Numba propagation dispatcher uses full-row argmax.

The experimental CPU/bitwise A/B proof is archived in scientific #227.
This permanent test exercises native backend selection, including fallbacks.
"""
import unittest
from unittest.mock import patch

import numpy as np

import pcc
from pcc import ParticleCompetitionAndCooperation as PCC


def _ring(n):
    ids = np.arange(n, dtype=np.int64)
    return np.column_stack(((ids - 1) % n, (ids + 1) % n))


class NumbaOnlyPostArgmaxTests(unittest.TestCase):
    def _fit_and_check(self, impl, expected_full):
        n, classes = 256, 12
        labels = np.full(n, -1, dtype=np.int64)
        labels[:3 * classes] = np.tile(np.arange(classes, dtype=np.int64), 3)
        obj = PCC(impl=impl)
        obj.set_graph(_ring(n), np.full(n, 2, dtype=np.int64), k_nn=2)

        # Capture only the actual post-propagation np.argmax call. The
        # scientific Numba candidate scans all n rows; the other backends
        # must continue scanning only previously unlabeled rows.
        original_argmax = np.argmax
        seen = []

        def spy(a, *args, **kwargs):
            arr = np.asarray(a)
            if arr.ndim == 2 and arr.shape[1] == classes and (
                    (len(args) > 0 and args[0] == 1) or kwargs.get("axis") == 1):
                seen.append(arr.shape)
            return original_argmax(a, *args, **kwargs)

        with patch.object(pcc.np, "argmax", side_effect=spy):
            prediction = obj.fit_predict(
                labels, max_iter=3, early_stop=False, seed=27913
            )

        self.assertEqual(len(seen), 1, seen)
        self.assertEqual(
            seen[0], (n if expected_full else n - 3 * classes, classes)
        )
        self.assertTrue(np.array_equal(prediction[:3 * classes], labels[:3 * classes]))
        self.assertTrue(obj.node.dominance.flags.c_contiguous == expected_full)
        self.assertFalse(np.shares_memory(prediction, obj.node.label))
        self.assertTrue(np.all((obj.node.label >= 0) & (obj.node.label < classes)))

        # Replay remains exact with a fixed native seed and the same graph.
        first = (prediction.copy(), obj.node.label.copy(),
                 obj.node.dominance.copy(), obj.owndeg.copy())
        second = obj.fit_predict(labels, max_iter=3, early_stop=False, seed=27913)
        for x, y in zip(first, (second, obj.node.label, obj.node.dominance, obj.owndeg)):
            np.testing.assert_array_equal(x, y)

    @unittest.skipUnless(pcc._HAS_NUMBA, "Numba unavailable")
    def test_actual_numba_dispatcher_uses_full_rows(self):
        from pcc_numba import pcc_propagate_numba
        self.assertIs(pcc.pcc_propagate_numba, pcc_propagate_numba)
        self.assertIs(PCC(impl="numba")._get_backend_fn(), pcc_propagate_numba)
        self._fit_and_check("numba", True)

    def test_numpy_preserves_masked_rows(self):
        self._fit_and_check("numpy", False)

    @unittest.skipUnless(pcc._HAS_CYTHON, "Cython unavailable")
    def test_cython_preserves_masked_rows(self):
        from pcc_step import pcc_propagate
        self.assertIs(PCC(impl="cython")._get_backend_fn(), pcc_propagate)
        self._fit_and_check("cython", False)

    @unittest.skipUnless(pcc._HAS_CYTHON, "Cython unavailable")
    def test_numba_requested_but_unavailable_preserves_cython_path(self):
        # The optimization must follow the true native dispatcher, not impl.
        with patch.object(pcc, "_HAS_NUMBA", False):
            self.assertIs(PCC(impl="numba")._get_backend_fn(),
                          PCC(impl="cython")._get_backend_fn())
            self._fit_and_check("numba", False)


if __name__ == "__main__":
    unittest.main()
