"""Cross-backend numerical parity of dominance transfer and particle strength."""
import unittest
import numpy as np
from test_sequential_safety import call_step, pcc_step


class DominanceStrengthNumericParityTests(unittest.TestCase):
    def backends(self):
        return ['numpy', 'numba'] + (['cython'] if pcc_step is not None else [])

    def run_case(self, row, strength, delta_v, deltap, labeled):
        c = len(row)
        edges = np.array([[1], [0]], dtype=np.int64)
        degrees = np.array([1, 0], dtype=np.int64)
        labels = np.array([-1, 0 if labeled else -1], dtype=np.int64)
        base = np.vstack([np.full(c, 1. / c), np.asarray(row, dtype=np.float64)])
        outputs = {}
        for backend in self.backends():
            dominance = base.copy()
            positions = np.array([0], dtype=np.int64)
            strengths = np.array([strength], dtype=np.float64)
            distances = np.array([[0], [255]], dtype=np.uint8)
            ownership = np.zeros((2, c), dtype=np.float64)
            args = [edges.copy(), degrees.copy(), labels.copy(), 0., delta_v,
                    c, np.zeros(c), positions, np.array([0], dtype=np.int64),
                    strengths, distances, dominance, ownership, deltap, 2.]
            call_step(backend, args)
            outputs[backend] = (dominance, strengths, distances, ownership, positions)
        return outputs

    def test_extreme_valid_dominance_transfers_match(self):
        cases = [
            ([.5, .5], 1., .1, 1.),
            ([.999999999999, 1e-12], .999999999999, .5, .4),
            ([1e-250, 1. - 1e-250], 1e-250, 1e-200, .4),
            ([.2, .3, .5], .6, .7, 1.),
            ([1e-15, .4, .6 - 1e-15], .99, 1., .25),
            ([.1, .2, .3, .4], .8, .8, .6),
            ([.0, .0, .0, 1.], 1., 1., 1.),
        ]
        for row, strength, delta_v, deltap in cases:
            for labeled in (False, True):
                with self.subTest(row=row, strength=strength, delta_v=delta_v,
                                  deltap=deltap, labeled=labeled):
                    results = self.run_case(row, strength, delta_v, deltap, labeled)
                    expected = results['numpy']
                    for backend, actual in results.items():
                        for a, e in zip(actual, expected):
                            if a.dtype.kind == 'f':
                                np.testing.assert_allclose(a, e, rtol=1e-12,
                                                           atol=1e-14,
                                                           err_msg=backend)
                            else:
                                np.testing.assert_array_equal(a, e,
                                                              err_msg=backend)
                    if not labeled:
                        self.assertAlmostEqual(float(expected[0][1].sum()),
                                               float(np.sum(row)), places=12)


if __name__ == '__main__':
    unittest.main()
