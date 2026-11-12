"""Synthetic within-Numba live milestone parity; no external datasets."""
import unittest
import numpy as np
from numba import njit

from pcc_numba import pcc_propagate_numba, _seed_numba_rng
from experiments.live_numba_milestones import observe_numba_sequential


@njit
def _next_numba_random():
    return np.random.random()


def case(*, max_iter=31, early_stop=False, stop_max=2, stuck=False):
    n, c, p = 8, 2, 2
    neighbors = np.full((n, 3), -1, dtype=np.int64)
    neighbors[:, :2] = np.asarray(
        [[(i - 1) % n, (i + 1) % n] for i in range(n)], dtype=np.int64,
    )
    degrees = np.zeros(n, dtype=np.int64) if stuck else np.full(n, 2, dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    labels[[0, 4]] = [0, 1]
    dominance = np.full((n, c), 0.5, dtype=np.float64, order="F")
    dominance[0] = [1., 0.]
    dominance[4] = [0., 1.]
    distances = np.full((n, p), n - 1, dtype=np.uint8, order="F")
    distances[0, 0] = 0
    distances[4, 1] = 0
    return dict(
        neib_list=neighbors, neib_qt=degrees, labels=labels,
        p_grd=0.5, delta_v=0.1, c=c, zerovec=np.zeros(c),
        part_curnode=np.array([0, 4], dtype=np.int64),
        part_label=np.array([0, 1], dtype=np.int64),
        part_strength=np.ones(p), dist_table=distances,
        dominance=dominance, owndeg=np.zeros((n, c), order="F"),
        deltap=1.0, dexp=2.0,
        dom_row=np.zeros(c), reduc=np.zeros(c),
        dom_list=np.zeros(3), dist_list=np.zeros(3),
        prob=np.zeros(3), slices=np.zeros(3),
        dist_weights=1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** 2,
        max_iter=max_iter, early_stop=early_stop, es_chk=10, stop_max=stop_max,
    )


MUTABLE = ("part_curnode", "part_strength", "dist_table", "dominance",
           "owndeg", "dom_row", "reduc", "dom_list", "dist_list", "prob", "slices")


class NumbaLiveMilestonesTests(unittest.TestCase):
    def assert_same_run(self, *, max_iter, early_stop, stop_max, stuck, milestones):
        kwargs = dict(max_iter=max_iter, early_stop=early_stop,
                      stop_max=stop_max, stuck=stuck)
        baseline, observed = case(**kwargs), case(**kwargs)
        info = pcc_propagate_numba(**baseline, seed=71, return_info=True)
        baseline_next = _next_numba_random()
        trace = observe_numba_sequential(**observed, seed=71, milestones=milestones)
        observed_next = _next_numba_random()
        self.assertEqual(dict(trace.info), info)
        self.assertEqual(observed_next, baseline_next)
        for name in MUTABLE:
            with self.subTest(mutable=name):
                np.testing.assert_array_equal(observed[name], baseline[name])
        expected = tuple(i for i in milestones if i <= info["effective_iterations"])
        self.assertEqual(trace.observed_iterations, expected)
        for snap in trace.milestones:
            original = case(**{**kwargs, "max_iter":snap.iteration})
            pcc_propagate_numba(**original, seed=71)
            for captured, source in (
                ("positions", "part_curnode"),
                ("strengths", "part_strength"),
                ("distances", "dist_table"),
                ("dominance", "dominance"),
                ("owndeg", "owndeg"),
            ):
                with self.subTest(iteration=snap.iteration, array=captured):
                    np.testing.assert_array_equal(snap.arrays[captured], original[source])
                    self.assertFalse(snap.arrays[captured].flags.writeable)
                    self.assertFalse(np.shares_memory(snap.arrays[captured], observed[source]))
        self.assertEqual(trace.bytes_copied,
                         len(trace.milestones) * sum(
                             baseline[x].nbytes for x in (
                                 "part_curnode", "part_strength", "dist_table",
                                 "dominance", "owndeg",
                             )
                         ))
        return info

    def test_full_length_exact_prefix_state_and_rng(self):
        info = self.assert_same_run(
            max_iter=31, early_stop=False, stop_max=2,
            stuck=False, milestones=(0, 1, 2, 10, 11, 21, 31),
        )
        self.assertEqual(info["effective_iterations"], 31)

    def test_stuck_early_stop(self):
        info = self.assert_same_run(
            max_iter=41, early_stop=True, stop_max=1,
            stuck=True, milestones=(0, 1, 10, 11, 21, 31, 41),
        )
        self.assertEqual(info["effective_iterations"], 21)
        self.assertEqual(info["stop_reason"], "early_stop")

    def test_zero_iterations(self):
        info = self.assert_same_run(
            max_iter=0, early_stop=False, stop_max=1,
            stuck=False, milestones=(0,),
        )
        self.assertEqual(info["effective_iterations"], 0)

    def test_budget_and_invalid_schedule_before_rng_or_mutation(self):
        for options in (
            {"milestones": ()},
            {"milestones": (1, 1)},
            {"milestones": (2, 1)},
            {"milestones": (False, 1)},
            {"milestones": (0, 99)},
            {"milestones": (0, 1), "max_bytes": 1},
            {"milestones": (0,), "max_bytes": True},
            {"milestones": (0, 1), "max_observations": 1},
        ):
            with self.subTest(options=options):
                data = case()
                before = {key: data[key].copy() for key in MUTABLE}
                _seed_numba_rng(7919)
                expected = _next_numba_random()
                _seed_numba_rng(7919)
                with self.assertRaises(ValueError):
                    observe_numba_sequential(**data, seed=71, **options)
                self.assertEqual(_next_numba_random(), expected)
                for key in MUTABLE:
                    np.testing.assert_array_equal(data[key], before[key])


if __name__ == "__main__":
    unittest.main()
