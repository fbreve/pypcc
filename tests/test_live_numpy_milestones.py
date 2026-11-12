"""Synthetic parity gates for separate opt-in NumPy mid-fit milestones.

Tests compare FULL mutable state, stop diagnostics and next RNG draw against
current production NumPy propagation with the same seed, never cross backend.
"""
import unittest

import numpy as np

from pcc_numpy import pcc_propagate_numpy
from experiments.live_numpy_milestones import observe_numpy_sequential


def case(*, max_iter=31, early_stop=False, stop_max=2, stuck=False):
    n, c, p = 8, 2, 2
    # Symmetric, undirected ring; intentionally padded DENSE neighbors.
    edges = np.zeros((n, 3), dtype=np.int64)
    edges[:, :2] = np.array([
        [(i - 1) % n, (i + 1) % n] for i in range(n)
    ], dtype=np.int64)
    degrees = np.zeros(n, dtype=np.int64) if stuck else np.full(n, 2, dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    labels[[0, 4]] = [0, 1]
    dominance = np.full((n, c), 0.5, dtype=np.float64, order="F")
    dominance[0] = [1., 0.]
    dominance[4] = [0., 1.]
    distances = np.full((n, p), n - 1, dtype=np.uint8, order="F")
    distances[0, 0] = 0
    distances[4, 1] = 0
    k = edges.shape[1]
    return dict(
        neib_list=edges, neib_qt=degrees, labels=labels,
        p_grd=0.5, delta_v=0.1, c=c, zerovec=np.zeros(c),
        part_curnode=np.array([0, 4], dtype=np.int64),
        part_label=np.array([0, 1], dtype=np.int64),
        part_strength=np.ones(p), dist_table=distances,
        dominance=dominance, owndeg=np.zeros((n, c), order="F"),
        deltap=1.0, dexp=2.0,
        dom_row=np.zeros(c), reduc=np.zeros(c),
        dom_list=np.zeros(k), dist_list=np.zeros(k),
        prob=np.zeros(k), slices=np.zeros(k),
        dist_weights=1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** 2,
        max_iter=max_iter, early_stop=early_stop, es_chk=10, stop_max=stop_max,
    )


MUTABLE = ("part_curnode", "part_label", "part_strength", "dist_table",
           "dominance", "owndeg", "dom_row", "reduc", "dom_list",
           "dist_list", "prob", "slices")


def state_copy(data):
    return {name: data[name].copy() for name in MUTABLE}


def check_state(testcase, actual, expected):
    for name in MUTABLE:
        with testcase.subTest(array=name):
            np.testing.assert_array_equal(actual[name], expected[name])


def check_rng(testcase, actual, expected):
    testcase.assertEqual(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    testcase.assertEqual(actual[2:], expected[2:])


class NumPyLiveMilestoneTests(unittest.TestCase):
    def setUp(self):
        self.saved_rng = np.random.get_state()

    def tearDown(self):
        np.random.set_state(self.saved_rng)

    def assert_same_run(self, *, max_iter, early_stop, stop_max, stuck, milestones):
        baseline = case(max_iter=max_iter, early_stop=early_stop,
                        stop_max=stop_max, stuck=stuck)
        observed = case(max_iter=max_iter, early_stop=early_stop,
                        stop_max=stop_max, stuck=stuck)
        info = pcc_propagate_numpy(**baseline, seed=71, return_info=True)
        after_baseline = np.random.get_state()
        # One extra random draw catches hidden overconsumption as well.
        next_baseline = np.random.random()
        trace = observe_numpy_sequential(**observed, seed=71,
                                         milestones=milestones)
        after_observed = np.random.get_state()
        next_observed = np.random.random()
        self.assertEqual(dict(trace.info), info)
        self.assertEqual(trace.info["effective_iterations"],
                         info["effective_iterations"])
        self.assertEqual(trace.observed_iterations,
                         tuple(i for i in milestones if
                               i <= info["effective_iterations"]))
        check_state(self, observed, baseline)
        check_rng(self, after_observed, after_baseline)
        self.assertEqual(next_observed, next_baseline)
        for snap in trace.milestones:
            self.assertTrue(all(not x.flags.writeable
                                for x in snap.arrays.values()))
            self.assertTrue(all(x.flags.c_contiguous
                                for x in snap.arrays.values()))
            for key, target in (
                ("positions", "part_curnode"), ("strengths", "part_strength"),
                ("distances", "dist_table"), ("dominance", "dominance"),
                ("owndeg", "owndeg"),
            ):
                self.assertFalse(np.shares_memory(snap.arrays[key],
                                                  observed[target]))
            # Each milestone reproduces an independent ordinary propagation
            # prefix from original initial arrays/seed, not a saved/replayed
            # partial state or an independently implemented particle kernel.
            original = case(max_iter=snap.iteration, early_stop=early_stop,
                            stop_max=stop_max, stuck=stuck)
            pcc_propagate_numpy(**original, seed=71)
            for capture, name in (
                ("positions", "part_curnode"), ("strengths", "part_strength"),
                ("distances", "dist_table"), ("dominance", "dominance"),
                ("owndeg", "owndeg"),
            ):
                with self.subTest(boundary=snap.iteration, array=capture):
                    np.testing.assert_array_equal(snap.arrays[capture],
                                                  original[name])
        self.assertEqual(trace.bytes_copied,
                         sum(baseline[name].nbytes for name in
                             ("part_curnode", "part_strength", "dist_table",
                              "dominance", "owndeg")) * len(trace.milestones))
        return info

    def test_full_length_observations_match_independent_prefix_and_rng(self):
        info = self.assert_same_run(
            max_iter=31, early_stop=False, stop_max=2, stuck=False,
            milestones=(0, 1, 2, 10, 11, 21, 31),
        )
        self.assertEqual(info["effective_iterations"], 31)
        self.assertEqual(info["stop_reason"], "max_iter")

    def test_stuck_particle_early_stop_is_identical(self):
        info = self.assert_same_run(
            max_iter=41, early_stop=True, stop_max=1, stuck=True,
            milestones=(0, 1, 10, 11, 21, 31, 41),
        )
        self.assertEqual(info["effective_iterations"], 21)
        self.assertEqual(info["stop_reason"], "early_stop")

    def test_zero_iterations_and_explicit_initial_snapshot(self):
        info = self.assert_same_run(
            max_iter=0, early_stop=False, stop_max=1, stuck=False,
            milestones=(0,),
        )
        self.assertEqual(info["effective_iterations"], 0)

    def test_full_fit_predict_postprocessing_and_rng_unchanged(self):
        # Research injection is entirely local to this synthetic test:
        # real model initialization, full propagation and post-fit label
        # conversion/owndeg normalization must still match normal fit_predict.
        from pcc import ParticleCompetitionAndCooperation

        def prepare():
            data = case(max_iter=25)
            model = ParticleCompetitionAndCooperation(impl="numpy")
            model.set_graph(data["neib_list"], data["neib_qt"], k_nn=2)
            return model, data["labels"].copy()

        original, labels = prepare()
        normal_predictions = original.fit_predict(
            labels, max_iter=25, early_stop=False, seed=71,
        )
        after_baseline = np.random.get_state()
        next_baseline = np.random.random()

        observed, labels2 = prepare()
        reports = []

        def experimental_numpy(*args, **kwargs):
            report = observe_numpy_sequential(
                *args, **kwargs, milestones=(0, 1, 11, 25),
            )
            reports.append(report)
            # The production fit call does not consume the propagator return.
            return None

        # Overrides instance dispatch in this test only, not any library module.
        observed._get_backend_fn = lambda: experimental_numpy
        observed_predictions = observed.fit_predict(
            labels2, max_iter=25, early_stop=False, seed=71,
        )
        after_observed = np.random.get_state()
        next_observed = np.random.random()

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].observed_iterations, (0, 1, 11, 25))
        np.testing.assert_array_equal(observed_predictions, normal_predictions)
        for field in ("dominance", "label"):
            np.testing.assert_array_equal(
                getattr(observed.node, field), getattr(original.node, field)
            )
        for field in ("curnode", "label", "strength", "dist_table", "homenode"):
            np.testing.assert_array_equal(
                getattr(observed.part, field), getattr(original.part, field)
            )
        np.testing.assert_array_equal(observed.owndeg, original.owndeg)
        check_rng(self, after_observed, after_baseline)
        self.assertEqual(next_observed, next_baseline)

    def test_invalid_schedule_and_budget_rejected_before_any_rng_draw(self):
        examples = [
            ({"milestones": ()}, ValueError),
            ({"milestones": (1, 1)}, ValueError),
            ({"milestones": (2, 1)}, ValueError),
            ({"milestones": (False, 1)}, ValueError),
            ({"milestones": (0, 42)}, ValueError),
            ({"milestones": (0, 1), "max_bytes": 1}, ValueError),
            ({"milestones": (0,), "max_bytes": True}, ValueError),
            ({"milestones": (0, 1), "max_observations": 1}, ValueError),
            ({"milestones": (0,), "max_observations": 33}, ValueError),
        ]
        for overrides, error in examples:
            with self.subTest(options=overrides):
                data = case()
                before = state_copy(data)
                np.random.seed(7919)
                rng_before = np.random.get_state()
                with self.assertRaises(error):
                    observe_numpy_sequential(**data, seed=71, **overrides)
                check_rng(self, np.random.get_state(), rng_before)
                check_state(self, data, before)

    def test_capture_arrays_are_independent_at_each_boundary(self):
        data = case(max_iter=3)
        trace = observe_numpy_sequential(
            **data, seed=71, milestones=(0, 1, 2, 3)
        )
        self.assertEqual(trace.observed_iterations, (0, 1, 2, 3))
        # Even when zero nodes move, each observation has independent storage.
        for name in trace.milestones[0].arrays:
            for a, b in zip(trace.milestones, trace.milestones[1:]):
                self.assertFalse(np.shares_memory(a.arrays[name], b.arrays[name]))


if __name__ == "__main__":
    unittest.main()
