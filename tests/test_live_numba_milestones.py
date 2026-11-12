"""Source-exact synthetic parity tests for opt-in Numba native milestones."""
import unittest

import numpy as np
from numba import njit, _helperlib

from pcc import ParticleCompetitionAndCooperation
from pcc_numba import pcc_propagate_numba
from experiments.live_numba_milestones import observe_numba_sequential


@njit
def _next_native_random():
    return np.random.random()


def case(*, max_iter=31, early_stop=False, stop_max=2, stuck=False):
    n, c, particles = 8, 2, 2
    # Undirected symmetric ring, deliberately overallocated dense neighbors.
    edges = np.full((n, 4), -1, dtype=np.int64)
    edges[:, :2] = np.array([
        [(i - 1) % n, (i + 1) % n] for i in range(n)
    ], dtype=np.int64)
    degrees = np.zeros(n, dtype=np.int64) if stuck else np.full(n, 2, dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    labels[[0, 4]] = [0, 1]
    dom = np.full((n, c), 0.5, dtype=np.float64, order="F")
    dom[0] = (1.0, 0.0)
    dom[4] = (0.0, 1.0)
    dist = np.full((n, particles), n - 1, dtype=np.uint8, order="F")
    dist[0, 0] = 0
    dist[4, 1] = 0
    return dict(
        neib_list=edges, neib_qt=degrees, labels=labels,
        p_grd=0.5, delta_v=0.1, c=c, zerovec=np.zeros(c),
        part_curnode=np.array([0, 4], dtype=np.int64),
        part_label=np.array([0, 1], dtype=np.int64),
        part_strength=np.ones(particles), dist_table=dist,
        dominance=dom, owndeg=np.zeros((n, c), dtype=np.float64, order="F"),
        deltap=1.0, dexp=2.0,
        dom_row=np.zeros(c), reduc=np.zeros(c),
        dom_list=np.zeros(4), dist_list=np.zeros(4),
        prob=np.zeros(4), slices=np.zeros(4),
        dist_weights=1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** 2,
        max_iter=max_iter, early_stop=early_stop, es_chk=10, stop_max=stop_max,
    )


MUTABLE = ("part_curnode", "part_label", "part_strength", "dist_table",
           "dominance", "owndeg", "dom_row", "reduc", "dom_list",
           "dist_list", "prob", "slices")
SNAP = {
    "positions": "part_curnode", "strengths": "part_strength",
    "distances": "dist_table", "dominance": "dominance", "owndeg": "owndeg",
}


def assert_state(test, observed, baseline):
    for name in MUTABLE:
        with test.subTest(buffer=name):
            np.testing.assert_array_equal(observed[name], baseline[name])


class NumbaLiveMilestoneTests(unittest.TestCase):
    def setUp(self):
        self.saved_numpy_rng = np.random.get_state()
        self.native_ptr = _helperlib.rnd_get_np_state_ptr()
        self.saved_native_rng = _helperlib.rnd_get_state(self.native_ptr)

    def tearDown(self):
        np.random.set_state(self.saved_numpy_rng)
        _helperlib.rnd_set_state(self.native_ptr, self.saved_native_rng)

    def check_case(self, *, max_iter, early_stop, stop_max, stuck, milestones):
        baseline = case(max_iter=max_iter, early_stop=early_stop,
                        stop_max=stop_max, stuck=stuck)
        observed = case(max_iter=max_iter, early_stop=early_stop,
                        stop_max=stop_max, stuck=stuck)
        normal_info = pcc_propagate_numba(**baseline, seed=71, return_info=True)
        after_normal = _helperlib.rnd_get_state(self.native_ptr)
        numpy_after_normal = np.random.get_state()
        next_normal = _next_native_random()

        result = observe_numba_sequential(
            **observed, seed=71, milestones=milestones,
        )
        after_observed = _helperlib.rnd_get_state(self.native_ptr)
        numpy_after_observed = np.random.get_state()
        next_observed = _next_native_random()

        self.assertEqual(dict(result.info), normal_info)
        self.assertEqual(after_normal, after_observed)
        self.assertEqual(next_normal, next_observed)
        self.assertEqual(numpy_after_normal[0], numpy_after_observed[0])
        np.testing.assert_array_equal(numpy_after_normal[1],
                                      numpy_after_observed[1])
        self.assertEqual(numpy_after_normal[2:], numpy_after_observed[2:])
        assert_state(self, observed, baseline)

        reached = tuple(i for i in milestones if
                        i <= normal_info["effective_iterations"])
        self.assertEqual(result.observed_iterations, reached)
        self.assertEqual(tuple(x.iteration for x in result.milestones), reached)

        per = sum(observed[name].nbytes for name in SNAP.values())
        self.assertEqual(result.reserved_bytes, per * len(milestones))
        self.assertEqual(result.bytes_copied, per * len(reached))
        for milestone in result.milestones:
            prefix = case(max_iter=milestone.iteration,
                          early_stop=early_stop, stop_max=stop_max,
                          stuck=stuck)
            pcc_propagate_numba(**prefix, seed=71)
            for name, source in SNAP.items():
                with self.subTest(iteration=milestone.iteration, array=name):
                    frozen = milestone.arrays[name]
                    np.testing.assert_array_equal(frozen, prefix[source])
                    self.assertFalse(frozen.flags.writeable)
                    self.assertTrue(frozen.flags.c_contiguous)
                    self.assertFalse(np.shares_memory(frozen, observed[source]))
        return normal_info

    def test_all_milestones_with_final_rng_continuation(self):
        info = self.check_case(
            max_iter=31, early_stop=False, stop_max=2, stuck=False,
            milestones=(0, 1, 2, 10, 11, 21, 31),
        )
        self.assertEqual(info["effective_iterations"], 31)
        self.assertEqual(info["stop_reason"], "max_iter")

    def test_early_stop_with_isolated_particles_preserves_stop_and_rng(self):
        info = self.check_case(
            max_iter=41, early_stop=True, stop_max=1, stuck=True,
            milestones=(0, 1, 10, 11, 21, 31, 41),
        )
        self.assertEqual(info["effective_iterations"], 21)
        self.assertEqual(info["stop_reason"], "early_stop")

    def test_zero_iterations_and_initial_state_copy(self):
        info = self.check_case(
            max_iter=0, early_stop=False, stop_max=1, stuck=False,
            milestones=(0,),
        )
        self.assertEqual(info["effective_iterations"], 0)

    def test_invalid_schedule_or_memory_budget_before_numba_seeding(self):
        bad = [
            {"milestones": ()}, {"milestones": (1, 1)},
            {"milestones": (2, 1)}, {"milestones": (False, 1)},
            {"milestones": (0, 42)}, {"milestones": (0, 1), "max_bytes": 1},
            {"milestones": (0,), "max_bytes": True},
            {"milestones": (0, 1), "max_snapshots": 1},
            {"milestones": (0,), "max_snapshots": 33},
        ]
        for params in bad:
            with self.subTest(params=params):
                source = case()
                originals = {key: source[key].copy() for key in MUTABLE}
                native_before = _helperlib.rnd_get_state(self.native_ptr)
                numpy_before = np.random.get_state()
                with self.assertRaises(ValueError):
                    observe_numba_sequential(**source, seed=71, **params)
                self.assertEqual(_helperlib.rnd_get_state(self.native_ptr),
                                 native_before)
                now = np.random.get_state()
                np.testing.assert_array_equal(now[1], numpy_before[1])
                self.assertEqual(now[0], numpy_before[0])
                self.assertEqual(now[2:], numpy_before[2:])
                assert_state(self, source, originals)

    def test_full_fit_predict_labels_postprocessing_and_rng_parity(self):
        def setup():
            data = case(max_iter=25)
            model = ParticleCompetitionAndCooperation(impl="numba")
            model.set_graph(data["neib_list"], data["neib_qt"], k_nn=2)
            return model, data["labels"].copy()

        normal, known = setup()
        normal_labels = normal.fit_predict(
            known, max_iter=25, early_stop=False, seed=71,
        )
        native_after_normal = _helperlib.rnd_get_state(self.native_ptr)
        next_normal = _next_native_random()

        observed, known2 = setup()
        records = []

        def instrumented_numba(*args, **kwargs):
            result = observe_numba_sequential(
                *args, **kwargs, milestones=(0, 1, 11, 25),
            )
            records.append(result)
            return None

        # Local override only: no global monkeypatch or production import.
        observed._get_backend_fn = lambda: instrumented_numba
        observed_labels = observed.fit_predict(
            known2, max_iter=25, early_stop=False, seed=71,
        )
        native_after_observed = _helperlib.rnd_get_state(self.native_ptr)
        next_observed = _next_native_random()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].observed_iterations, (0, 1, 11, 25))
        np.testing.assert_array_equal(observed_labels, normal_labels)
        self.assertEqual(native_after_observed, native_after_normal)
        self.assertEqual(next_observed, next_normal)
        for name in ("dominance", "label"):
            np.testing.assert_array_equal(getattr(observed.node, name),
                                          getattr(normal.node, name))
        for name in ("curnode", "homenode", "label", "strength", "dist_table"):
            np.testing.assert_array_equal(getattr(observed.part, name),
                                          getattr(normal.part, name))
        np.testing.assert_array_equal(observed.owndeg, normal.owndeg)

    def test_seed_none_preserves_inherited_native_rng_stream(self):
        # Explicitly clone Numba's own MT state rather than assuming
        # NumPy's Python generator controls native np.random calls.
        before_run = _helperlib.rnd_get_state(self.native_ptr)
        normal = case(max_iter=29)
        normal["deltap"] = 0.35
        normal_info = pcc_propagate_numba(**normal, seed=None, return_info=True)
        normal_state = _helperlib.rnd_get_state(self.native_ptr)
        normal_next = _next_native_random()

        _helperlib.rnd_set_state(self.native_ptr, before_run)
        observed = case(max_iter=29)
        observed["deltap"] = 0.35
        monitored = observe_numba_sequential(
            **observed, seed=None, milestones=(0, 1, 10, 11, 29),
        )
        observed_state = _helperlib.rnd_get_state(self.native_ptr)
        observed_next = _next_native_random()
        self.assertEqual(dict(monitored.info), normal_info)
        assert_state(self, observed, normal)
        self.assertEqual(observed_state, normal_state)
        self.assertEqual(observed_next, normal_next)

    def test_nondefault_strength_and_pure_random_movement(self):
        plain = case(max_iter=25)
        trace_data = case(max_iter=25)
        for data in (plain, trace_data):
            data["p_grd"] = 0.0
            data["deltap"] = 0.33
        baseline_info = pcc_propagate_numba(
            **plain, seed=991, return_info=True,
        )
        baseline_rng = _helperlib.rnd_get_state(self.native_ptr)
        baseline_next = _next_native_random()

        observed = observe_numba_sequential(
            **trace_data, seed=991, milestones=(0, 1, 11, 25),
        )
        self.assertEqual(dict(observed.info), baseline_info)
        assert_state(self, trace_data, plain)
        self.assertEqual(_helperlib.rnd_get_state(self.native_ptr),
                         baseline_rng)
        self.assertEqual(_next_native_random(), baseline_next)
        for boundary in observed.milestones:
            original = case(max_iter=boundary.iteration)
            original["p_grd"] = 0.0
            original["deltap"] = 0.33
            pcc_propagate_numba(**original, seed=991)
            for name, source in SNAP.items():
                np.testing.assert_array_equal(
                    boundary.arrays[name], original[source]
                )

    def test_all_snapshot_boundaries_have_independent_memory(self):
        data = case(max_iter=3)
        result = observe_numba_sequential(
            **data, seed=71, milestones=(0, 1, 2, 3)
        )
        self.assertEqual(result.observed_iterations, (0, 1, 2, 3))
        for name in SNAP:
            for before, after in zip(result.milestones,
                                     result.milestones[1:]):
                self.assertFalse(np.shares_memory(
                    before.arrays[name], after.arrays[name]
                ))


if __name__ == "__main__":
    unittest.main()
