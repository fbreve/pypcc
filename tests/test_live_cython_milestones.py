"""Synthetic within-Cython native milestone and xorshift32 parity gates.

No real datasets, no PR #18 trajectories and no production source edits.
Requires a separately compiled pcc_step_live_research extension.
"""
import unittest

import numpy as np

import pcc_step
try:
    from pcc_step_live_research import pcc_propagate as live_native
except ImportError as error:
    # The general production regression workflow intentionally does not
    # compile the independent research extension. Only the dedicated research
    # workflow has an obligation to compile and execute these tests.
    raise unittest.SkipTest("research-only Cython extension is not built") from error
from experiments.live_cython_milestones import (
    observe_cython_sequential, xorshift32_next, _record_buffers,
)
from experiments.build_live_cython import generate, ORIGINAL


MUTABLE = (
    "part_curnode", "part_strength", "dist_table", "dominance",
    "owndeg", "dom_row", "reduc", "dom_list", "dist_list", "prob", "slices",
)


def fixture(*, max_iter=31, early_stop=False, stop_max=2,
            stuck=False, seed=71, p_grd=0.5):
    nodes, classes, particles = 8, 2, 2
    neighbors = np.full((nodes, 3), -1, dtype=np.int64)
    for i in range(nodes):
        neighbors[i, :2] = [(i - 1) % nodes, (i + 1) % nodes]
    degree = np.zeros(nodes, dtype=np.int64) if stuck else np.full(nodes, 2, dtype=np.int64)
    labels = np.full(nodes, -1, dtype=np.int64)
    labels[[0, 4]] = [0, 1]
    dominance = np.full((nodes, classes), 0.5, order="F", dtype=np.float64)
    dominance[0] = [1., 0.]
    dominance[4] = [0., 1.]
    distances = np.full((nodes, particles), nodes - 1, order="F", dtype=np.uint8)
    distances[0, 0] = 0
    distances[4, 1] = 0
    return dict(
        neib_list=neighbors, neib_qt=degree, labels=labels,
        p_grd=p_grd, delta_v=0.1, c=classes,
        zerovec=np.zeros(classes),
        part_curnode=np.array([0, 4], dtype=np.int64),
        part_label=np.array([0, 1], dtype=np.int64),
        part_strength=np.ones(particles), dist_table=distances,
        dominance=dominance, owndeg=np.zeros((nodes, classes), order="F"),
        deltap=1.0, dexp=2.0,
        dom_row=np.zeros(classes), reduc=np.zeros(classes),
        dom_list=np.zeros(3), dist_list=np.zeros(3),
        prob=np.zeros(3), slices=np.zeros(3),
        dist_weights=1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** 2,
        max_iter=max_iter, early_stop=early_stop,
        es_chk=10, stop_max=stop_max, seed=seed,
    )


def state_match(tc, actual, expected):
    for key in MUTABLE:
        with tc.subTest(buffer=key):
            np.testing.assert_array_equal(actual[key], expected[key])


def return_rng_only(data, *, use_cache, canonical):
    # Independently invoke the compiled clone with ZERO checkpoints to read
    # the local terminal native RNG, not the Python or Numba RNG.
    schedule = np.empty(0, dtype=np.int64)
    arrays = {
        "positions": data["part_curnode"],
        "strengths": data["part_strength"],
        "distances": data["dist_table"],
        "dominance": data["dominance"],
        "owndeg": data["owndeg"],
        "rowmax": np.empty(data["dominance"].shape[0]),
    }
    buffers = _record_buffers(arrays, schedule)
    info = live_native(
        **data, return_info=True,
        experimental_rowmax_cache=use_cache,
        experimental_canonical_rowmax_init=canonical,
        research_observer=buffers,
    )
    assert int(buffers["recorded"][0]) == 0
    return info, int(buffers["final_rng"][0])


class TestCythonLiveMilestones(unittest.TestCase):
    def assert_exact(self, *, max_iter, early_stop, stop_max,
                     stuck, seed, milestones, use_cache, canonical=False,
                     p_grd=0.5):
        kwargs = dict(
            max_iter=max_iter, early_stop=early_stop, stop_max=stop_max,
            stuck=stuck, seed=seed, p_grd=p_grd,
        )
        original = fixture(**kwargs)
        observed = fixture(**kwargs)
        shadow = fixture(**kwargs)

        base_info = pcc_step.pcc_propagate(
            **original, return_info=True,
            experimental_rowmax_cache=use_cache,
            experimental_canonical_rowmax_init=canonical,
        )
        shadow_info, shadow_rng = return_rng_only(
            shadow, use_cache=use_cache, canonical=canonical,
        )
        trace = observe_cython_sequential(
            **observed, milestones=milestones,
            experimental_rowmax_cache=use_cache,
            experimental_canonical_rowmax_init=canonical,
        )
        state_match(self, observed, original)
        state_match(self, observed, shadow)
        self.assertEqual(dict(trace.info), base_info)
        self.assertEqual(shadow_info, base_info)
        self.assertEqual(trace.final_rng_state_uint32, shadow_rng)
        self.assertEqual(trace.next_rng_uint32, xorshift32_next(shadow_rng))
        actual = tuple(
            it for it in milestones if it <= base_info["effective_iterations"]
        )
        self.assertEqual(trace.observed_iterations, actual)
        self.assertEqual(trace.cached_rowmax_enabled, bool(use_cache and early_stop))
        for snapshot in trace.milestones:
            prefix = fixture(**{**kwargs, "max_iter":snapshot.iteration})
            shadow_prefix = fixture(**{**kwargs, "max_iter":snapshot.iteration})
            pcc_step.pcc_propagate(
                **prefix,
                experimental_rowmax_cache=use_cache,
                experimental_canonical_rowmax_init=canonical,
            )
            _, prefix_rng = return_rng_only(
                shadow_prefix, use_cache=use_cache, canonical=canonical,
            )
            for captured, name in (
                ("positions", "part_curnode"),
                ("strengths", "part_strength"),
                ("distances", "dist_table"),
                ("dominance", "dominance"),
                ("owndeg", "owndeg"),
            ):
                with self.subTest(iteration=snapshot.iteration, buffer=captured):
                    np.testing.assert_array_equal(snapshot.arrays[captured], prefix[name])
                    self.assertFalse(snapshot.arrays[captured].flags.writeable)
                    self.assertFalse(np.shares_memory(snapshot.arrays[captured], observed[name]))
            np.testing.assert_array_equal(
                snapshot.arrays["rowmax"],
                np.max(prefix["dominance"], axis=1),
            )
            self.assertEqual(snapshot.rng_state_uint32, prefix_rng)
        self.assertEqual(
            trace.bytes_copied,
            sum(x.nbytes for x in (
                original["part_curnode"], original["part_strength"],
                original["dist_table"], original["dominance"],
                original["owndeg"],
                np.empty(original["dominance"].shape[0]),
            )) * len(trace.milestones) + 4 * len(trace.milestones),
        )
        return base_info

    def test_exact_cached_default_path(self):
        self.assert_exact(
            max_iter=41, early_stop=True, stop_max=2,
            stuck=False, seed=71,
            milestones=(0, 1, 10, 11, 21, 31, 41),
            use_cache=True,
        )

    def test_exact_legacy_path_with_early_stop(self):
        self.assert_exact(
            max_iter=41, early_stop=True, stop_max=2,
            stuck=False, seed=71,
            milestones=(0, 1, 11, 21, 31, 41),
            use_cache=False,
        )

    def test_cache_canonical_initialization(self):
        self.assert_exact(
            max_iter=31, early_stop=True, stop_max=2,
            stuck=False, seed=71,
            milestones=(0, 1, 11, 21, 31),
            use_cache=True, canonical=True,
        )

    def test_stuck_early_stop_cached_and_legacy(self):
        for use_cache in (False, True):
            with self.subTest(cached=use_cache):
                info = self.assert_exact(
                    max_iter=41, early_stop=True, stop_max=1, stuck=True,
                    seed=71, milestones=(0, 1, 11, 21, 31, 41),
                    use_cache=use_cache,
                )
                self.assertEqual(info["effective_iterations"], 21)
                self.assertEqual(info["stop_reason"], "early_stop")

    def test_zero_iterations_and_seed_zero(self):
        self.assert_exact(
            max_iter=0, early_stop=True, stop_max=1, stuck=False,
            seed=0, milestones=(0,), use_cache=True,
        )

    def test_none_and_maximum_uint32_seed(self):
        for seed in (None, 0xffffffff):
            with self.subTest(seed=seed):
                self.assert_exact(
                    max_iter=12, early_stop=False, stop_max=1,
                    stuck=False, seed=seed,
                    milestones=(0, 1, 2, 12), use_cache=True,
                )

    def test_rng_advance_known_draw_count_random_walk(self):
        data = fixture(max_iter=13, early_stop=False, p_grd=0.0, seed=71)
        trace = observe_cython_sequential(
            **data, milestones=(0, 1, 2, 5, 13),
        )
        for snapshot in trace.milestones:
            expected = 71
            # Each of two active particles draws one rand_double and one
            # xorshift32 neighbour index when p_grd is exactly 0.
            for _ in range(2 * 2 * snapshot.iteration):
                expected = xorshift32_next(expected)
            self.assertEqual(snapshot.rng_state_uint32, expected)
        self.assertEqual(trace.final_rng_state_uint32, expected)

    def test_invalid_schedule_before_any_mutation(self):
        for overrides in (
            {"milestones": ()},
            {"milestones": (1, 1)},
            {"milestones": (2, 1)},
            {"milestones": (False, 1)},
            {"milestones": (0, 42)},
            {"milestones": (0, 1), "max_bytes": 1},
            {"milestones": (0,), "max_bytes": True},
            {"milestones": (0, 1), "max_observations": 1},
            {"milestones": (0,), "max_observations": 33},
        ):
            with self.subTest(options=overrides):
                data = fixture()
                state = {name: data[name].copy() for name in MUTABLE}
                py_rng = np.random.get_state()
                with self.assertRaises(ValueError):
                    observe_cython_sequential(**data, **overrides)
                state_match(self, data, state)
                after_rng = np.random.get_state()
                self.assertEqual(py_rng[0], after_rng[0])
                np.testing.assert_array_equal(py_rng[1], after_rng[1])
                self.assertEqual(py_rng[2:], after_rng[2:])

    def test_full_fit_predict_postprocessing(self):
        from pcc import ParticleCompetitionAndCooperation

        def prepare():
            data = fixture(max_iter=25)
            model = ParticleCompetitionAndCooperation(impl="cython")
            model.set_graph(data["neib_list"], data["neib_qt"], k_nn=2)
            return model, data["labels"].copy()

        normal, labels = prepare()
        baseline_labels = normal.fit_predict(
            labels, max_iter=25, early_stop=False, seed=71,
        )
        observed, copy = prepare()
        reports = []

        def experimental(*args, **kwargs):
            result = observe_cython_sequential(
                *args, **kwargs, milestones=(0, 1, 11, 25),
            )
            reports.append(result)
            return None

        observed._get_backend_fn = lambda: experimental
        observed_labels = observed.fit_predict(
            copy, max_iter=25, early_stop=False, seed=71,
        )
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].observed_iterations, (0, 1, 11, 25))
        np.testing.assert_array_equal(observed_labels, baseline_labels)
        for field in ("dominance", "label"):
            np.testing.assert_array_equal(
                getattr(normal.node, field), getattr(observed.node, field)
            )
        for field in ("homenode", "curnode", "label", "strength", "dist_table"):
            np.testing.assert_array_equal(
                getattr(normal.part, field), getattr(observed.part, field)
            )
        np.testing.assert_array_equal(observed.owndeg, normal.owndeg)


if __name__ == "__main__":
    unittest.main()
