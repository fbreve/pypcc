"""Synthetic-only parity checks for opt-in current-master terminal snapshots."""
import unittest

import numpy as np

from pcc import ParticleCompetitionAndCooperation
from experiments.terminal_state_probe import capture_terminal_state

try:
    import pcc_step  # noqa: F401 -- check native availability
except ImportError:
    pcc_step = None


def small_model(backend):
    # Symmetric undirected ring, deliberately dense neighbor storage.
    edges = np.array([
        [1, 5], [0, 2], [1, 3],
        [2, 4], [3, 5], [0, 4],
    ], dtype=np.int64)
    degrees = np.full(6, 2, dtype=np.int64)
    model = ParticleCompetitionAndCooperation(impl=backend)
    model.set_graph(edges, degrees, k_nn=2)
    return model


def run(backend, iterations=13):
    model = small_model(backend)
    labels = np.array([0, -1, -1, 1, -1, -1], dtype=np.int64)
    prediction = model.fit_predict(
        labels, p_grd=0.5, delta_v=0.1, max_iter=iterations,
        early_stop=False, seed=71
    )
    return model, prediction


class TerminalStateProbeTests(unittest.TestCase):
    def backends(self):
        return ("numpy", "numba") + (("cython",) if pcc_step is not None else ())

    def test_terminal_probe_exactly_matches_current_model_and_is_owned(self):
        previous_rng = np.random.get_state()
        try:
            for backend in self.backends():
                with self.subTest(backend=backend):
                    model, prediction = run(backend)
                    state = capture_terminal_state(model, include_distances=True)
                    expected = {
                        "dominance": model.node.dominance,
                        "node_classes": model.node.label,
                        "particle_positions": model.part.curnode,
                        "particle_strengths": model.part.strength,
                        "particle_homes": model.part.homenode,
                        "particle_classes": model.part.label,
                        "mapped_labels": model.mapped_labels,
                        "normalized_owndeg": model.owndeg,
                        "particle_distances": model.part.dist_table,
                    }
                    self.assertFalse(state.actual_iterations_known)
                    self.assertEqual(state.iteration_cap, 13)
                    self.assertEqual(state.total_bytes, sum(a.nbytes for a in expected.values()))
                    self.assertEqual(set(state.arrays), set(expected))
                    for name, source in expected.items():
                        snap = state.arrays[name]
                        np.testing.assert_array_equal(snap, source)
                        self.assertFalse(np.shares_memory(snap, source))
                        self.assertFalse(snap.flags.writeable)
                        self.assertTrue(snap.flags.c_contiguous)
                        self.assertEqual(len(state.sha256[name]), 64)
                    np.testing.assert_array_equal(
                        model.unique_labels[state.arrays["node_classes"]],
                        prediction
                    )
                    before_hashes = dict(state.sha256)
                    # Original arrays can change without changing the copied snapshot.
                    old = model.part.strength[0]
                    model.part.strength[0] = old * 0.9
                    self.assertEqual(before_hashes, dict(state.sha256))
                    self.assertEqual(state.arrays["particle_strengths"][0], old)
        finally:
            np.random.set_state(previous_rng)

    def test_replay_same_backend_exact_final_state(self):
        previous_rng = np.random.get_state()
        try:
            for backend in self.backends():
                with self.subTest(backend=backend):
                    first_model, first_prediction = run(backend)
                    first = capture_terminal_state(first_model)
                    second_model, second_prediction = run(backend)
                    second = capture_terminal_state(second_model)
                    np.testing.assert_array_equal(first_prediction, second_prediction)
                    self.assertEqual(dict(first.sha256), dict(second.sha256))
                    self.assertNotIn("particle_distances", first.arrays)
        finally:
            np.random.set_state(previous_rng)

    def test_rng_unchanged_by_snapshot_not_by_fit(self):
        previous_rng = np.random.get_state()
        try:
            for backend in self.backends():
                with self.subTest(backend=backend):
                    model, _ = run(backend)
                    before = np.random.get_state()
                    snap = capture_terminal_state(model, include_distances=True)
                    after = np.random.get_state()
                    self.assertEqual(before[0], after[0])
                    np.testing.assert_array_equal(before[1], after[1])
                    self.assertEqual(before[2:], after[2:])
                    self.assertTrue(snap.arrays)
        finally:
            np.random.set_state(previous_rng)

    def test_invalid_and_unfitted_rejected_without_mutation(self):
        model = small_model("numpy")
        with self.assertRaisesRegex(ValueError, "completed fitted state"):
            capture_terminal_state(model)
        model, _ = run("numpy", iterations=0)
        zero = capture_terminal_state(model, max_bytes=4096)
        self.assertEqual(zero.iteration_cap, 0)
        self.assertFalse(zero.actual_iterations_known)
        required = zero.total_bytes
        with self.assertRaisesRegex(ValueError, "exceeds max_bytes"):
            capture_terminal_state(model, max_bytes=required-1)
        for invalid in (0, -1, True, 0.5, "4096"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                capture_terminal_state(model, max_bytes=invalid)
        with self.assertRaises(ValueError):
            capture_terminal_state(model, include_distances=1)


if __name__ == "__main__":
    unittest.main()
