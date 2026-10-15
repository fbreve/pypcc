"""Exact within-backend parity, never cross-backend bitwise equivalence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numba import _pcc_propagate_numba_sequential, pcc_propagate_numba
from experiments.convergence_trajectories import Trajectory, array_hash, observe_fit, metadata, write_output, read_output
from experiments.numba_trajectories import allocate_buffer, propagate_observed, record_native, seed_numba
from tests.test_sequential_safety import state, buffers, next_random


def strip_timing(rows):
    return [{k: v for k, v in row.items() if k != 'elapsed_instrumented_s'} for row in rows]


class NativeTrajectoryTests(unittest.TestCase):
    def test_snapshots_match_unobserved_prefix_states(self):
        schedule = [0, 1, 3, 11, 27]
        args = state()
        args[3] = .5
        args += buffers(args) + [27, False, 2000, 1]
        trace = Trajectory(args[2], milestones=tuple(schedule))
        trace.execution = {}
        buffer, _ = allocate_buffer(args, schedule, trace)
        seed_numba(19)
        propagate_observed(args, buffer, trace)
        # Independent prefixes are test oracles only; collection uses one loop.
        for row in trace.rows:
            prefix = state()
            prefix[3] = .5
            seed_numba(19)
            pcc_propagate_numba(*prefix, *buffers(prefix), row['iteration'], False, 2000, 1)
            for name, index in dict(dominance=11, positions=7, strengths=9, distances=10, owndeg=12).items():
                self.assertEqual(row['state_hashes'][name], array_hash(prefix[index]))

    def test_native_loop_parity_stopping_state_and_rng(self):
        for cap, early, plateau in ((50, False, False), (100, True, True),
                                    (12, True, False), (21, True, True),
                                    (0, True, True), (1, False, False)):
            for seed in (10, 19):
                outputs = []
                for observed in (False, True, True):
                    args = state()
                    args[3] = .5
                    if plateau:
                        args[1][:] = 0
                    args += buffers(args) + [cap, early, 2000, 1]
                    schedule = sorted({i for i in (0, 2, 7, cap) if i <= cap})
                    trace = Trajectory(args[2], milestones=tuple(schedule))
                    trace.execution = {}  # low-level test; not exportable provenance
                    recording, _ = allocate_buffer(args, schedule, trace)
                    seed_numba(seed)
                    np.random.seed(123)
                    if observed:
                        info = propagate_observed(args, recording, trace)
                    else:
                        info = pcc_propagate_numba(*args, return_info=True)
                    hashes = {i: array_hash(args[i]) for i in (7, 9, 10, 11, 12)}
                    continuation = [next_random() for _ in range(10)]
                    outputs.append((info, hashes, continuation, np.random.random(), strip_timing(trace.rows)))
                    if observed:
                        self.assertEqual(trace.rows[-1]['iteration'], info['effective_iterations'])
                        for name, index in dict(dominance=11, positions=7, strengths=9, distances=10, owndeg=12).items():
                            self.assertEqual(trace.rows[-1]['state_hashes'][name], hashes[index])
                        self.assertNotIn('numpy_rng', trace.rows[-1]['state_hashes'])
                    np.testing.assert_allclose(args[11].sum(axis=1), 1., rtol=0, atol=1e-14)
                    self.assertTrue(np.isfinite(args[11]).all())
                    self.assertTrue(((args[7] >= 0) & (args[7] < args[11].shape[0])).all())
                    self.assertEqual(args[10].dtype, np.uint8)
                self.assertEqual(outputs[0][:4], outputs[1][:4])
                self.assertEqual(outputs[1], outputs[2])
        for compiled in (_pcc_propagate_numba_sequential, record_native):
            self.assertTrue(compiled.nopython_signatures)
            self.assertTrue(all(not result.objectmode for result in compiled.overloads.values()))

    def test_public_fit_branching_multiclass_predicates_and_bundle(self):
        # Three classes, several particles, nontrivial topology and non-default strength.
        edges = np.array([[(i - 1) % 9, (i + 1) % 9, (i + 3) % 9] for i in range(9)])
        labels = np.array([10, -1, 20, -1, 30, -1, 10, -1, 30])
        truth = np.array([10, 10, 20, 20, 30, 30, 10, 20, 30])
        for seed in (10, 19):
            results = []
            for mode in ('plain', 'observed', 'repeat', 'changed-truth'):
                model = PCC(impl='numba')
                model.set_graph(edges, np.full(9, 3))
                seed_numba(seed)
                if mode == 'plain':
                    metrics = {}
                    def plain(*args):
                        metrics.update(pcc_propagate_numba(*args, return_info=True))
                    model._get_backend_fn = lambda: plain
                    predictions = model.fit_predict(labels, max_iter=80, early_stop=True, deltap=.7, dexp=1.5)
                    trace_rows = None
                else:
                    predictions, trace, metrics = observe_fit(model, labels,
                        truth=truth if mode != 'changed-truth' else truth[::-1],
                        propagation_seed=seed, milestones=(0, 3, 7, 60), include_checks=True,
                        max_iter=80, early_stop=True, deltap=.7, dexp=1.5)
                    trace_rows = strip_timing(trace.rows)
                    self.assertIs(model._get_backend_fn(), pcc_propagate_numba)
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / 'trace'
                        write_output(path, trace, metadata(dataset='synthetic-nine', label_seed=0, collector=trace))
                        bundle = read_output(path)
                        self.assertEqual(bundle['observations'], trace.rows)
                        self.assertEqual(bundle['metadata']['seed_policy'], 'numba-seed-once')
                known = labels != -1
                np.testing.assert_array_equal(predictions[known], labels[known])
                np.testing.assert_array_equal(model.node.dominance[known], np.eye(3)[model.mapped_labels[known]])
                np.testing.assert_array_equal(model.part.dist_table[model.part.homenode, np.arange(model.part.amount)], 0)
                hashes = [array_hash(a) for a in (predictions, model.node.dominance,
                    model.part.curnode, model.part.strength, model.part.dist_table, model.owndeg)]
                results.append((hashes, metrics, [next_random() for _ in range(10)], trace_rows))
            for result in results[1:]:
                self.assertEqual(results[0][:3], result[:3])
            self.assertEqual(results[1], results[2])
            self.assertEqual([r['state_hashes'] for r in results[1][3]], [r['state_hashes'] for r in results[3][3]])

    def test_caller_rng_is_not_reseeded_or_claimed(self):
        def run(observed):
            model = PCC(impl='numba')
            model.set_graph(np.array([[1, 2], [0, 2], [0, 1]]), np.full(3, 2))
            seed_numba(19)
            next_random()  # caller consumes one draw before the fit
            if observed:
                p, trace, _ = observe_fit(model, [0, -1, 1], max_iter=20)
                self.assertIsNone(trace.execution['propagation_seed'])
                with self.assertRaises(ValueError):
                    metadata(dataset='tiny', label_seed=0, propagation_seed=19, collector=trace)
            else:
                p = model.fit_predict([0, -1, 1], max_iter=20)
            return p.tolist(), array_hash(model.node.dominance), next_random()
        self.assertEqual(run(False), run(True))


if __name__ == '__main__':
    unittest.main()
