"""Cython single-run snapshots and exact native RNG continuation parity."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pcc_step

from pcc import ParticleCompetitionAndCooperation as PCC
from experiments.convergence_trajectories import Trajectory, array_hash, observe_fit, metadata, write_output, read_output
from experiments.native_trajectories import allocate_buffer, propagate_cython
from experiments.trajectory_io import validate_payload
from tests.test_sequential_safety import state, buffers
from tests.test_numba_trajectories import strip_timing


class CythonTrajectoryTests(unittest.TestCase):
    def test_exact_state_stop_rng_and_trace_parity(self):
        cases = [(0, False, False), (1, False, False), (50, False, False),
                 (12, True, False), (21, True, True), (100, True, True)]
        for cap, early, plateau in cases:
            outputs = []
            for mode in ('public', 'probe', 'observed', 'repeat'):
                args = state()
                args[3] = .5
                if plateau:
                    args[1][:] = 0
                args += buffers(args) + [cap, early, 2000, 1]
                schedule = sorted({i for i in (0, 1, 3, 11, 21, cap) if i <= cap})
                trace = Trajectory(args[2], milestones=tuple(schedule))
                trace.execution = {}
                recording, _ = allocate_buffer(args, schedule, trace, cython=True)
                probe = np.empty(11, dtype=np.uint32)
                np.random.seed(123)
                if mode == 'public':
                    info = pcc_step.pcc_propagate(*args, return_info=True)
                else:
                    native = pcc_step._TrajectoryBuffer(recording) if mode in ('observed', 'repeat') else None
                    info = pcc_step._pcc_propagate_impl(*args, return_info=True,
                        _observations=native, _rng_probe=probe)
                    if native is not None:
                        from experiments.native_trajectories import finish_trace
                        finish_trace(recording, trace, info, 0., cython=True)
                        self.assertEqual(trace.rows[-1]['state_hashes']['cython_rng'], array_hash(probe[:1]))
                hashes = [array_hash(args[i]) for i in (7, 9, 10, 11, 12)]
                hashes.append(array_hash(np.argmax(args[11], axis=1)))
                outputs.append((info, hashes, np.random.random(),
                                probe.tolist() if mode != 'public' else None, strip_timing(trace.rows)))
                if mode in ('observed', 'repeat'):
                    observed_iterations = [r['iteration'] for r in trace.rows]
                    self.assertEqual(len(observed_iterations), len(set(observed_iterations)))
                    self.assertEqual(sum(r['final'] for r in trace.rows), 1)
                    self.assertEqual(observed_iterations[-1], info['effective_iterations'])
                    if cap in (21, 100):
                        self.assertEqual(observed_iterations, [0, 1, 3, 11, 21])
                        self.assertEqual(info['stop_reason'], 'early_stop')
                np.testing.assert_allclose(args[11].sum(axis=1), 1., rtol=0., atol=1e-14)
                self.assertTrue(np.isfinite(args[11]).all())
            for output in outputs[1:]:
                self.assertEqual(outputs[0][:3], output[:3])
            self.assertEqual(outputs[1][:4], outputs[2][:4])
            self.assertEqual(outputs[2], outputs[3])
        # Independent oracle for the COPY-only xorshift continuation probe.
        for before, after in zip(probe, probe[1:]):
            value = int(before)
            value ^= (value << 13) & 0xffffffff
            value ^= value >> 17
            value ^= (value << 5) & 0xffffffff
            self.assertEqual(value, int(after))

    def test_snapshots_match_unobserved_prefixes_including_rng(self):
        schedule = [0, 1, 3, 11, 27]
        args = state()
        args[3] = .5
        args += buffers(args) + [27, False, 2000, 1]
        trace = Trajectory(args[2], milestones=tuple(schedule))
        trace.execution = {}
        buffer, _ = allocate_buffer(args, schedule, trace, cython=True)
        propagate_cython(args, buffer, trace)
        for row in trace.rows:
            prefix = state()
            prefix[3] = .5
            probe = np.empty(11, dtype=np.uint32)
            pcc_step._pcc_propagate_impl(*prefix, *buffers(prefix), row['iteration'], False,
                                       2000, 1, _rng_probe=probe)
            for name, index in dict(dominance=11, positions=7, strengths=9, distances=10, owndeg=12).items():
                self.assertEqual(row['state_hashes'][name], array_hash(prefix[index]))
            self.assertEqual(row['state_hashes']['cython_rng'], array_hash(probe[:1]))

    def test_single_slot_capacity_full_dedup_and_rejected_buffers(self):
        for cap, schedule in ((0, [0]), (5, [5]), (5, [0, 1, 2, 3, 4, 5])):
            args = state()
            args += buffers(args) + [cap, False, 2000, 1]
            trace = Trajectory(args[2], milestones=tuple(schedule), max_observations=len(schedule))
            trace.execution = {}
            buffer, required = allocate_buffer(args, schedule, trace, cython=True)
            trace.max_bytes = trace.estimated_bytes(len(schedule), args[5], required)
            propagate_cython(args, buffer, trace)
            self.assertEqual([r['iteration'] for r in trace.rows], schedule)
            self.assertEqual(int(buffer[7][0]), len(schedule))
            self.assertEqual(sum(r['final'] for r in trace.rows), 1)
            with self.assertRaises(ValueError):
                propagate_cython(args, buffer, trace)  # cannot reuse terminal buffers
            trace.max_bytes -= 1
            with self.assertRaises(ValueError):
                allocate_buffer(args, schedule, trace, cython=True)
        args = state()
        args += buffers(args) + [3, False, 2000, 1]
        trace = Trajectory(args[2])
        for malformed in ([], [0, 1], [0, 3, 3], [3, 0], [-1, 3], [0, 4], [0., 3]):
            with self.subTest(schedule=malformed), self.assertRaises(ValueError):
                allocate_buffer(args, malformed, trace, cython=True)
        trace.max_observations = 1
        with self.assertRaises(ValueError):
            allocate_buffer(args, [0, 3], trace, cython=True)
        trace.max_observations = 128
        good, _ = allocate_buffer(args, [0, 3], trace, cython=True)
        for index in (1, 2, 3, 4, 5, 6, 8):
            short = list(good)
            short[index] = short[index][:-1]
            hashes = [array_hash(args[i]) for i in (7, 9, 10, 11, 12)]
            with self.assertRaises(ValueError):
                pcc_step._pcc_propagate_impl(*args, _observations=pcc_step._TrajectoryBuffer(tuple(short)))
            self.assertEqual(hashes, [array_hash(args[i]) for i in (7, 9, 10, 11, 12)])
        with self.assertRaises(ValueError):
            pcc_step._pcc_propagate_impl(*args, _rng_probe=np.empty(10, dtype=np.uint32))
        oversized = list(args)
        oversized[11] = np.full((3, 3), .5)
        with self.assertRaises(ValueError):
            pcc_step._pcc_propagate_impl(*oversized, _observations=pcc_step._TrajectoryBuffer(good))

    def test_multiclass_fit_bundle_truth_and_fixed_rng_policy(self):
        edges = np.array([[(i - 1) % 9, (i + 1) % 9, (i + 3) % 9] for i in range(9)])
        labels = np.array([10, -1, 20, -1, 30, -1, 10, -1, 30])
        truth = np.array([10, 10, 20, 20, 30, 30, 10, 20, 30])
        outputs = []
        for mode in ('plain', 'observed', 'repeat', 'changed-truth'):
            model = PCC(impl='cython')
            model.set_graph(edges, np.full(9, 3))
            info = {}
            if mode == 'plain':
                def ordinary(*args):
                    info.update(pcc_step.pcc_propagate(*args, return_info=True))
                model._get_backend_fn = lambda: ordinary
                predictions = model.fit_predict(labels, max_iter=80, deltap=.7, dexp=1.5)
                rows = None
            else:
                predictions, trace, info = observe_fit(model, labels,
                    truth=truth if mode != 'changed-truth' else truth[::-1],
                    milestones=(0, 3, 7, 60), max_iter=80, deltap=.7, dexp=1.5)
                rows = strip_timing(trace.rows)
                np.testing.assert_array_equal(predictions[labels != -1], labels[labels != -1])
                np.testing.assert_array_equal(model.node.dominance[labels != -1], np.eye(3)[model.mapped_labels[labels != -1]])
                np.testing.assert_array_equal(model.part.dist_table[model.part.homenode, np.arange(model.part.amount)], 0)
                self.assertTrue(((model.part.curnode >= 0) & (model.part.curnode < 9)).all())
                self.assertTrue(np.isfinite(model.node.dominance).all())
                np.testing.assert_allclose(model.node.dominance.sum(axis=1), 1., rtol=0., atol=1e-14)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'trace'
                    write_output(path, trace, metadata(dataset='nine', label_seed=10, collector=trace))
                    payload = read_output(path)
                    self.assertEqual(payload['observations'], trace.rows)
                    self.assertEqual(payload['metadata']['seed_policy'], 'cython-fixed-seed')
                    for key, value in (('propagation_seed', 19), ('rng_fingerprint', 'not-captured')):
                        bad = copy.deepcopy(payload)
                        bad['metadata'][key] = value
                        with self.assertRaises(ValueError):
                            validate_payload(bad)
            hashes = [array_hash(a) for a in (predictions, model.node.dominance, model.part.curnode,
                       model.part.strength, model.part.dist_table, model.owndeg)]
            outputs.append((hashes, info, rows))
        for output in outputs[1:]:
            self.assertEqual(outputs[0][:2], output[:2])
        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual([r['state_hashes'] for r in outputs[1][2]], [r['state_hashes'] for r in outputs[3][2]])
        with self.assertRaises(ValueError):
            observe_fit(PCC(impl='cython'), [0, 1], propagation_seed=19)


if __name__ == '__main__':
    unittest.main()
