"""Tiny Stage A synthetic checks. No historical fixture loads or downloads."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.convergence_trajectories import (
    Trajectory, array_hash, metadata, observe_fit, read_output, write_output)
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numpy import pcc_propagate_numpy
from tests.test_sequential_safety import state, buffers


class TrajectoryTests(unittest.TestCase):
    def test_continuous_state_rng_and_stopping_parity(self):
        for cap, early, plateau in ((50, False, False), (100, True, True),
                                    (12, True, False), (21, True, True), (0, True, True)):
            outputs = []
            for observed in (False, True, True):
                args = state()
                args[3] = .5
                if plateau:
                    args[1][:] = 0
                collector = Trajectory(args[2], truth=np.zeros(3, dtype=int),
                                       milestones=(0, 2, 10, 10000))
                np.random.seed(19)
                info = pcc_propagate_numpy(*args, *buffers(args), cap, early, 2000, 1,
                                           return_info=True, observer=collector if observed else None)
                hashes = {i: array_hash(args[i]) for i in (7, 9, 10, 11, 12)}
                outputs.append((info, hashes, np.random.random(), collector.rows))
                if observed:
                    self.assertEqual(collector.rows[-1]['iteration'], info['effective_iterations'])
                    self.assertEqual(collector.rows[-1]['stop_reason'], info['stop_reason'])
                    self.assertEqual(len({r['iteration'] for r in collector.rows}), len(collector.rows))
                    self.assertEqual(collector.rows[-1]['state_hashes']['dominance'], hashes[11])
                    np.testing.assert_allclose(args[11].sum(axis=1), 1.)
                    self.assertTrue(np.isfinite(args[11]).all())
                    self.assertTrue(((args[7] >= 0) & (args[7] < 3)).all())
                    self.assertEqual(args[10].dtype, np.uint8)
            self.assertEqual(outputs[0][:3], outputs[1][:3])
            self.assertEqual(outputs[1][:3], outputs[2][:3])
            strip = lambda rows: [{k: v for k, v in r.items() if k != 'elapsed_instrumented_s'} for r in rows]
            self.assertEqual(strip(outputs[1][3]), strip(outputs[2][3]))

    def test_fit_predictions_and_label_mapping(self):
        labels = np.array([10, -1, -1, 20])
        truth = np.array([10, 10, 20, 20])
        edges = np.array([[1, 2], [0, 2], [1, 3], [1, 2]])
        results = []
        for observed in (False, True):
            model = PCC(impl='numpy')
            model.set_graph(edges, np.full(4, 2))
            np.random.seed(42)
            if observed:
                predictions, trajectory, info = observe_fit(model, labels, truth=truth,
                                                            max_iter=30, early_stop=False)
                self.assertEqual(info['effective_iterations'], 30)
                self.assertEqual(trajectory.rows[-1]['accuracy_unlabeled_retrospective'],
                                 float(np.mean(predictions[labels == -1] == truth[labels == -1])))
                self.assertIs(model._get_backend_fn(), pcc_propagate_numpy)
            else:
                predictions = model.fit_predict(labels, max_iter=30, early_stop=False)
            results.append((predictions, model))
            np.testing.assert_array_equal(predictions[labels != -1], labels[labels != -1])
        np.testing.assert_array_equal(results[0][0], results[1][0])
        for attr in ('curnode', 'strength', 'dist_table'):
            self.assertEqual(array_hash(getattr(results[0][1].part, attr)),
                             array_hash(getattr(results[1][1].part, attr)))
        for model_attr in ('owndeg',):
            np.testing.assert_array_equal(getattr(results[0][1], model_attr), getattr(results[1][1], model_attr))
        np.testing.assert_array_equal(results[0][1].node.dominance, results[1][1].node.dominance)

    def test_two_class_branching_graph_seed_and_truth_independence(self):
        edges = np.array([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=np.int64)
        for seed in (10, 19):
            outputs = []
            for truth in (None, np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])):
                model = PCC(impl='numpy')
                model.set_graph(edges, np.full(4, 2))
                np.random.seed(seed)
                prediction, trajectory, info = observe_fit(
                    model, [0, -1, -1, 1], truth=truth, milestones=(0, 3, 7),
                    max_iter=35, early_stop=True)
                outputs.append((prediction.tolist(), info,
                                [row['state_hashes'] for row in trajectory.rows]))
                np.testing.assert_allclose(model.node.dominance.sum(axis=1), 1.)
                self.assertTrue(np.isfinite(model.node.dominance).all())
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[1], outputs[2])

    def test_validation_ties_zero_changes_and_missing_classes(self):
        for schedule in ((2, 1), (1, 1), (-1,), (1.5,), (True,)):
            with self.assertRaises(ValueError):
                Trajectory([-1, -1], milestones=schedule)
        with self.assertRaises(ValueError):
            observe_fit(PCC(impl='numba'), [0, 1])
        with self.assertRaises(ValueError):
            observe_fit(PCC(impl='numpy'), [0, -1])
        args = state()
        args[1][:] = 0
        trajectory = Trajectory(args[2], milestones=(0, 1))
        pcc_propagate_numpy(*args, *buffers(args), 1, False, 2000, 1, observer=trajectory)
        self.assertEqual(trajectory.rows[-1]['class_changes'], 0)
        self.assertEqual(trajectory.rows[-1]['margin_min'], 0.)
        self.assertEqual(trajectory.rows[-1]['dominance_max_abs_change'], 0.)
        self.assertEqual(trajectory.rows[-1]['assignments'], [0, 0, 0])

    def test_portable_bundle_and_partial_corruption(self):
        args = state()
        trajectory = Trajectory(args[2], milestones=(0,))
        provenance = metadata(dataset='tiny', label_seed=0, propagation_seed=19,
                              arrays={'labels': args[2], 'neighbors': args[0]}, parameters={'max_iter': 2})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'bundle'
            with self.assertRaises(ValueError):
                write_output(directory, trajectory, provenance)
            pcc_propagate_numpy(*args, *buffers(args), 2, False, 2000, 1, observer=trajectory)
            write_output(directory, trajectory, provenance)
            self.assertEqual(read_output(directory)['observations'], trajectory.rows)
            with (directory / 'trajectory.csv').open(newline='', encoding='utf-8') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(trajectory.rows))
            (directory / 'trajectory.csv').write_text('corrupt', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_output(directory)
            (directory / 'manifest.json').unlink()
            with self.assertRaises(FileNotFoundError):
                read_output(directory)


if __name__ == '__main__':
    unittest.main()
