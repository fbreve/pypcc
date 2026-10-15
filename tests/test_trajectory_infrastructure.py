"""Diagnostic claims must agree with execution, not just with file checksums."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC
from experiments.convergence_trajectories import (Trajectory, observe_fit, metadata,
    write_output, read_output, numpy_rng_hash, schedule_for)
from experiments.trajectory_io import validate_payload


def fixture(backend='numpy', **kwargs):
    edges = np.array([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=np.int64)
    arrays = dict(neighbors=edges, degrees=np.full(4, 2), labels=np.array([10, -1, -1, 20]),
                  truth=np.array([10, 10, 20, 20]))
    model = PCC(impl=backend)
    model.set_graph(edges, arrays['degrees'])
    _, trace, _ = observe_fit(model, arrays['labels'], truth=arrays['truth'],
                             milestones=(0, 3, 7), max_iter=12, early_stop=False, **kwargs)
    return trace, arrays


class InfrastructureTests(unittest.TestCase):
    def test_zero_cap_early_stop_and_fully_labeled_bundles(self):
        for backend in ('numpy', 'numba'):
            for cap in (0, 21, 100):
                model = PCC(impl=backend)
                model.set_graph(np.empty((4, 0), dtype=np.int64), np.zeros(4, dtype=np.int64))
                predictions, trace, info = observe_fit(model, [10, 10, 20, 20],
                    truth=[10, 10, 20, 20], max_iter=cap, milestones=(0, 7),
                    include_checks=True, propagation_seed=19)
                self.assertEqual(info['effective_iterations'], min(cap, 21))
                self.assertEqual(info['stop_reason'], 'max_iter' if cap == 0 else 'early_stop')
                self.assertIsNone(trace.rows[-1]['accuracy_unlabeled_retrospective'])
                np.testing.assert_array_equal(predictions, [10, 10, 20, 20])
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'trace'
                    write_output(path, trace, metadata(dataset='isolated', label_seed=0, collector=trace))
                    self.assertEqual(read_output(path)['metadata']['metrics'], info)

    def test_captured_provenance_defaults_and_claim_consistency(self):
        trace, arrays = fixture(propagation_seed=19)
        provenance = metadata(dataset='tiny', label_seed=10, collector=trace)
        self.assertEqual(provenance['parameters']['es_chk'], 2000)
        self.assertEqual(provenance['parameters']['max_iter'], 12)
        self.assertEqual(provenance['propagation_seed'], 19)
        for kwargs in (dict(parameters={'max_iter': 13}), dict(backend='numba'),
                       dict(propagation_seed=20), dict(k_nn=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                metadata(dataset='tiny', label_seed=0, collector=trace, **kwargs)
        changed = copy.deepcopy(arrays)
        changed['truth'][1] = 20
        with self.assertRaises(ValueError):
            metadata(dataset='tiny', label_seed=0, collector=trace, arrays=changed)
        with self.assertRaises(ValueError):
            metadata(dataset='tiny', label_seed=0, arrays={'labels': arrays['labels']}, parameters={})
        with self.assertRaises(ValueError):
            metadata(dataset='tiny', label_seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            for key, value in (('parameters', {'max_iter': 99}), ('backend', 'numba'),
                               ('nodes', 9), ('seed_policy', 'unverified'), ('source_commit', '0' * 40)):
                bad = copy.deepcopy(provenance)
                bad[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    write_output(Path(tmp) / key, trace, bad)
                self.assertFalse((Path(tmp) / key).exists())
            bad = copy.deepcopy(provenance)
            del bad['fixture_hashes']
            with self.assertRaises(ValueError):
                write_output(Path(tmp) / 'missing', trace, bad)

    def test_legacy_numpy_seed_claim_is_checked_not_assumed(self):
        np.random.seed(19)
        trace, arrays = fixture()
        provenance = metadata(dataset='tiny', label_seed=0, propagation_seed=19, arrays=arrays,
                              parameters=dict(max_iter=12, early_stop=False))
        with tempfile.TemporaryDirectory() as tmp:
            write_output(Path(tmp) / 'good', trace, provenance)
            result = read_output(Path(tmp) / 'good')
            self.assertEqual(result['metadata']['seed_policy'], 'numpy-caller-seed-verified')
            provenance['propagation_seed'] = 20
            with self.assertRaises(ValueError):
                write_output(Path(tmp) / 'wrong-seed', trace, provenance)
        np.random.seed(19)
        np.random.random()
        advanced, _ = fixture()
        with self.assertRaises(ValueError):
            metadata(dataset='tiny', label_seed=0, propagation_seed=19, collector=advanced)

    def test_bounded_schedule_and_preflight_preserve_rng_and_dispatch(self):
        self.assertEqual(schedule_for(500000, [0, 10000], False, 128), [0, 10000, 500000])
        with self.assertRaises(ValueError):
            schedule_for(500000, [0, 10000], True, 128)
        for backend in ('numpy', 'numba'):
            model = PCC(impl=backend)
            model.set_graph(np.array([[1], [2], [0]]), np.ones(3, dtype=np.int64))
            original = model._get_backend_fn()
            before = numpy_rng_hash()
            for kwargs in (dict(include_checks=True, max_iter=500000), dict(max_bytes=1, max_iter=10),
                           dict(max_observations=1, milestones=(0, 2), max_iter=10)):
                with self.subTest(backend=backend, kwargs=kwargs), self.assertRaises(ValueError):
                    observe_fit(model, [0, -1, 1], propagation_seed=19, **kwargs)
                self.assertIs(model._get_backend_fn(), original)
                self.assertEqual(before, numpy_rng_hash())
        with self.assertRaises(ValueError):
            Trajectory([-1], max_observations=0)
        with self.assertRaises(ValueError):
            Trajectory([-1], max_bytes=0)

    def test_schema_semantics_and_rechecksummed_corruption(self):
        trace, _ = fixture(propagation_seed=19)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'trace'
            write_output(directory, trace, metadata(dataset='tiny', label_seed=0, collector=trace))
            original = read_output(directory)
            mutations = [lambda p: p['observations'][1].update(iteration=0),
                         lambda p: p['observations'][1].update(class_changes=999),
                         lambda p: p['observations'][0].update(final=True),
                         lambda p: p['observations'][0].update(margin_min=float('nan')),
                         lambda p: p['observations'][1].update(assignments=[0]),
                         lambda p: p['observations'][-1].update(stop_reason='early_stop'),
                         lambda p: p['metadata']['metrics'].update(effective_iterations=11),
                         lambda p: p['metadata']['parameters'].pop('p_grd'),
                         lambda p: p['metadata'].update(classes=3),
                         lambda p: p['metadata'].update(stop_max=1),
                         lambda p: p['metadata'].update(backend='numba'),
                         lambda p: p['metadata'].update(seed_policy='caller promises'),
                         lambda p: p['metadata'].update(propagation_seed=20),
                         lambda p: p['metadata']['fixture_hashes'].pop('truth'),
                         lambda p: p['metadata']['budget'].update(native_buffer_bytes=1),
                         lambda p: p['metadata'].pop('source_hashes')]
            for mutate in mutations:
                payload = copy.deepcopy(original)
                mutate(payload)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    validate_payload(payload)
            # A correct checksum cannot legitimize invalid row ordering.
            bad = copy.deepcopy(original)
            bad['observations'].reverse()
            path = directory / 'trajectory.json'
            path.write_text(json.dumps(bad), encoding='utf-8')
            manifest_path = directory / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                read_output(directory)
            path.write_text(json.dumps(original), encoding='utf-8')
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            csv_path = directory / 'trajectory.csv'
            csv_path.write_text('a different,checksummed,csv\n')
            manifest[csv_path.name] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                read_output(directory)


if __name__ == '__main__':
    unittest.main()
