"""C2-R1 synthetic tests only; no historical fixture or real fit in CI."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.checkpoints import CheckpointError, canonical, digest, file_inventory, read_record
from experiments import convergence_c2_r1 as r
from experiments import mechanistic_pilot as m
from experiments import convergence_pilot as b
from experiments.convergence_trajectories import array_hash, numpy_rng_hash
from experiments.numba_trajectories import seed_numba
from pcc import ParticleCompetitionAndCooperation as PCC
from tests.test_sequential_safety import next_random
from tests.test_numba_trajectories import strip_timing


class C2R1Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.pre = self.root/'preflight'
        r.preflight(self.pre, synthetic=True, iterations=3)
        self.pilot = r.C2R1.create(self.root/'store', self.pre)

    def test_scientific_scope_and_preflight_never_fit(self):
        approved = r.science()
        self.assertEqual(digest(approved), r.SCIENTIFIC_SHA)
        spec = r.specification()
        self.assertEqual({x['name'] for x in spec['cases']}, {'banknote', 'sonar'})
        self.assertEqual({x['label_seed'] for x in spec['cases']}, {10, 11})
        for key, value in approved['scope'].items():
            if key != 'protocol': self.assertEqual(spec[key], value)
        self.assertEqual(len(spec['milestones']), 10)
        self.assertEqual(spec['max_observations'], 128)
        with patch.object(m, 'observe_fit', side_effect=AssertionError('preflight fit')):
            r.preflight(self.root/'dry', synthetic=True, iterations=3)
        altered = copy.deepcopy(approved)
        altered['metrics']['primary_interval'] = [0, 3]
        path = self.root/'bad-science.json'
        path.write_bytes(canonical(dict(schema=1, payload=altered, sha256=digest(altered))))
        with self.assertRaises(CheckpointError): r.science(path)

    def test_resume_all_backends_determinism_and_no_duplicates(self):
        other = r.C2R1.create(self.root/'repeat', self.pre)
        with patch.object(m, 'observe_fit', wraps=m.observe_fit) as fit:
            self.assertEqual(self.pilot.execute(max_runs=1)['complete'], 1)
            self.assertEqual(self.pilot.execute(max_runs=11)['complete'], 12)
            self.pilot.execute()
            self.assertEqual(fit.call_count, 12)
        other.execute()
        for task in self.pilot.store.plan()['tasks']:
            rows = []
            for pilot in (self.pilot, other):
                state = pilot.audit()
                path = pilot.store.root/'sessions'/state['last'][task['task_id']]/'trajectory'
                rows.append(pilot.validate_bundle(path, task, state['report'])['observations'])
            self.assertEqual(strip_timing(rows[0]), strip_timing(rows[1]))

    def test_adapter_unobserved_state_stop_and_rng_parity(self):
        report = read_record(self.pre/'report.json')
        for task in report['specification']['tasks'][:3]:
            backend = task['backend']
            case = next(b.Case(**x) for x in report['specification']['cases'] if b.Case(**x).case_id == task['case_id'])
            arrays, _ = b.load_fixture(self.pre/'fixtures'/case.case_id, case)
            # One task for each backend through the actual transaction/export path.
            np.random.seed(314)
            observed_models = []
            def factory(**kwargs):
                observed_models.append(PCC(**kwargs))
                return observed_models[-1]
            with patch.object(m, 'PCC', side_effect=factory):
                self.pilot.execute(max_runs=1)
            observed_rng = numpy_rng_hash()
            observed_native = [next_random() for _ in range(5)] if backend == 'numba' else None
            state = self.pilot.audit()
            payload = self.pilot.validate_bundle(self.pilot.store.root/'sessions'/state['last'][task['task_id']]/'trajectory', task, report)
            model = PCC(impl=backend)
            model.set_graph(arrays['neighbors'], arrays['degrees'], k_nn=case.k_nn)
            np.random.seed(314)
            if backend == 'numpy': np.random.seed(0)
            if backend == 'numba': seed_numba(0)
            original = model._get_backend_fn()
            info = {}
            def ordinary(*args, **kwargs):
                info.update(original(*args, **kwargs, return_info=True))
            model._get_backend_fn = lambda: ordinary
            pred = model.fit_predict(arrays['labels'], **report['specification']['parameters'])
            ordinary_rng = numpy_rng_hash()
            ordinary_native = [next_random() for _ in range(5)] if backend == 'numba' else None
            final = payload['observations'][-1]
            np.testing.assert_array_equal(pred, np.array(payload['metadata']['class_index_to_label'])[final['assignments']])
            for key, value in dict(dominance=model.node.dominance, positions=model.part.curnode,
                strengths=model.part.strength, distances=model.part.dist_table).items():
                self.assertEqual(array_hash(value), final['state_hashes'][key])
            observed = observed_models[0]
            for actual, expected in ((model.node.dominance, observed.node.dominance),
                (model.part.curnode, observed.part.curnode), (model.part.strength, observed.part.strength),
                (model.part.dist_table, observed.part.dist_table), (model.owndeg, observed.owndeg),
                (model.part.label, observed.part.label), (model.part.homenode, observed.part.homenode)):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(info['effective_iterations'], final['iteration'])
            self.assertEqual(info['stop_reason'], final['stop_reason'])
            self.assertEqual(observed_rng, ordinary_rng)
            self.assertEqual(observed_native, ordinary_native)
        # The copy-only native Cython continuation probe is independently exercised
        # by test_mechanistic_snapshots; public Cython reseeds each propagation.

    def test_interruption_recovery_and_incomplete_retry(self):
        def fault(point):
            if point == 'bundle_finished': raise RuntimeError('interrupt')
        with self.assertRaises(RuntimeError): self.pilot.execute(max_runs=1, fault=fault)
        self.assertEqual(self.pilot.status()['recoverable'], ['t0000'])
        with patch.object(m, 'observe_fit', side_effect=AssertionError('duplicate fit')):
            self.assertEqual(self.pilot.execute(max_runs=0)['complete'], 1)
        def partial(point):
            if point == 'fit_finished': raise RuntimeError('partial')
        with self.assertRaises(RuntimeError): self.pilot.execute(max_runs=1, fault=partial)
        before = file_inventory(self.pilot.store.root/'sessions')
        with self.assertRaises(CheckpointError): self.pilot.execute(max_runs=1)
        self.pilot.execute(max_runs=1, retry_task='t0001', reason='synthetic partial recovery')
        after = file_inventory(self.pilot.store.root/'sessions')
        self.assertTrue(all(after[k] == v for k, v in before.items()))

    def test_corruption_identity_and_rechecksummed_resource_drift(self):
        self.pilot.execute(max_runs=1)
        state = self.pilot.audit()
        path = self.pilot.store.root/'sessions'/state['last']['t0000']/'trajectory'/'dominance.npy'
        with path.open('ab') as f: f.write(b'corrupt')
        with patch.object(m, 'observe_fit', side_effect=AssertionError('refit')):
            with self.assertRaises((ValueError, RuntimeError)): self.pilot.execute()
        report = read_record(self.pre/'report.json')
        report['resources'][0]['observations'] += 1
        (self.pre/'report.json').write_bytes(canonical(dict(schema=1, payload=report, sha256=digest(report))))
        with self.assertRaises(CheckpointError): r.verify_report(self.pre)

    def test_real_execution_gate_and_protected_destinations(self):
        original = self.pilot.audit
        def fake():
            state = original()
            state['report'] = copy.deepcopy(state['report'])
            state['report']['specification']['synthetic'] = False
            return state
        with patch.object(self.pilot, 'audit', side_effect=fake), patch.object(m, 'observe_fit', side_effect=AssertionError('real fit')):
            with self.assertRaises(CheckpointError): self.pilot.execute(reviewed_preflight=r.SCIENTIFIC_SHA)
        with self.assertRaises(CheckpointError): r.C2R1.create(self.pre/'nested', self.pre)
        with self.assertRaises(CheckpointError): m.MechanisticPilot(self.pilot.store.root)
        with patch.object(r, 'identity', return_value={}):
            with self.assertRaises(CheckpointError): r.verify_report(self.pre)

    def test_historical_prefix_requires_exact_states_and_accuracy(self):
        self.pilot.execute(max_runs=1)
        state = self.pilot.audit()
        task = state['plan']['tasks'][0]
        payload = self.pilot.validate_bundle(self.pilot.store.root/'sessions'/state['last']['t0000']/'trajectory', task, state['report'])
        payload = copy.deepcopy(payload)
        row = payload['observations'][-1]
        row['iteration'] = 10000
        pred = np.array(payload['metadata']['class_index_to_label'], dtype=np.int64)[row['assignments']]
        hashes = {k: v for k, v in row['state_hashes'].items() if k not in ('assignments', 'numpy_rng')}
        hashes['predictions'] = array_hash(pred)
        history = dict(prefixes={'t0000': [dict(state_hashes=hashes, accuracy_unlabeled=row['accuracy_unlabeled_retrospective']) for _ in range(6)]})
        r.validate_prefix(payload, task, history)
        # PR16 owndeg was normalized after propagation; this milestone hash is
        # not comparable. Four common fingerprints are the historical contract.
        history['prefixes']['t0000'][0]['state_hashes']['owndeg'] = 'not-comparable'
        r.validate_prefix(payload, task, history)
        history['prefixes']['t0000'][3]['accuracy_unlabeled'] = -1
        with self.assertRaises(CheckpointError): r.validate_prefix(payload, task, history)

    def test_wrong_original_preflight_sha_refused_before_writes(self):
        output = self.root/'refused-real-preflight'
        with patch.object(m, 'observe_fit', side_effect=AssertionError('real fit')):
            with self.assertRaises(CheckpointError):
                r.preflight(output, original_c2=self.pre, scientific_directory=self.root,
                            stage_b=self.root, stage_c1=self.root, mechanistic=self.root)
        self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
