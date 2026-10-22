"""Small synthetic Stage B scheduling/recovery tests; never load historical data."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.checkpoints import CheckpointError, Store, file_inventory, read_record
from experiments.convergence_pilot import (Pilot, ROOT, MAX_BYTES, MILESTONES,
    preflight, specification, validate_spec, verify_report, resources, load_fixture, validate_fixture, digest_fixture)
from benchmarks.fixtures import Case, array_hash
from benchmarks.historical import fingerprint


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.preflight = self.base/'preflight'
        self.report = preflight(self.preflight, synthetic=True, iterations=7)
        self.pilot = Pilot.create(self.base/'run', self.preflight)

    def test_real_specification_and_dry_run_never_fit(self):
        spec = specification()
        self.assertEqual(len(spec['tasks']), 12)
        self.assertEqual(spec['parameters']['max_iter'], 100000)
        self.assertIs(spec['parameters']['early_stop'], False)
        self.assertEqual(spec['milestones'], list(MILESTONES))
        self.assertEqual(len(set(t['case_id'] for t in spec['tasks'])), 4)
        self.assertEqual(spec['propagation_seeds'], dict(numpy=0, numba=0, cython=123456789))
        for key, value in (('max_iter', 99999), ('early_stop', True), ('p_grd', .6)):
            changed = copy.deepcopy(spec)
            changed['parameters'][key] = value
            with self.assertRaises(CheckpointError):
                validate_spec(changed)
        with patch('experiments.convergence_pilot.observe_fit', side_effect=AssertionError('fit called')):
            report = preflight(self.base/'another-dry-run', synthetic=True, iterations=7)
            self.assertFalse(report['execution_started'])
            self.assertEqual(report['total_iteration_budget'], 84)
            self.assertTrue(report['observation_budget_fits'])
        for case in (Case(**d) for d in spec['cases']):
            # Dimensions-only resource formula; no historical arrays are loaded.
            import numpy as np
            arrays = dict(labels=np.full(case.nodes, -1), degrees=np.full(case.nodes, 10),
                          neighbors=np.zeros((case.nodes, 77 if case.name == 'spambase' else 24), dtype=np.int64))
            for backend in ('numpy', 'numba', 'cython'):
                row = resources(case, backend, spec, arrays)
                self.assertEqual(row['expected_observations'], 10)
                self.assertLess(row['estimated_peak_bytes'], MAX_BYTES)

    def test_resume_completes_exactly_twelve_and_skips_completed_tasks(self):
        from experiments.convergence_pilot import observe_fit
        with patch('experiments.convergence_pilot.observe_fit', wraps=observe_fit) as fitted:
            first = self.pilot.execute(max_runs=2)
            self.assertEqual(first['complete'], 2)
            results = {p.name: p.read_bytes() for p in (self.pilot.store.root/'results').iterdir()}
            done = Pilot(self.pilot.store.root).execute()
            self.assertEqual(done['complete'], 12)
            self.assertEqual(fitted.call_count, 12)
            self.assertEqual(self.pilot.execute()['complete'], 12)
            self.assertEqual(fitted.call_count, 12)
            self.assertTrue(all((self.pilot.store.root/'results'/name).read_bytes() == data for name, data in results.items()))
            with self.assertRaises(CheckpointError):
                self.pilot.execute(retry_task='t0000', reason='completed task must not repeat')

    def test_bundle_and_result_publication_reconcile_without_refit(self):
        for point in ('bundle_finished', 'result_published', 'committed'):
            pilot = Pilot.create(self.base/point, self.preflight)
            def stop(at):
                if at == point:
                    raise RuntimeError('simulated interruption')
            with self.assertRaises(RuntimeError):
                pilot.execute(max_runs=1, fault=stop)
            with patch('experiments.convergence_pilot.observe_fit', side_effect=AssertionError('duplicate fit')):
                resumed = Pilot(pilot.store.root).execute(max_runs=0)
            self.assertEqual(resumed['complete'], 1)
            self.assertEqual(len([e for e in pilot.store.events() if e['kind'] == 'pilot_started']), 1)

    def test_incomplete_fit_requires_retry_and_preserves_partial_output(self):
        def stop(at):
            if at == 'fit_finished':
                raise RuntimeError('interrupted before publication')
        with self.assertRaises(RuntimeError):
            self.pilot.execute(max_runs=1, fault=stop)
        state = self.pilot.audit()
        attempt = self.pilot.store.root/'sessions'/state['last']['t0000']
        partial = attempt/'trajectory'
        partial.mkdir()
        (partial/'trajectory.json').write_text('partial', encoding='utf-8')
        with patch('experiments.convergence_pilot.observe_fit', side_effect=AssertionError('automatic retry')):
            with self.assertRaises(CheckpointError):
                self.pilot.execute(max_runs=1)
        result = self.pilot.execute(max_runs=1, retry_task='t0000', reason='test interruption before manifest')
        self.assertEqual(result['complete'], 1)
        self.assertEqual((partial/'trajectory.json').read_text(), 'partial')
        (partial/'trajectory.json').write_text('changed preserved partial')
        with self.assertRaises(CheckpointError):
            self.pilot.status()

    def test_corruption_and_fixture_drift_are_rejected_without_fit(self):
        self.pilot.execute(max_runs=1)
        state = self.pilot.audit()
        manifest = self.pilot.store.root/'sessions'/state['last']['t0000']/'trajectory'/'manifest.json'
        manifest.write_text('{}')
        with patch('experiments.convergence_pilot.observe_fit', side_effect=AssertionError('corruption rerun')):
            with self.assertRaises((ValueError, CheckpointError)):
                self.pilot.execute()
        fixture = self.preflight/'fixtures'/'pilot_a-labels10'/'arrays.npz'
        with fixture.open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaises(CheckpointError):
            verify_report(self.preflight)

    def test_review_gate_and_environment_drift_reject_execution(self):
        state = self.pilot.audit()
        state['report']['specification']['synthetic'] = False
        with patch.object(self.pilot, 'audit', return_value=state), patch('experiments.convergence_pilot.observe_fit') as fitted:
            with self.assertRaisesRegex(CheckpointError, 'review'):
                self.pilot.execute()
            fitted.assert_not_called()
        with patch('experiments.convergence_pilot.execution_identity', return_value={'changed': True}):
            with self.assertRaises(CheckpointError):
                self.pilot.execute()

    def test_older_campaign_is_refused_before_any_write(self):
        legacy = Store.create(self.base/'legacy', dict(campaign_id='a'*32, tasks=[]))
        before = file_inventory(legacy.root)
        with self.assertRaises(CheckpointError):
            Pilot(legacy.root)
        self.assertEqual(file_inventory(legacy.root), before)

    def test_frozen_checks_reject_a_rehashed_small_graph(self):
        # Small synthetic stand-in exercises independent frozen pin comparison;
        # no PageBlocks/Spambase source or graph is constructed in CI.
        case = Case(**self.report['specification']['cases'][0])
        arrays, info = load_fixture(self.preflight/'fixtures'/case.case_id, case)
        historical = Case('pageblocks', case.nodes, case.features, case.classes,
                          k_nn=case.k_nn, label_seed=10)
        info['case'] = dict(info['case'], name='pageblocks', synthetic=False)
        reference = dict(raw_data_fingerprint=fingerprint(arrays['raw'], arrays['truth']),
            processed_data_fingerprint=fingerprint(arrays['x'], arrays['truth']),
            graph_fingerprint=fingerprint(arrays['neighbors'], arrays['degrees']),
            label_fingerprints={'10': fingerprint(arrays['labels'])})
        with patch.object(Case, 'validate'), patch('experiments.convergence_pilot.FROZEN', {'pageblocks': reference}):
            validate_fixture(historical, arrays, info)
            arrays['neighbors'][0, :2] = arrays['neighbors'][0, :2][::-1]
            info['hashes'] = {k: array_hash(v) for k, v in arrays.items()}
            info['fixture_sha256'] = digest_fixture(info['hashes'])
            with self.assertRaisesRegex(CheckpointError, 'frozen'):
                validate_fixture(historical, arrays, info)

    def test_duplicate_completion_event_is_refused(self):
        self.pilot.execute(max_runs=1)
        state = self.pilot.audit()
        with self.pilot.store.lock():
            self.pilot.store.event('pilot_completed', task_id='t0000', attempt_id=state['last']['t0000'])
        with self.assertRaises(CheckpointError):
            self.pilot.status()

    def test_real_process_exit_leaves_lock_and_recovers_without_refit(self):
        root = self.pilot.store.root
        code = ('import os; from experiments.convergence_pilot import Pilot; '
                f'Pilot({str(root)!r}).execute(max_runs=1, '
                'fault=lambda point: os._exit(73) if point == "bundle_finished" else None)')
        child = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(child.returncode, 73, child.stderr)
        owner = read_record(root/'writer.lock')
        with self.assertRaises(FileExistsError):
            self.pilot.execute(max_runs=0)
        self.pilot.store.recover_lock(owner['token'], 'test child is dead; preserve and reconcile published bundle')
        with patch('experiments.convergence_pilot.observe_fit', side_effect=AssertionError('duplicate fit')):
            self.assertEqual(self.pilot.execute(max_runs=0)['complete'], 1)


if __name__ == '__main__':
    unittest.main()
