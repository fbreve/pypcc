"""Synthetic-only full-snapshot pilot transactions and preflight gates."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.checkpoints import CheckpointError, file_inventory
from experiments import mechanistic_pilot as m
from experiments import convergence_pilot as b


class MechanisticPilotTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.pre=self.root/'preflight'
        m.preflight(self.pre,synthetic=True,iterations=3)
        self.pilot=m.MechanisticPilot.create(self.root/'pilot',self.pre)

    def test_scope_limits_and_preflight_without_fit(self):
        spec=m.specification()
        self.assertEqual({d['name'] for d in spec['cases']},{'pageblocks','spambase'})
        self.assertEqual({d['label_seed'] for d in spec['cases']},{10,13})
        self.assertEqual(len(spec['tasks']),12);self.assertEqual(len(spec['milestones']),16)
        self.assertEqual(spec['max_observations'],32);self.assertFalse(spec['parameters']['early_stop'])
        self.assertEqual(spec['parameters']['max_iter'],100000)
        with patch.object(m,'observe_fit',side_effect=AssertionError('fit in preflight')):
            r=m.preflight(self.root/'another',synthetic=True,iterations=3)
        self.assertFalse(r['execution_started']);self.assertEqual(r['total_iteration_budget'],36)

    def test_resume_exactly_twelve_and_no_duplicate(self):
        with patch.object(m,'observe_fit',wraps=m.observe_fit) as fit:
            self.assertEqual(self.pilot.execute(max_runs=2)['complete'],2)
            before=file_inventory(self.pilot.store.root/'results')
            self.assertEqual(m.MechanisticPilot(self.pilot.store.root).execute()['complete'],12)
            self.assertEqual(fit.call_count,12)
            self.assertEqual(self.pilot.execute()['complete'],12);self.assertEqual(fit.call_count,12)
            after=file_inventory(self.pilot.store.root/'results');self.assertTrue(all(after[k]==v for k,v in before.items()))

    def test_recover_valid_publication_without_fit(self):
        def fault(point):
            if point=='bundle_finished':raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):self.pilot.execute(max_runs=1,fault=fault)
        self.assertEqual(self.pilot.status()['recoverable'],['t0000'])
        with patch.object(m,'observe_fit',side_effect=AssertionError('duplicate fit')):
            self.assertEqual(self.pilot.execute(max_runs=0)['complete'],1)
        self.assertFalse(self.pilot.status()['ambiguous'])

    def test_incomplete_fit_requires_explicit_retry_and_preserves_attempt(self):
        def fault(point):
            if point=='fit_finished':raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):self.pilot.execute(max_runs=1,fault=fault)
        before=file_inventory(self.pilot.store.root/'sessions')
        with self.assertRaises(CheckpointError):self.pilot.execute(max_runs=1)
        self.pilot.execute(max_runs=1,retry_task='t0000',reason='synthetic incomplete fit')
        self.assertTrue(all(file_inventory(self.pilot.store.root/'sessions')[k]==v for k,v in before.items()))
        self.assertEqual(self.pilot.status()['complete'],1)

    def test_corruption_or_unreviewed_real_gate_never_refits(self):
        self.pilot.execute(max_runs=1)
        s=self.pilot.audit();path=self.pilot.store.root/'sessions'/s['last']['t0000']/'trajectory'/'positions.npy'
        with path.open('ab') as f:f.write(b'corruption')
        with patch.object(m,'observe_fit',side_effect=AssertionError('fit after corruption')):
            with self.assertRaises((ValueError,RuntimeError)):self.pilot.execute()
        other=m.MechanisticPilot.create(self.root/'other',self.pre);original=other.audit
        def fake():
            state=original();state['report']=copy.deepcopy(state['report']);state['report']['specification']['synthetic']=False;return state
        with patch.object(other,'audit',side_effect=fake),patch.object(m,'observe_fit',side_effect=AssertionError('unreviewed fit')):
            with self.assertRaises(CheckpointError):other.execute(reviewed_preflight='wrong',max_runs=1)

    def test_old_store_or_protected_destination_refused_before_write(self):
        oldpre=self.root/'old-pre';b.preflight(oldpre,synthetic=True,iterations=3)
        old=b.Pilot.create(self.root/'old',oldpre);before=file_inventory(old.store.root)
        with self.assertRaises(CheckpointError):m.MechanisticPilot(old.store.root)
        self.assertEqual(file_inventory(old.store.root),before)
        with self.assertRaises(CheckpointError):m.MechanisticPilot.create(self.pre/'nested',self.pre)
        with patch.object(m,'identity',return_value={}):
            with self.assertRaises(CheckpointError):m.verify_report(self.pre)


if __name__=='__main__':unittest.main()
