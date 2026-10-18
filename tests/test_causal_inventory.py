"""Real Store/ledger, full frozen inventory, simulated bundles; never any fit."""
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from benchmarks.checkpoints import CheckpointError, Store, digest, file_inventory, read_record
from experiments import causal_selection as a
from experiments import mechanistic_pilot as m

# Approved a2b32da preflight's ordered tasks, removing only operational task_id.
APPROVED_ORDER_SHA = '82108f7bda9f2ff4412935a51ce06989b8adcfad6ce964027ac119222aa6fe71'


class CausalInventoryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)/'simulated-store'
        self.spec = a.specification(); self.tasks = self.spec['tasks']
        self.report = dict(specification=self.spec)
        self.plan = dict(pilot_kind=a.KIND, campaign_id=uuid.uuid4().hex, tasks=self.tasks,
            report=self.report, report_sha256=digest(self.report), preflight='unused-simulation')
        Store.create(self.root, self.plan)
        self.pilot = a.CausalPilot(self.root)  # Actual Store.plan must accept all102.
        for target in ('observe_fit',):
            p = patch.object(a, target, side_effect=AssertionError('no propagation allowed'))
            p.start(); self.addCleanup(p.stop)
        p = patch.object(a.m, 'PCC', side_effect=AssertionError('no model allowed'))
        p.start(); self.addCleanup(p.stop)
        # Only preflight loading and binary propagation payload are simulated;
        # plan pin, locks, ledger, result paths and reconciliation remain real.
        p = patch.object(self.pilot, 'context', side_effect=lambda: (self.pilot.store.plan(), self.report))
        p.start(); self.addCleanup(p.stop)
        def validate(path, task, report):
            payload = json.loads((path/'manifest.json').read_text())
            self.assertEqual(payload, dict(task_id=task['task_id'], slate_id=task['slate_id'],
                dataset_annotation=task['dataset_annotation'], backend=task['backend']))
            return payload
        p = patch.object(self.pilot, 'validate_bundle', side_effect=validate)
        p.start(); self.addCleanup(p.stop)

    def publish(self, task, complete=True):
        aid = uuid.uuid4().hex
        self.pilot.store.event('pilot_started', task_id=task['task_id'], attempt_id=aid)
        path = self.root/'sessions'/aid/'trajectory'; path.mkdir(parents=True)
        if complete:
            (path/'manifest.json').write_text(json.dumps(dict(task_id=task['task_id'], slate_id=task['slate_id'],
                dataset_annotation=task['dataset_annotation'], backend=task['backend'])))
        return aid

    def test_exact_order_scope_and_actual_store_roundtrip(self):
        self.assertEqual(len(self.tasks), 102)
        self.assertEqual([t['task_id'] for t in self.tasks], [f't{i:04d}' for i in range(102)])
        stripped = [{k:v for k,v in t.items() if k!='task_id'} for t in self.tasks]
        self.assertEqual(digest(stripped), APPROVED_ORDER_SHA)
        self.assertEqual(self.spec, a.specification())
        self.assertEqual(Store(self.root).plan(), self.plan)
        self.assertEqual(len({t['slate_id'] for t in self.tasks}), 34)
        self.assertEqual(len({(t['dataset'],t['block']) for t in self.tasks}), 17)
        self.assertEqual(len({(t['dataset'],t['block'],t['arm'],t['backend']) for t in self.tasks}),102)
        self.assertEqual(len({t['task_id']+'.json' for t in self.tasks}),102)

    def test_empty_status_and_original_mechanistic_twelve(self):
        status = self.pilot.status()
        self.assertEqual((status['complete'],status['total'],len(status['pending'])),(0,102,102))
        # Mechanistic uses the unchanged shared status against its actual scope.
        other = object.__new__(m.MechanisticPilot)
        tasks = m.specification()['tasks']; self.assertEqual(len(tasks),12)
        other.audit = lambda: dict(plan=dict(report_sha256='original',tasks=tasks), done=[],recoverable=[],ambiguous=[])
        self.assertEqual(other.status()['total'],12)
        self.assertEqual(len(other.status()['pending']),12)

    def test_partial_recoverable_and_complete_reconcile_without_omission_or_duplication(self):
        with self.pilot.store.lock():
            for task in self.tasks:
                self.publish(task)
            state = self.pilot.audit()
            for task in self.tasks[:19]:
                self.pilot.commit_bundle(state, task['task_id'])
        status = self.pilot.status()
        self.assertEqual((status['complete'],status['total'],len(status['pending']),len(status['recoverable'])),(19,102,83,83))
        final = self.pilot.execute(reviewed_preflight=digest(self.report),max_runs=0)
        self.assertEqual((final['complete'],final['total'],final['pending'],final['recoverable']),(102,102,[],[]))
        state = self.pilot.audit()
        self.assertEqual(set(state['done']), {t['task_id'] for t in self.tasks})
        for task in self.tasks:
            record = read_record(self.root/'results'/(task['task_id']+'.json'))
            self.assertEqual(record['task_id'],task['task_id'])
            self.assertEqual(record['attempt_id'],state['last'][task['task_id']])
            self.assertEqual(record['plan_sha256'],digest(self.plan))
        before = file_inventory(self.root/'results'); events = self.pilot.store.events()
        self.assertEqual(len([e for e in events if e['kind']=='pilot_started']),102)
        self.assertEqual(len([e for e in events if e['kind']=='pilot_completed']),102)
        self.assertEqual(self.pilot.execute(reviewed_preflight=digest(self.report),max_runs=102)['complete'],102)
        self.assertEqual(file_inventory(self.root/'results'),before)
        self.assertEqual(self.pilot.store.events(),events)

    def test_failed_attempt_explicit_retry_and_last_task_recovery(self):
        task = self.tasks[-1]  # Not limited to the original12 task IDs.
        with self.pilot.store.lock():
            aid = self.publish(task,complete=False)
        status = self.pilot.status()
        self.assertEqual((status['total'],status['complete'],len(status['pending'])),(102,0,102))
        self.assertEqual(status['ambiguous'],[task['task_id']])
        with self.assertRaises(CheckpointError):
            self.pilot.execute(reviewed_preflight=digest(self.report),max_runs=0)
        old = file_inventory(self.root/'sessions'/aid)
        self.pilot.execute(reviewed_preflight=digest(self.report),max_runs=0,retry_task=task['task_id'],reason='simulated incomplete export')
        with self.pilot.store.lock():
            self.publish(task)
        self.assertEqual(self.pilot.status()['recoverable'],[task['task_id']])
        result = self.pilot.execute(reviewed_preflight=digest(self.report),max_runs=0)
        self.assertEqual((result['total'],result['complete'],len(result['pending'])),(102,1,101))
        self.assertEqual(file_inventory(self.root/'sessions'/aid),old)

    def test_result_identity_collision_or_cross_task_reference_rejected(self):
        with self.pilot.store.lock():
            aid = self.publish(self.tasks[-1])
        path = self.root/'sessions'/aid/'trajectory'/'manifest.json'
        payload = json.loads(path.read_text()); payload['task_id'] = self.tasks[0]['task_id']
        path.write_text(json.dumps(payload))
        with self.assertRaises(AssertionError): self.pilot.audit()


if __name__ == '__main__': unittest.main()
