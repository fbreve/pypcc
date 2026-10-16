"""Synthetic-only Stage C tests: no cached real data and no historical fits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.checkpoints import CheckpointError, Store, file_inventory, read_record, digest, canonical, durable_file
from experiments import convergence_pilot as b
from experiments import convergence_stage_c as c


class StageCTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)

    def prepare(self,part='c2'):
        directory=self.root/(part+'-preflight')
        c.preflight(directory,part=part,synthetic=True,iterations=3)
        return c.StageC.create(self.root/part,directory)

    def test_fixed_scope_excludes_completed_stage_b(self):
        one,two=c.specification('c1'),c.specification('c2')
        self.assertEqual(len(one['tasks']),48); self.assertEqual(len(two['tasks']),12)
        self.assertEqual({d['label_seed'] for d in one['cases']},set(range(12,20)))
        self.assertEqual({d['name'] for d in two['cases']},{'banknote','sonar'})
        for spec in (one,two):
            self.assertEqual(spec['milestones'],list(b.MILESTONES))
            self.assertEqual(spec['parameters'],b.specification()['parameters'])
            self.assertEqual(spec['max_bytes'],64*1024*1024)
            self.assertEqual(spec['max_observations'],128)
            changed=copy.deepcopy(spec);changed['cases'][0]['label_seed']=10
            if spec['part']=='c2': changed['parameters']['early_stop']=True
            with self.assertRaises(CheckpointError): c.validate_spec(changed)

    def test_preflight_does_not_fit_and_checks_totals(self):
        with patch.object(b,'observe_fit',side_effect=AssertionError('fit in preflight')):
            pilot=self.prepare('c1');report=c.verify_report(pilot.store.plan()['preflight'])
            self.assertEqual(report['run_count'],48)
            self.assertEqual(report['total_iteration_budget'],144)
            self.assertFalse(report['execution_started'])
            self.assertEqual(report['aggregate_export_limit_bytes'],48*b.MAX_BYTES)

    def test_c1_all_48_resume_and_duplicate_prevention(self):
        pilot=self.prepare('c1')
        with patch.object(b,'observe_fit',wraps=b.observe_fit) as fit:
            self.assertEqual(pilot.execute(max_runs=13)['complete'],13)
            before=file_inventory(pilot.store.root/'results')
            self.assertEqual(c.StageC(pilot.store.root).execute()['complete'],48)
            self.assertEqual(fit.call_count,48)
            after=file_inventory(pilot.store.root/'results')
            self.assertTrue(all(after[k]==v for k,v in before.items()))
            self.assertEqual(pilot.execute()['complete'],48);self.assertEqual(fit.call_count,48)
            with self.assertRaises(CheckpointError):pilot.execute(retry_task='t0000',reason='duplicate')

    def test_publication_interruption_recovers_without_refitting(self):
        pilot=self.prepare()
        def fault(point):
            if point=='bundle_finished':raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):pilot.execute(max_runs=1,fault=fault)
        with patch.object(b,'observe_fit',side_effect=AssertionError('duplicate fit')):
            self.assertEqual(c.StageC(pilot.store.root).execute(max_runs=0)['complete'],1)
        self.assertEqual(pilot.execute()['complete'],12)

    def test_partial_requires_explicit_retry_and_preserves_artifacts(self):
        pilot=self.prepare()
        def fault(point):
            if point=='fit_finished':raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):pilot.execute(max_runs=1,fault=fault)
        before=file_inventory(pilot.store.root/'sessions')
        with self.assertRaises(CheckpointError):pilot.execute(max_runs=1)
        pilot.execute(max_runs=1,retry_task='t0000',reason='synthetic interruption')
        after=file_inventory(pilot.store.root/'sessions')
        self.assertTrue(all(after[k]==v for k,v in before.items()))
        self.assertEqual(pilot.status()['complete'],1)

    def test_corruption_and_environment_drift_rejected(self):
        pilot=self.prepare();pilot.execute(max_runs=1)
        state=pilot.audit(); bundle=pilot.store.root/'sessions'/state['last']['t0000']/'trajectory'
        (bundle/'trajectory.csv').write_text('corrupt',encoding='utf-8')
        with self.assertRaises(CheckpointError):pilot.status()
        with patch.object(c,'identity',return_value={}):
            with self.assertRaises(CheckpointError):c.verify_report(pilot.store.plan()['preflight'])

    def test_refuse_stage_b_store_before_any_write(self):
        pre=self.root/'b-preflight';b.preflight(pre,synthetic=True,iterations=3)
        old=b.Pilot.create(self.root/'b',pre);before=file_inventory(old.store.root)
        with self.assertRaises(CheckpointError):c.StageC(old.store.root)
        self.assertEqual(file_inventory(old.store.root),before)

    def test_frozen_stage_b_audit_ignores_current_identity_but_cannot_execute(self):
        pre=self.root/'old-preflight';pre.mkdir()
        report=dict(specification=b.specification(),identity=dict(source_commit='old-commit'))
        durable_file(pre/'report.json',canonical(dict(schema=1,payload=report,sha256=digest(report)))+b'\n')
        plan=dict(pilot_kind='convergence_trajectory_pilot',campaign_id='b'*32,
                  tasks=report['specification']['tasks'],report=report,report_sha256=digest(report),preflight=str(pre))
        Store.create(self.root/'old',plan);frozen=c.FrozenStageB(self.root/'old')
        with patch.object(c,'check_fixtures'),patch.object(b,'execution_identity',side_effect=AssertionError('current environment consulted')):
            self.assertEqual(frozen.context()[1],report)
        with self.assertRaises(CheckpointError):frozen.execute()

    def test_sealed_dependency_rejects_changes(self):
        source=self.root/'source';source.mkdir();(source/'a').write_text('original')
        dep=dict(root=str(source),analysis=str(source),preflight=str(source),
                 files=file_inventory(source),analysis_files=file_inventory(source),preflight_files=file_inventory(source))
        c.verify_dependency(dep);(source/'a').write_text('changed')
        with self.assertRaises(CheckpointError):c.verify_dependency(dep)

    def test_output_cannot_overlap_protected_sources(self):
        source=self.root/'historical';source.mkdir()
        before=file_inventory(source)
        with self.assertRaises(CheckpointError):
            c.preflight(source/'nested',part='c2',synthetic=True,fixture_cache=source)
        self.assertEqual(file_inventory(source),before)
        pilot=self.prepare()
        with self.assertRaises(CheckpointError):
            c.StageC.create(Path(pilot.store.plan()['preflight'])/'nested',pilot.store.plan()['preflight'])

    def test_real_gate_requires_exact_reviewed_sha_without_fit(self):
        pilot=self.prepare();plan=pilot.store.plan();report=plan['report']
        fake=copy.deepcopy(report);fake['specification']['synthetic']=False
        original=pilot.audit
        def audit():
            state=original();state['report']=fake;return state
        with patch.object(pilot,'audit',side_effect=audit),patch.object(b,'observe_fit',side_effect=AssertionError('fit')):
            with self.assertRaises(CheckpointError):pilot.execute(max_runs=1,reviewed_preflight='wrong')

    def test_historical_matching_and_fixture_drift_without_real_data(self):
        spec=c.specification('c2',True,3)
        params={k:spec['parameters'][k] for k in ('p_grd','delta_v','deltap','dexp','es_chk')}
        old=self.root/'fixed';old.mkdir();early=self.root/'early'
        tasks=[dict(t,task_id=f't{6*i+j:04d}',measurement_index=j) for i,t in enumerate(spec['tasks']) for j in range(6)]
        plan=dict(campaign_id='a'*32,tasks=tasks,parameters=params,max_iter=500000,propagation_seed=0)
        Store.create(early,plan)
        (old/'manifest.json').write_text(json.dumps(dict(configuration=dict(parameters=params))))
        fixtures={t['case_id']:dict(fixture_sha256='f'*64) for t in spec['tasks']}
        fixed=[]
        for t in tasks:
            row=dict(case_id=t['case_id'],backend=t['backend'],repeat=t['measurement_index'],
                     fixture_sha256='f'*64,propagation_seed_effective=spec['propagation_seeds'][t['backend']],
                     mode='fixed',max_iter=10000,effective_iterations=10000)
            fixed.append(row)
            row=dict(row,mode='early_stopping',max_iter=500000,effective_iterations=12345)
            rec=dict(task_id=t['task_id'],plan_sha256=digest(plan),row=row)
            durable_file(early/'results'/(t['task_id']+'.json'),canonical(dict(schema=1,payload=rec,sha256=digest(rec)))+b'\n')
        (old/'measurements.jsonl').write_text('\n'.join(json.dumps(r) for r in fixed))
        result=c.historical_check(old,early,spec,fixtures)
        self.assertTrue(result['compatible']);self.assertEqual(len(result['groups']),12)
        changed=copy.deepcopy(fixtures);changed[spec['tasks'][0]['case_id']]['fixture_sha256']='e'*64
        with self.assertRaises(CheckpointError):c.historical_check(old,early,spec,changed)


if __name__=='__main__':unittest.main()
