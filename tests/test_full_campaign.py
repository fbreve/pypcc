"""Full matrix shape, stage gating and bounded mock recovery. No historical fits."""
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
import io
from itertools import permutations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.checkpoints import CheckpointError, Store, digest, read_record
from benchmarks.fixtures import Case, FixturePreparer
from benchmarks.resumable import Controller, full_specification, main
from tests.test_resumable_benchmarks import fake_manifest, fake_measure


class MockMatrixController(Controller):
    """Scheduling/storage validation only: no real arrays or backend execution."""
    def fixture(self,case,plan,directory=None):
        path=(directory or self.store.root/'fixtures'/case.case_id)/'mock-fixture.json'
        if read_record(path)!=asdict(case):
            raise CheckpointError('mock fixture identity changed')
        return {},dict(fixture_sha256=digest(asdict(case)),timings=dict(
            dataset_load_s=0.,preprocessing_and_labels_s=0.,graph_build_s=0.))

    def prepare(self,case,plan):
        target=self.store.root/'fixtures'/case.case_id
        target.mkdir(exist_ok=False)
        self.store.write(target/'mock-fixture.json',asdict(case))
        return self.fixture(case,plan)


class FullCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'campaign'
        self.calls=[]
        def measured(*args,**kwargs):
            if args[3]>2:
                self.calls.append((args[1].case_id,args[2]))
            return fake_measure(*args,**kwargs)
        self.measure=measured
        for name in ('expected_backend','seed_backend'):
            p=patch('benchmarks.resumable.'+name)
            p.start(); self.addCleanup(p.stop)

    def mock(self,synthetic=True):
        return MockMatrixController.create(self.root,full_specification(synthetic),
            manifest_fn=fake_manifest,measure_fn=self.measure)

    def execute(self,controller,**kwargs):
        with redirect_stdout(io.StringIO()):
            return controller.execute(**kwargs)

    def test_full_matrix_identity_permutations_and_plan_is_read_only(self):
        spec=full_specification()
        self.assertEqual(len(spec['cases']),200)
        self.assertEqual(len(spec['tasks']),3600)
        self.assertEqual(spec,full_specification())
        self.assertEqual(spec['max_iter'],500000)
        cases={Case(**c).case_id:Case(**c) for c in spec['cases']}
        self.assertEqual(Counter(c.label_seed for c in cases.values()),{s:20 for s in range(10,20)})
        for case_id in cases:
            selected=[t for t in spec['tasks'] if t['case_id']==case_id]
            orders=[tuple(t['execution_order']) for t in selected if t['order_index']==0]
            self.assertEqual(set(orders),set(permutations(('numpy','numba','cython'))))
            self.assertEqual(len(orders),6)
            for backend in ('numpy','numba','cython'):
                tasks=[t for t in selected if t['backend']==backend]
                self.assertEqual({t['measurement_index'] for t in tasks},set(range(6)))
                self.assertEqual(Counter(t['order_index'] for t in tasks),{0:2,1:2,2:2})
                self.assertEqual(len({t['warmup_task_id'] for t in tasks}),1)
        stage=[t for t in spec['tasks'] if cases[t['case_id']].label_seed==10]
        self.assertEqual([t['task_id'] for t in stage],[f't{i:04d}' for i in range(360)])
        with redirect_stdout(io.StringIO()) as stream:
            main(['--full-campaign'])
        self.assertEqual(json.loads(stream.getvalue()),spec)
        self.assertFalse(self.root.exists())

    def test_incremental_resume_stage_barrier_and_warmup_reuse(self):
        controller=self.mock()
        plan=(self.root/'plan.json').read_bytes()
        with self.assertRaises(CheckpointError):
            self.execute(controller)
        with self.assertRaisesRegex(CheckpointError,'previous seed'):
            self.execute(controller,stage_seed=11)
        first=self.execute(controller,stage_seed=10,max_tasks=5)
        self.assertEqual(first['stages']['10']['completed'],5)
        before={p.name:p.read_bytes() for p in (self.root/'results').iterdir()}
        second=self.execute(controller,stage_seed=10)
        self.assertEqual(second['status'],'partial')
        self.assertEqual(second['stages']['10']['status'],'complete')
        self.assertEqual(second['stages']['11']['status'],'not_started')
        self.assertEqual(len(self.calls),18)
        self.assertTrue(all((self.root/'results'/n).read_bytes()==b for n,b in before.items()))
        self.assertTrue(all(g['repetitions']==6 and g['complete'] for g in second['groups']))
        self.execute(controller,stage_seed=10)
        self.assertEqual(len(self.calls),18)
        self.assertEqual(len(list((self.root/'sessions').glob('*/warmups/*.json'))),12)
        self.execute(controller,stage_seed=11)
        self.assertEqual(len(self.calls),36)
        self.assertEqual((self.root/'plan.json').read_bytes(),plan)

    def test_complete_360_fit_stage_without_touching_remaining_seeds(self):
        controller=self.mock(False)
        spec=controller.store.plan()
        snapshot=self.execute(controller,stage_seed=10)
        self.assertEqual(len(self.calls),360)
        self.assertEqual(snapshot['stages']['10']['completed'],360)
        self.assertEqual(len(snapshot['groups']),60)
        self.assertEqual(len(list((self.root/'sessions').glob('*/warmups/*.json'))),120)
        self.assertEqual(len(controller.audit()['pending']),3240)
        self.assertEqual(controller.store.plan(),spec)
        self.execute(controller,stage_seed=10)
        self.assertEqual(len(self.calls),360)

    def test_repeatability_failure_blocks_publication(self):
        controller=self.mock()
        original=self.measure
        def changed(*args,**kwargs):
            row=original(*args,**kwargs)
            if args[3]>2 and len(self.calls)>3:
                row['state_hashes']['dominance']='b'*64
            return row
        controller.measure=changed
        with self.assertRaisesRegex(CheckpointError,'nonrepeatable'):
            self.execute(controller,stage_seed=10)
        self.assertEqual(len(list((self.root/'results').iterdir())),3)
        self.assertEqual(len(controller.audit()['ambiguous']),1)

    def test_partial_singleton_summary_and_recovery_mid_permutation(self):
        controller=self.mock()
        first=self.execute(controller,stage_seed=10,max_tasks=1)
        self.assertIsNone(first['groups'][0]['propagation_s']['iqr'])
        controller.store.fault=lambda p:(_ for _ in ()).throw(RuntimeError('crash')) if p=='after_result' else None
        with self.assertRaises(RuntimeError):
            self.execute(controller,stage_seed=10)
        self.assertEqual(len(self.calls),2)
        controller.store.fault=lambda p:None
        final=self.execute(controller,stage_seed=10)
        self.assertEqual(len(self.calls),18)
        self.assertEqual(final['stages']['10']['completed'],18)

    def test_event_cache_does_not_hide_corruption_from_strict_audit(self):
        controller=self.mock()
        with controller.store.lock():
            controller.store.event('lock_recovered',token='a'*32,reason='test')
            controller.store.events()
            path=self.root/'events/000000.json'
            path.write_bytes(b'broken chain')
            with self.assertRaises(CheckpointError):
                controller.audit()

    def test_anchor_reuses_data_graph_with_fresh_labels(self):
        controller=Controller.create(self.root,full_specification(True),manifest_fn=fake_manifest,measure_fn=self.measure)
        self.execute(controller,stage_seed=10)
        with patch.object(controller.preparer,'prepare',side_effect=RuntimeError('must reuse saved graph')):
            self.execute(controller,stage_seed=11)
        plan=controller.store.plan()
        a,ma=controller.fixture(Case(**plan['cases'][0]),plan)
        b,mb=controller.fixture(Case(**plan['cases'][1]),plan)
        self.assertEqual(mb['timings']['graph_build_s'],0.)
        for name in ma['hashes']:
            if name!='labels':
                self.assertEqual(ma['hashes'][name],mb['hashes'][name])
        self.assertNotEqual(ma['hashes']['labels'],mb['hashes']['labels'])


if __name__=='__main__':
    unittest.main()
