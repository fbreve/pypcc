"""Synthetic-only causal adapter coverage; real datasets never run in CI."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.checkpoints import CheckpointError, digest
from experiments import causal_selection as a
from experiments.convergence_trajectories import array_hash, observe_fit, numpy_rng_hash
from experiments.mechanistic_snapshots import SnapshotConfig
from experiments.numba_trajectories import seed_numba
from pcc import ParticleCompetitionAndCooperation as PCC
from tests.test_numba_trajectories import strip_timing
from tests.test_sequential_safety import next_random


class CausalSelectionTests(unittest.TestCase):
    def test_scientific_catalog_scope_and_no_hidden_selector_argument(self):
        s = a.science(); tasks = a.specification()['tasks']
        self.assertEqual(digest(s), a.SCIENCE_SHA)
        self.assertEqual(len(s['catalog']),34); self.assertEqual(len(tasks),102)
        self.assertEqual({r['dataset'] for r in tasks},{'pageblocks','spambase'})
        self.assertEqual(len({r['slate_id'] for r in tasks}),34)
        for ds, blocks, candidates in [('pageblocks',10,132),('spambase',7,60)]:
            known=a.candidate_registry(ds); self.assertEqual(len(known),candidates)
            entries=[r for r in s['catalog'] if r['dataset']==ds]
            masks=[]
            for r in entries:
                labels,mask=a.intervention(5473 if ds=='pageblocks' else 4601,known,r['selected_known_node_ids'].split(';'))
                self.assertEqual(np.count_nonzero(labels!=-1),3*len(set(known.values())))
                masks.append(mask)
            self.assertEqual(len({r['block'] for r in entries}),blocks)
            for mask in masks:self.assertTrue(np.array_equal(mask,masks[0]))
        import inspect
        for fn in (a.intervention,a.geometry,a.regenerate_pageblocks):
            self.assertNotIn('truth', inspect.signature(fn).parameters)
        with self.assertRaises(CheckpointError):a.intervention(10,{0:0,1:0,2:0},[0,1,9])

    def test_evaluator_common_mask_components_missing_class(self):
        known={0:0,1:1};truth=np.array([0,1,0,1,1]);pred=np.array([0,1,0,0,1])
        result=a.evaluate(pred,truth,np.array([False,False,True,True,True]),np.array([0,0,0,1,1]),known,2)
        self.assertEqual(result['all']['support'],[1,2]);self.assertEqual(result['all']['accuracy'],2/3)
        self.assertIsNone(result['without_candidates']['balanced_accuracy'])
        self.assertEqual(result['without_candidates']['recall'],[None,.5])

    def test_inventory_tamper_refused_and_preflight_never_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'source';p.write_bytes(b'frozen')
            expected={'source':a.hashlib.sha256(p.read_bytes()).hexdigest()}
            a.verify_inventory(root,expected);p.write_bytes(b'changed')
            with self.assertRaises(CheckpointError):a.verify_inventory(root,expected)
            with patch.object(a,'verify_sources',return_value={}),patch.object(a.b,'source_metadata',return_value={'source_dirty':True}),patch.object(a,'observe_fit',side_effect=AssertionError('preflight fit')):
                with self.assertRaises(CheckpointError):a.preflight(root/'output',root)
            self.assertFalse((root/'output').exists())

    def test_rechecksummed_resource_and_mask_corruption_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot,report,_,_=self.store(Path(tmp));pre=Path(tmp)/'pre'
            altered=copy.deepcopy(report);altered['resources'][0]['snapshot_array_bytes']+=1
            (pre/'report.json').write_bytes(a.canonical(dict(schema=1,payload=altered,sha256=digest(altered))))
            with self.assertRaises(CheckpointError):a.verify_report(pre)
            altered=copy.deepcopy(report);altered['masks']['toy']['hash']='wrong'
            (pre/'report.json').write_bytes(a.canonical(dict(schema=1,payload=altered,sha256=digest(altered))))
            with self.assertRaises(CheckpointError):a.verify_report(pre)

    def test_partial_retry_is_explicit_and_original_attempt_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot,report,_,_=self.store(Path(tmp));sha=digest(report)
            def partial(p):
                if p=='fit_finished':raise RuntimeError('partial')
            with self.assertRaises(RuntimeError):pilot.execute(reviewed_preflight=sha,fault=partial)
            before=a.file_inventory(pilot.store.root/'sessions')
            with self.assertRaises(CheckpointError):pilot.execute(reviewed_preflight=sha)
            self.assertEqual(pilot.execute(reviewed_preflight=sha,retry_task='t0000',reason='synthetic interrupted export')['complete'],1)
            after=a.file_inventory(pilot.store.root/'sessions')
            self.assertTrue(all(after[k]==v for k,v in before.items()))

    def fixture(self):
        n=36;known={i:0 if i<6 else 1 for i in range(12)}
        selected=[0,1,2,6,7,8]
        labels,mask=a.intervention(n,known,selected)
        arrays=dict(neighbors=np.array([[j for j in range(n) if j!=i] for i in range(n)],dtype=np.int64),
            degrees=np.full(n,n-1,dtype=np.int64),truth=np.array([i%2 for i in range(n)],dtype=np.int64),
            source_indices=np.arange(n),evaluation=mask,components=np.zeros(n,dtype=np.int64))
        return known,selected,labels,arrays

    def test_geometry_matching_and_deterministic_structured_regeneration(self):
        # Same-class candidates interleaved on an undirected ring: nontrivial geodesics.
        n=60;neighbors=np.array([[(i-1)%n,(i+1)%n] for i in range(n)],dtype=np.int64)
        known={i:i%5 for i in range(n)}
        g=a.geometry(neighbors,np.full(n,2,dtype=np.int64),known)
        first=a.regenerate_pageblocks(known,g);second=a.regenerate_pageblocks(known,g)
        self.assertEqual(first,second);self.assertEqual(len(first),10)
        self.assertEqual(len({s for p in first for s in p}),20)
        for lo,hi in first:
            total=0
            for k in range(5):
                x,z=[a.triple([v for v in t if known[v]==k],g) for t in (lo,hi)]
                self.assertEqual(x['signature'],z['signature']);self.assertAlmostEqual(x['harmonic'],z['harmonic'])
                total+=z['numerator']-x['numerator']
            self.assertGreaterEqual(total,15)

    def store(self,root):
        known,selected,labels,arrays=self.fixture(); pre=root/'pre';(pre/'fixtures').mkdir(parents=True)
        with (pre/'fixtures'/'toy.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        spec=a.specification();spec['parameters']=dict(spec['parameters'],max_iter=3);spec['milestones']=[0,1,3]
        spec['tasks']=[dict(task_id=f't{i:04d}',dataset='toy',block=0,arm='compact',slate_id='toy',selected=selected,
            backend=b,dataset_annotation='toy:compact') for i,b in enumerate(a.b.BACKENDS)]
        self.enterContext(patch.object(a,'specification',return_value=spec))
        self.enterContext(patch.object(a,'candidate_registry',return_value=known))
        self.enterContext(patch.object(a,'verify_sources',return_value={}))
        resources=[dict(task_id=t['task_id'],**a.estimates(36,2,6,3,t['backend'])) for t in spec['tasks']]
        report=dict(scientific_sha256=a.SCIENCE_SHA,catalog_sha256=a.science()['catalog_sha256'],specification=spec,
            identity=a.identity(),references={},fixtures=dict(toy=dict(file='toy.npz',sha256=a.hashlib.sha256((pre/'fixtures'/'toy.npz').read_bytes()).hexdigest(),
            hashes={k:array_hash(v) for k,v in arrays.items()},nodes=36,classes=2)),masks=dict(toy=dict(hash=array_hash(np.flatnonzero(arrays['evaluation'])))),
            resources=resources,estimated_export_bytes=sum(x['estimated_export_bytes'] for x in resources),
            peak_diagnostic_memory_bytes=max(x['estimated_memory_bytes'] for x in resources),aggregate_export_limit_bytes=3*spec['snapshot_config']['max_export_bytes'],
            execution_started=False,run_count=3,total_iteration_budget=9,observations=9)
        a.seal(pre/'report.json',report)
        return a.CausalPilot.create(root/'store',pre),report,arrays,labels

    def test_transactions_determinism_parity_rng_all_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,report,arrays,labels=self.store(root)
            sha=digest(report)
            with self.assertRaises(CheckpointError):pilot.execute(reviewed_preflight='wrong')
            recorded=[]
            def factory(**kw):
                model=PCC(**kw);recorded.append(model);return model
            for t in report['specification']['tasks']:
                backend=t['backend'];np.random.seed(314)
                with patch.object(a.m,'PCC',side_effect=factory):pilot.execute(reviewed_preflight=sha,max_runs=1)
                observed=recorded[-1];rng=numpy_rng_hash();native=[next_random() for _ in range(5)] if backend=='numba' else None
                ordinary=PCC(impl=backend);ordinary.set_graph(arrays['neighbors'],arrays['degrees'],k_nn=10)
                np.random.seed(314)
                if backend=='numpy':np.random.seed(0)
                if backend=='numba':seed_numba(0)
                original=ordinary._get_backend_fn(); info={}
                def ordinary_native(*args,**kwargs):
                    info.update(original(*args,**kwargs,return_info=True))
                ordinary._get_backend_fn=lambda:ordinary_native
                pred=ordinary.fit_predict(labels,**report['specification']['parameters'])
                self.assertEqual(rng,numpy_rng_hash())
                if native is not None:self.assertEqual(native,[next_random() for _ in range(5)])
                for x,y in ((ordinary.node.dominance,observed.node.dominance),(ordinary.part.curnode,observed.part.curnode),
                    (ordinary.part.strength,observed.part.strength),(ordinary.part.dist_table,observed.part.dist_table),
                    (ordinary.owndeg,observed.owndeg),(ordinary.part.label,observed.part.label),(ordinary.part.homenode,observed.part.homenode)):
                    np.testing.assert_array_equal(x,y)
                state=pilot.audit();path=pilot.store.root/'sessions'/state['last'][t['task_id']]/'trajectory'
                payload=pilot.validate_bundle(path,t,report)
                self.assertEqual(payload['observations'][-1]['iteration'],3)
                self.assertEqual(info['effective_iterations'],3)
                self.assertEqual(info['stop_reason'],payload['observations'][-1]['stop_reason'])
                np.testing.assert_array_equal(pred,np.array(payload['metadata']['class_index_to_label'])[payload['observations'][-1]['assignments']])
                # Existing observer itself verifies final-state and native RNG contracts.
                repeat=PCC(impl=backend);repeat.set_graph(arrays['neighbors'],arrays['degrees'],k_nn=10)
                _,trace,_=observe_fit(repeat,labels,truth=arrays['truth'],milestones=[0,1,3],propagation_seed=report['specification']['propagation_seeds'][backend],snapshot_config=SnapshotConfig(),**report['specification']['parameters'])
                self.assertEqual(strip_timing(payload['observations']),strip_timing(trace.rows))
            with patch.object(a,'observe_fit',side_effect=AssertionError('duplicate')):
                self.assertEqual(pilot.execute(reviewed_preflight=sha,max_runs=102)['complete'],3)

    def test_interruption_recovery_corruption_and_preflight_no_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot,report,_,_=self.store(Path(tmp));sha=digest(report)
            def fault(p):
                if p=='bundle_finished':raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):pilot.execute(reviewed_preflight=sha,fault=fault)
            self.assertEqual(pilot.status()['recoverable'],['t0000'])
            with patch.object(a,'observe_fit',side_effect=AssertionError('duplicate')):
                pilot.execute(reviewed_preflight=sha,max_runs=0)
            state=pilot.audit();path=pilot.store.root/'sessions'/state['last']['t0000']/'trajectory'/'dominance.npy'
            path.write_bytes(path.read_bytes()+b'corruption')
            with self.assertRaises((ValueError,CheckpointError)):pilot.audit()


if __name__=='__main__':unittest.main()
