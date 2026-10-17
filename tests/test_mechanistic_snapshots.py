"""Small synthetic parity and fail-closed full-snapshot exports; no real data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pcc_step
from pcc import ParticleCompetitionAndCooperation as PCC
from benchmarks.checkpoints import canonical, digest, file_inventory, read_record
from experiments.convergence_trajectories import observe_fit, metadata, array_hash, numpy_rng_hash, Trajectory
from experiments.mechanistic_snapshots import SnapshotConfig, Snapshots, estimates, write_snapshots, read_snapshots, _array_digest
from experiments.native_trajectories import allocate_buffer, finish_trace
from experiments.numba_trajectories import seed_numba
from tests.test_sequential_safety import next_random, state, buffers
from tests.test_numba_trajectories import strip_timing


class MechanisticTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)

    def model(self,backend,plateau=False):
        model=PCC(impl=backend)
        graph=np.array([[1,5],[0,2],[1,3],[2,4],[3,5],[4,0]],dtype=np.int64)
        if plateau:graph[:]=[0,3]
        model.set_graph(graph,np.full(6,2,dtype=np.int64),k_nn=2)
        return model

    def test_streamed_array_hash_uses_existing_canonical_contract(self):
        for shape in ((4,),(3,4),(2,3,4)):
            for dtype in ('<i8','<f8'):
                a=np.arange(np.prod(shape),dtype=dtype).reshape(shape)
                self.assertEqual(_array_digest(a),array_hash(a))

    def run_fit(self,backend,enabled=True,cap=35,early=False,plateau=False):
        model=self.model(backend,plateau)
        np.random.seed(314)
        args=dict(max_iter=cap,early_stop=early,es_chk=0)
        seed=123456789 if backend=='cython' else 0
        labels=np.array([10,-1,-1,20,-1,-1]);truth=np.array([10,10,10,20,20,20])
        if enabled is None:
            if backend=='numpy':np.random.seed(seed)
            if backend=='numba':seed_numba(seed)
            original=model._get_backend_fn();captured={}
            def unobserved(*a,**kw):
                captured['info']=original(*a,**kw,return_info=True)
            model._get_backend_fn=lambda:unobserved
            pred=model.fit_predict(labels,**args);del model._get_backend_fn
            trace=None;info=captured['info']
        else:
            pred,trace,info=observe_fit(model,labels,truth=truth,propagation_seed=seed,
                milestones=tuple(sorted({i for i in (0,1,3,11,21,cap) if i<=cap})),
                snapshot_config=SnapshotConfig() if enabled else None,**args)
        rng=numpy_rng_hash()
        native=[next_random() for _ in range(10)] if backend=='numba' else None
        hashes=[array_hash(a) for a in (pred,model.node.dominance,model.part.curnode,model.part.strength,model.part.dist_table,model.owndeg,model.part.label,model.part.homenode)]
        return model,trace,info,hashes,(rng,native)

    def test_observed_full_and_ordinary_parity_all_backends(self):
        for backend in ('numpy','numba','cython'):
            for cap,early,plateau in ((0,False,False),(35,False,False),(100,True,True)):
                with self.subTest(backend=backend,cap=cap):
                    ordinary=self.run_fit(backend,None,cap,early,plateau)
                    old=self.run_fit(backend,False,cap,early,plateau)
                    full=self.run_fit(backend,True,cap,early,plateau)
                    repeated=self.run_fit(backend,True,cap,early,plateau)
                    self.assertEqual(ordinary[2:],old[2:]);self.assertEqual(old[2:],full[2:]);self.assertEqual(full[2:],repeated[2:])
                    self.assertEqual(strip_timing(old[1].rows),strip_timing(full[1].rows))
                    self.assertEqual(strip_timing(full[1].rows),strip_timing(repeated[1].rows))
                    self.assertIsNone(old[1].snapshots)
                    self.assertEqual(full[1].snapshots.used,len(full[1].rows))
                    if early:self.assertEqual(full[2]['effective_iterations'],21)
                    self.assertEqual(sum(r['final'] for r in full[1].rows),1)

    def test_binary_roundtrip_mapping_readonly_and_no_state_alias(self):
        for backend in ('numpy','numba','cython'):
            model,trace,_,_,_=self.run_fit(backend)
            original=trace.snapshots.arrays['dominance'].copy()
            self.assertFalse(np.shares_memory(model.node.dominance,trace.snapshots.arrays['dominance']))
            model.node.dominance[:]=0
            np.testing.assert_array_equal(original,trace.snapshots.arrays['dominance'])
            dest=self.root/backend
            write_snapshots(dest,trace,metadata(dataset='tiny',label_seed=10,collector=trace))
            with read_snapshots(dest) as bundle:
                np.testing.assert_array_equal(bundle['arrays']['particle_home_nodes'],[0,3])
                np.testing.assert_array_equal(bundle['arrays']['particle_classes'],[0,1])
                np.testing.assert_array_equal(bundle['arrays']['particle_ids'],[0,1])
                self.assertEqual(bundle['summary']['metadata']['class_index_to_label'],[10,20])
                self.assertFalse(bundle['arrays']['dominance'].flags.writeable)
                self.assertEqual(bundle['arrays']['dominance'].dtype,np.dtype('<f8'))
            with self.assertRaises(FileExistsError):write_snapshots(dest,trace,metadata(dataset='tiny',label_seed=10,collector=trace))

    def test_native_buffers_are_reused_not_copied(self):
        for backend in ('numba','cython'):
            captured={}
            from experiments import numba_trajectories,native_trajectories
            module=numba_trajectories if backend=='numba' else native_trajectories
            original=module.allocate_buffer
            def allocate(*args,**kwargs):
                value=original(*args,**kwargs);captured['state']=value[0];return value
            with patch.object(module,'allocate_buffer',side_effect=allocate):
                _,trace,_,_,_=self.run_fit(backend)
            for key,index in [('dominance',2),('positions',3),('strengths',4)]:
                self.assertIs(trace.snapshots.arrays[key],captured['state'][index])

    def test_cython_rng_continuation_with_full_snapshots(self):
        outputs=[]
        for enabled in (None,False,True):
            args=state();args[2]=np.array([0,1,-1],dtype=np.int64)
            args[7]=np.array([0,1],dtype=np.int64);args[8]=np.array([0,1],dtype=np.int64);args[9]=np.ones(2)
            args[10]=np.full((3,2),255,dtype=np.uint8);args[10][[0,1],[0,1]]=0
            args[11]=np.array([[1.,0.],[0.,1.],[.5,.5]])
            args+=buffers(args)+[30,False,2000,1]
            trace=Trajectory(args[2],milestones=(0,1,3,30));trace.execution={}
            recording,_=allocate_buffer(args,[0,1,3,30],trace,cython=True)
            if enabled:
                trace.snapshots=Snapshots(SnapshotConfig(),args[2],np.array([0,1]),args[8],[0,1,3,30],'cython')
                trace.snapshots.bind_native(recording)
            probe=np.empty(11,dtype=np.uint32)
            info=pcc_step._pcc_propagate_impl(*args,return_info=True,_observations=pcc_step._TrajectoryBuffer(recording) if enabled is not None else None,_rng_probe=probe)
            if enabled is not None:finish_trace(recording,trace,info,0,cython=True)
            outputs.append((probe.tolist(),[array_hash(args[i]) for i in (7,9,10,11,12)],strip_timing(trace.rows)))
        self.assertEqual(outputs[0][:2],outputs[1][:2]);self.assertEqual(outputs[1],outputs[2])

    def test_export_zero_and_exact_early_final_all_backends(self):
        for backend in ('numpy','numba','cython'):
            for cap,early in ((0,False),(100,True)):
                _,trace,_,_,_=self.run_fit(backend,cap=cap,early=early,plateau=early)
                dest=self.root/(backend+str(cap))
                write_snapshots(dest,trace,metadata(dataset='tiny',label_seed=10,collector=trace))
                with read_snapshots(dest) as bundle:
                    self.assertEqual(bundle['arrays']['iterations'].tolist(),[r['iteration'] for r in trace.rows])
                    self.assertEqual(bundle['arrays']['iterations'][-1],21 if early else 0)

    def test_exact_capacity_and_export_does_not_consume_backend_rng(self):
        for backend in ('numpy','numba','cython'):
            estimate=estimates(6,2,2,2,backend)
            config=SnapshotConfig(estimate['snapshot_array_bytes'],estimate['estimated_memory_bytes'],estimate['estimated_export_bytes'])
            _,trace,_=observe_fit(self.model(backend),[10,-1,-1,20,-1,-1],max_iter=3,early_stop=False,
                milestones=(0,3),max_observations=2,snapshot_config=config)
            np.random.seed(113);seed_numba(113);expected_native=[next_random() for _ in range(10)]
            seed_numba(113);expected_numpy=np.random.RandomState(113).random_sample(5)
            write_snapshots(self.root/backend,trace,metadata(dataset='tiny',label_seed=10,collector=trace))
            np.testing.assert_array_equal(np.random.random(5),expected_numpy)
            self.assertEqual([next_random() for _ in range(10)],expected_native)

    def test_limits_and_malformed_requests_fail_before_fit(self):
        for config in (SnapshotConfig(max_bytes=1),SnapshotConfig(max_memory_bytes=1),SnapshotConfig(max_export_bytes=1)):
            model=self.model('numpy')
            with patch.object(model,'fit_predict',side_effect=AssertionError('fit called')):
                # effective_parameters introspects PCC's class signature, not this instance.
                with self.assertRaises(ValueError):observe_fit(model,[10,-1,-1,20,-1,-1],max_iter=3,milestones=(0,3),snapshot_config=config)
        for bad in (True,{},0):
            with self.assertRaises(TypeError):observe_fit(self.model('numpy'),[10,-1,-1,20,-1,-1],max_iter=3,milestones=(0,3),snapshot_config=bad)
        for value in (True,0,-1,1.5):
            with self.assertRaises(ValueError):SnapshotConfig(max_bytes=value)
        for schedule in ((3,0),(0,0),(-1,3)):
            with self.assertRaises(ValueError):observe_fit(self.model('numba'),[10,-1,-1,20,-1,-1],max_iter=3,milestones=schedule,snapshot_config=SnapshotConfig())

    def test_interrupted_exports_never_publish_valid_bundle(self):
        _,trace,_,_,_=self.run_fit('numpy')
        provenance=metadata(dataset='tiny',label_seed=10,collector=trace)
        for point in ('summary_written','array_written','before_manifest','manifest_published'):
            def stop(at):
                if at==point:raise RuntimeError('interrupted')
            dest=self.root/point
            with self.assertRaises(RuntimeError):write_snapshots(dest,trace,provenance,fault=stop)
            if point=='manifest_published':
                with read_snapshots(dest):pass
            else:
                with self.assertRaises((ValueError,RuntimeError,OSError)):read_snapshots(dest)
        trace.finished=False
        with self.assertRaises(ValueError):write_snapshots(self.root/'unfinished',trace,provenance)
        self.assertFalse((self.root/'unfinished').exists())

    def rehash(self,root):
        files=file_inventory(root);files.pop('manifest.json')
        payload=dict(protocol='pypcc-mechanistic-snapshots-v1',files=files)
        (root/'manifest.json').write_bytes(canonical(dict(schema=1,payload=payload,sha256=digest(payload)))+b'\n')

    def test_rechecksummed_schema_mapping_and_state_corruption_rejected(self):
        _,trace,_,_,_=self.run_fit('numpy');provenance=metadata(dataset='tiny',label_seed=10,collector=trace)
        for key,value in [('iterations',[0]),('backend','numba'),('observations',True),('node_order','reordered'),('source_hashes',{})]:
            dest=self.root/key;write_snapshots(dest,trace,provenance)
            meta=json.loads((dest/'mechanism.json').read_text());meta[key]=value
            (dest/'mechanism.json').write_text(json.dumps(meta));self.rehash(dest)
            with self.assertRaises((ValueError,RuntimeError)):read_snapshots(dest)
        for key in ('dominance','particle_home_nodes','positions'):
            dest=self.root/(key+'-array');write_snapshots(dest,trace,provenance)
            a=np.load(dest/(key+'.npy'));a.flat[0]=0 if key=='dominance' else 1
            np.save(dest/(key+'.npy'),a,allow_pickle=False)
            meta=json.loads((dest/'mechanism.json').read_text());meta['arrays'][key]['array_sha256']=array_hash(a)
            (dest/'mechanism.json').write_text(json.dumps(meta));self.rehash(dest)
            with self.assertRaises(ValueError):read_snapshots(dest)

    def test_reader_budget_trailing_bytes_and_metadata_drift(self):
        _,trace,_,_,_=self.run_fit('numpy');provenance=metadata(dataset='tiny',label_seed=10,collector=trace)
        dest=self.root/'bundle';write_snapshots(dest,trace,provenance)
        with self.assertRaises(ValueError):read_snapshots(dest,max_export_bytes=1)
        with (dest/'positions.npy').open('ab') as f:f.write(b'corrupt')
        self.rehash(dest)
        with self.assertRaises(ValueError):read_snapshots(dest)
        bad=copy.deepcopy(provenance);bad['parameters']['p_grd']=.7
        with self.assertRaises(ValueError):write_snapshots(self.root/'bad-provenance',trace,bad)
        self.assertFalse((self.root/'bad-provenance').exists())


if __name__=='__main__':unittest.main()
