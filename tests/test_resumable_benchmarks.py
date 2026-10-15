"""Whole-fit recovery tests; fake measurements and tiny synthetic fixtures only."""
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

from benchmarks.checkpoints import CheckpointError, Store, digest, read_record
from benchmarks.resumable import Controller, specification, main


def fake_manifest(*args):
    return dict(repository=dict(source_hashes={'test-source':'a'}),
                environment=dict(cpu='test',versions={'numpy':'test'}),
                cython_extension=dict(sha256='a'*64))


def fake_measure(arrays,case,backend,iterations,early_stop,seed,**kwargs):
    stop_max=max(1,round(case.nodes/(case.particles*case.k_nn)*200))
    return dict(effective_iterations=iterations,checks=(iterations+9)//10,stop_reason='max_iter',
        reached_max_iter=True,stop_max=stop_max,stalled_checks=0,
        initialization_and_dispatch_s=.01,propagation_s=1.,postprocessing_s=.02,fit_predict_s=1.03,
        backend=backend,backend_identity={'numpy':'pcc_numpy.pcc_propagate_numpy',
            'numba':'pcc_numba.pcc_propagate_numba','cython':'pcc_step.pcc_propagate'}[backend],
        mode='early_stopping',max_iter=iterations,stopped_before_cap=False,
        propagation_seed_requested=0,propagation_seed_effective=123456789 if backend=='cython' else 0,
        propagation_s_per_iteration=1./iterations,accuracy_unlabeled=.5,
        state_hashes={name:'a'*64 for name in ('predictions','dominance','positions','strengths','distances','owndeg')})


def crash_worker(root,point):
    controller=Controller(Store(root),measure_fn=fake_measure,manifest_fn=fake_manifest)
    controller.store.fault=lambda p:os._exit(91) if p==point else None
    with patch('benchmarks.resumable.expected_backend'),patch('benchmarks.resumable.seed_backend'):
        controller.execute()


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'campaign'
        self.calls=[]
        def measured(*args,**kwargs):
            if args[3]==3:
                self.calls.append(args[2])
            return fake_measure(*args,**kwargs)
        self.controller=Controller.create(self.root,specification(True),measure_fn=measured,manifest_fn=fake_manifest)
        for name in ('expected_backend','seed_backend'):
            mock=patch('benchmarks.resumable.'+name)
            mock.start()
            self.addCleanup(mock.stop)

    def run_controller(self,**kwargs):
        with redirect_stdout(io.StringIO()):
            return self.controller.execute(**kwargs)

    def modify(self,path,change,rehash=False):
        envelope=json.loads(path.read_text())
        change(envelope['payload'])
        if rehash:
            envelope['sha256']=digest(envelope['payload'])
        path.write_text(json.dumps(envelope))

    def test_clean_stop_resume_and_censored_never_repeat(self):
        first=self.run_controller(max_tasks=1)
        self.assertEqual(first['status'],'partial')
        old=(self.root/'results/t0000.json').read_bytes()
        summary=self.run_controller()
        self.assertEqual(self.calls,['numba','cython','numpy'])
        self.assertEqual(summary['censored'],3)
        self.assertEqual(summary['cap_hits'],3)
        self.assertEqual(summary['status'],'complete')
        self.run_controller()
        self.assertEqual(len(self.calls),3)
        self.assertEqual(old,(self.root/'results/t0000.json').read_bytes())
        self.assertEqual(len(list((self.root/'sessions').iterdir())),2)
        self.assertEqual(len(list((self.root/'sessions').glob('*/warmups/*.json'))),6)

    def test_successful_early_stop_never_repeated(self):
        calls=[]
        def early(*args,**kwargs):
            row=fake_measure(*args,**kwargs)
            if args[3]==5000:
                calls.append(args[2])
                n=10*row['stop_max']+1
                row.update(effective_iterations=n,checks=(n+9)//10,stalled_checks=row['stop_max']+1,
                           stop_reason='early_stop',reached_max_iter=False,stopped_before_cap=True,
                           propagation_s_per_iteration=1./n)
            return row
        other=Controller.create(Path(self.temp.name)/'early',specification(True,5000),
                                measure_fn=early,manifest_fn=fake_manifest)
        with redirect_stdout(io.StringIO()):
            result=other.execute()
            other.execute()
        self.assertEqual(len(calls),3)
        self.assertEqual(result['censored'],0)
        self.assertEqual(result['cap_hits'],0)

    def test_wrong_binary_dependencies_source_and_fixture_metadata_rejected(self):
        for area in ('source','binary','dependencies'):
            with self.subTest(area=area):
                changed=fake_manifest()
                if area=='source':
                    changed['repository']['source_hashes']['test-source']='different'
                elif area=='binary':
                    changed['cython_extension']['sha256']='b'*64
                else:
                    changed['environment']['versions']['numpy']='different'
                self.controller.manifest=lambda *args:changed
                with self.assertRaises(CheckpointError):
                    self.run_controller()
        self.controller.manifest=fake_manifest
        self.run_controller(max_tasks=1)
        metadata=self.root/'fixtures/checkpoint_synthetic-labels10/fixture.json'
        content=json.loads(metadata.read_text())
        content['case']['label_seed']=11
        metadata.write_text(json.dumps(content))
        with self.assertRaises(CheckpointError):
            self.controller.audit()
        self.assertEqual(len(self.calls),1)

    def test_status_read_only_and_recomputed_pin_corruption_refused(self):
        self.run_controller(max_tasks=1)
        from benchmarks.checkpoints import file_inventory
        before=file_inventory(self.root)
        state=self.controller.audit()
        self.assertEqual(state['pending'],['t0001','t0002'])
        self.assertEqual(file_inventory(self.root),before)
        path=self.root/'plan_identity.json'
        self.modify(path,lambda p:p.update(plan_sha256='b'*64),rehash=True)
        with self.assertRaises(CheckpointError):
            self.controller.audit()

    def test_partial_event_and_missing_warmup_fail_closed(self):
        self.run_controller(max_tasks=1)
        result=read_record(self.root/'results/t0000.json')
        warm=self.root/result['warmup_files'][0]
        content=warm.read_bytes()
        warm.unlink()
        with self.assertRaises(CheckpointError):
            self.controller.audit()
        warm.write_bytes(content)
        (self.root/'events'/'999999.json').write_text('partial event')
        with self.assertRaises(CheckpointError):
            self.controller.audit()

    def test_unknown_retry_task_rejected_even_with_valid_checksum(self):
        with self.controller.store.lock():
            self.controller.store.event('retry_authorized',task_id='unknown',attempt_id=None,reason='injected invalid event')
        with self.assertRaisesRegex(CheckpointError,'unknown task'):
            self.controller.audit()

    def test_before_start_resume_safe(self):
        def fail(point):
            if point=='before_started':
                raise RuntimeError('crash')
        self.controller.store.fault=fail
        with self.assertRaises(RuntimeError):
            self.run_controller()
        self.assertEqual(self.calls,[])
        self.assertEqual(self.controller.audit()['ambiguous'],[])
        self.controller.store.fault=lambda p:None
        self.run_controller()
        self.assertEqual(len(self.calls),3)
        self.assertEqual(len(list((self.root/'sessions').glob('*/warmups/*.json'))),8)

    def test_ambiguous_requires_retry_and_retains_history(self):
        self.controller.store.fault=lambda p:(_ for _ in ()).throw(RuntimeError('crash')) if p=='before_result' else None
        with self.assertRaises(RuntimeError):
            self.run_controller()
        self.assertEqual(self.calls,['numba'])
        self.controller.store.fault=lambda p:None
        with self.assertRaisesRegex(CheckpointError,'ambiguous'):
            self.run_controller()
        with self.assertRaises(CheckpointError):
            self.run_controller(retry_task='t0000')
        self.run_controller(retry_task='t0000',reason='explicit test authorization')
        self.assertEqual(self.calls,['numba','numba','cython','numpy'])
        events=self.controller.store.events()
        self.assertEqual(sum(e['kind']=='started' for e in events),4)
        self.assertEqual(sum(e['kind']=='retry_authorized' for e in events),1)
        self.assertEqual(sum(e['kind']=='failed' for e in events),1)
        with self.assertRaises(CheckpointError):
            self.run_controller(retry_task='t0000',reason='must not repeat completed')

    def test_result_before_event_reconciles_without_repeat(self):
        self.controller.store.fault=lambda p:(_ for _ in ()).throw(RuntimeError('crash')) if p=='after_result' else None
        with self.assertRaises(RuntimeError):
            self.run_controller()
        self.assertEqual(self.controller.audit()['reconcile'],['t0000'])
        self.controller.store.fault=lambda p:None
        self.run_controller()
        self.assertEqual(self.calls,['numba','cython','numpy'])
        self.assertTrue(any(e['kind']=='reconciled' for e in self.controller.store.events()))

    def test_exclusive_lock_and_live_recovery_refused(self):
        with self.controller.store.lock():
            other=Store(self.root)
            with self.assertRaises(FileExistsError):
                with other.lock():
                    pass
            with self.assertRaises(CheckpointError):
                other.recover_lock(self.controller.store.token,'cannot steal live writer')
            with self.assertRaises(CheckpointError):
                other.event('bad')

    def test_result_checksum_corruption_stops_before_fit(self):
        self.run_controller(max_tasks=1)
        self.modify(self.root/'results/t0000.json',lambda p:p['row'].update(accuracy_unlabeled=.9))
        with self.assertRaises(CheckpointError):
            self.run_controller()
        self.assertEqual(self.calls,['numba'])

    def test_semantic_corruption_even_with_recomputed_checksum(self):
        self.run_controller(max_tasks=1)
        path=self.root/'results/t0000.json'
        original=path.read_bytes()
        for changes in ({'effective_iterations':2},{'backend_identity':'fallback'},
                        {'accuracy_unlabeled':2},{'propagation_seed_effective':10},
                        {'execution_order':['numpy','numba','cython']},
                        {'stalled_checks':-1},{'state_hashes':{}},{'fit_predict_s':99}):
            with self.subTest(changes=changes):
                path.write_bytes(original)
                self.modify(path,lambda p:p['row'].update(changes),rehash=True)
                with self.assertRaises(CheckpointError):
                    self.controller.audit()
        path.write_bytes(original)

    def test_plan_corruption_and_source_environment_mismatch(self):
        path=self.root/'plan.json'
        original=path.read_bytes()
        self.modify(path,lambda p:p.update(max_iter=9),rehash=True)
        with self.assertRaisesRegex(CheckpointError,'immutable'):
            self.run_controller()
        self.modify(path,lambda p:p['tasks'][0].update(backend='numpy'),rehash=True)
        with self.assertRaises(CheckpointError):
            self.run_controller()
        path.write_bytes(original)
        changed=fake_manifest()
        changed['environment']['cpu']='other'
        self.controller.manifest=lambda *args:changed
        with self.assertRaises(CheckpointError):
            self.run_controller()
        self.assertEqual(self.calls,[])

    def test_fixture_corruption_stops_and_partial_requires_explicit_retry(self):
        self.controller.store.fault=lambda p:(_ for _ in ()).throw(RuntimeError('crash')) if p=='fixture_written' else None
        with self.assertRaises(RuntimeError):
            self.run_controller()
        partial=list((self.root/'staging').glob('fixture-*'))[0]
        original=(partial/'arrays.npz').read_bytes()
        self.controller.store.fault=lambda p:None
        with self.assertRaisesRegex(CheckpointError,'partial fixture'):
            self.run_controller()
        self.run_controller(retry_fixture='checkpoint_synthetic-labels10',reason='explicit fixture rebuild')
        self.assertEqual(original,(partial/'arrays.npz').read_bytes())
        fixture=self.root/'fixtures/checkpoint_synthetic-labels10/arrays.npz'
        fixture.write_bytes(b'corrupt')
        with self.assertRaises(CheckpointError):
            self.run_controller()
        self.assertEqual(len(self.calls),3)

    def test_sealed_fixture_recovers_without_repreparation(self):
        self.controller.store.fault=lambda p:(_ for _ in ()).throw(RuntimeError('crash')) if p=='fixture_sealed' else None
        with self.assertRaises(RuntimeError):
            self.run_controller()
        self.controller.store.fault=lambda p:None
        with patch.object(self.controller.preparer,'prepare',side_effect=RuntimeError('must reuse sealed fixture')):
            self.run_controller()
        self.assertEqual(len(self.calls),3)

    def test_duplicate_unknown_results_and_broken_event_chain(self):
        self.run_controller(max_tasks=1)
        duplicate=self.root/'results/t0000-copy.json'
        duplicate.write_bytes((self.root/'results/t0000.json').read_bytes())
        with self.assertRaises(CheckpointError):
            self.controller.audit()
        duplicate.unlink()  # Test-only removal of injected corruption.
        path=self.root/'events/000000.json'
        self.modify(path,lambda p:p.update(previous_sha256='bad'),rehash=True)
        with self.assertRaises(CheckpointError):
            self.controller.audit()

    def test_summary_interruption_preserves_generation_and_never_repeats(self):
        self.controller.store.fault=lambda p:(_ for _ in ()).throw(RuntimeError('crash')) if p=='summary_written' else None
        with self.assertRaises(RuntimeError):
            self.run_controller()
        old=list((self.root/'summaries').iterdir())[0]
        original=(old/'summary.json').read_bytes()
        self.controller.store.fault=lambda p:None
        self.run_controller()
        self.assertEqual(len(self.calls),3)
        self.assertEqual(original,(old/'summary.json').read_bytes())
        self.assertEqual(len(list((self.root/'summaries').iterdir())),2)

    def test_corrupt_summary_explicit_rebuild_without_fits(self):
        self.run_controller()
        generation=list((self.root/'summaries').iterdir())[0]
        path=generation/'summary.json'
        path.write_bytes(b'broken summary')
        with self.assertRaises(CheckpointError):
            self.run_controller()
        with self.assertRaises(CheckpointError):
            self.controller.summarize()
        summary=self.controller.summarize('explicit repair of derived view')
        self.assertEqual(summary['status'],'complete')
        self.assertEqual(path.read_bytes(),b'broken summary')
        self.assertEqual(len(self.calls),3)
        self.controller.audit()

    def test_warmup_corruption_missing_result_and_bad_lock_token(self):
        self.run_controller(max_tasks=1)
        result_path=self.root/'results/t0000.json'
        result=read_record(result_path)
        warm_path=self.root/result['warmup_files'][0]
        original=warm_path.read_bytes()
        self.modify(warm_path,lambda p:p['row'].update(backend_identity='wrong'),rehash=True)
        with self.assertRaises(CheckpointError):
            self.controller.audit()
        warm_path.write_bytes(original)
        result_path.unlink()
        with self.assertRaises(CheckpointError):
            self.controller.audit()
        with self.controller.store.lock():
            with self.assertRaises(CheckpointError):
                self.controller.store.recover_lock('b'*32,'wrong lock token')

    def test_no_overwrite_and_nonfinite_or_duplicate_json_refused(self):
        with self.controller.store.lock():
            with self.assertRaises(FileExistsError):
                self.controller.store.write(self.root/'plan.json',{'bad':True})
            with self.assertRaises(ValueError):
                self.controller.store.write(self.root/'results/bad.json',{'x':float('nan')})
        path=self.root/'results/bad.json'
        path.write_text('{"schema":1,"schema":1,"payload":{},"sha256":"bad"}')
        with self.assertRaises(CheckpointError):
            read_record(path)

    def test_cli_defaults_read_only_and_requires_execution_optin(self):
        with redirect_stdout(io.StringIO()) as stream:
            main([])
        self.assertEqual(len(json.loads(stream.getvalue())['tasks']),48)
        for argv in (['--action','start','--root',str(self.root)],
                     ['--action','resume','--root',str(self.root),'--retry-task','t0000']):
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(argv)


class RealProcessCrashTests(unittest.TestCase):
    def test_actual_process_exit_boundaries(self):
        for point,ambiguous in (('before_started',False),('after_started',True),
                               ('result_record_flushed',True),('result_record_published',False),
                               ('after_completed',False),('summary_sealed',False)):
            with self.subTest(point=point),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'campaign'
                controller=Controller.create(root,specification(True),measure_fn=fake_measure,manifest_fn=fake_manifest)
                code="from tests.test_resumable_benchmarks import crash_worker; import sys; crash_worker(sys.argv[1],sys.argv[2])"
                child=subprocess.run([sys.executable,'-c',code,str(root),point],capture_output=True,text=True)
                self.assertEqual(child.returncode,91,child.stderr)
                self.assertTrue((root/'writer.lock').exists())
                with self.assertRaises(FileExistsError):
                    with controller.store.lock():
                        pass
                token=read_record(root/'writer.lock')['token']
                controller.store.recover_lock(token,'explicit test recovery of exited child')
                state=controller.audit()
                self.assertEqual(bool(state['ambiguous']),ambiguous)
                if ambiguous:
                    with self.assertRaises(CheckpointError),patch('benchmarks.resumable.expected_backend'),patch('benchmarks.resumable.seed_backend'):
                        controller.execute()
                else:
                    previous={p.name:p.read_bytes() for p in (root/'results').iterdir()}
                    with patch('benchmarks.resumable.expected_backend'),patch('benchmarks.resumable.seed_backend'),redirect_stdout(io.StringIO()):
                        controller.execute()
                    self.assertEqual(len(controller.audit()['results']),3)
                    self.assertTrue(all((root/'results'/n).read_bytes()==b for n,b in previous.items()))


if __name__=='__main__':
    unittest.main()
