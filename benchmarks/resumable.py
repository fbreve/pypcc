"""Optional whole-fit checkpoint controller; defaults to a read-only plan.

python -m benchmarks.resumable --help. No production PCC code is modified.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .checkpoints import (CheckpointError, Store, digest, durable_file, file_inventory,
                          read_record, sync_directory, utc)
from .fixtures import (Case, REAL_DATASETS, FixturePreparer, array_hash, save_fixture,
                       select_labels, validate_historical_labels)
from .historical import FROZEN, ENTRIES, fingerprint
from .run import execution_order
from .runner import PARAMETERS, TIMINGS, environment_manifest, expected_backend, measure, seed_backend

ORDER = ('pendigits', 'pageblocks', 'yeast', 'glass', 'segment', 'vehicle', 'iris',
         'balancescale', 'banknote', 'diabetes', 'bloodtransfusion', 'breast_cancer',
         'ionosphere', 'spambase', 'sonar', 'qsarbiodeg')
BACKENDS = ('numba', 'cython', 'numpy')


def identity(manifest):
    return dict(source_hashes=manifest['repository']['source_hashes'],
                environment=manifest['environment'], cython_extension=manifest['cython_extension'])


def specification(synthetic=False, iterations=3):
    if synthetic:
        cases = [Case('checkpoint_synthetic', 30, 2, 2, synthetic=True, k_nn=3, label_seed=10)]
        budget = iterations
    else:
        cases = [Case(name, *REAL_DATASETS[name][:3], label_seed=10) for name in ORDER]
        budget = 500000
    for case in cases:
        case.validate()
    if type(budget) is not int or budget < 1:
        raise ValueError('positive iteration cap required')
    tasks = [dict(task_id=f't{3*i+j:04d}', case_id=case.case_id, backend=backend,
                  measurement_index=0, order_index=j)
             for i,case in enumerate(cases) for j,backend in enumerate(BACKENDS)]
    return dict(schema=1, protocol='synthetic_validation' if synthetic else 'remaining_historical_seed10',
                cases=[asdict(c) for c in cases], tasks=tasks, max_iter=budget,
                parameters=dict(PARAMETERS), propagation_seed=0, warmup_iterations=2,
                warmups_per_task=2, backend_order=list(BACKENDS))


def full_specification(synthetic=False, iterations=3):
    """Freeze the entire matrix once. Stage selection never regenerates IDs."""
    cases=[]
    for seed in range(10,20):
        if synthetic:
            cases.append(Case('checkpoint_synthetic',30,2,2,synthetic=True,k_nn=3,label_seed=seed))
        else:
            cases.extend(Case(name,*REAL_DATASETS[name][:3],label_seed=seed) for name in ENTRIES)
    tasks=[]
    for case_index,case in enumerate(cases):
        case.validate()
        group_ids={}
        for repeat in range(6):
            order=execution_order(['numpy','numba','cython'],case_index,repeat,'permutations',0)
            for position,backend in enumerate(order):
                tid=f't{len(tasks):04d}'
                group_ids.setdefault(backend,tid)
                tasks.append(dict(task_id=tid,case_id=case.case_id,backend=backend,
                    measurement_index=repeat,order_index=position,execution_order=order,
                    warmup_task_id=group_ids[backend]))
    budget=iterations if synthetic else 500000
    if type(budget) is not int or budget<1:
        raise ValueError('positive iteration cap required')
    return dict(schema=1,protocol='synthetic_full_validation' if synthetic else 'historical_early_stopping_full_v1',
        cases=[asdict(c) for c in cases],tasks=tasks,max_iter=budget,parameters=dict(PARAMETERS),
        propagation_seed=0,warmup_iterations=2,warmups_per_group=2,repetitions=6,
        order_policy='permutations',order_seed=0,label_seeds=list(range(10,20)))


def validate_spec(plan):
    if plan['protocol'] in ('historical_early_stopping_full_v1','synthetic_full_validation'):
        expected=full_specification(plan['protocol']=='synthetic_full_validation',plan['max_iter'])
    elif plan['protocol'] in ('remaining_historical_seed10','synthetic_validation'):
        expected = specification(plan['protocol']=='synthetic_validation', plan['max_iter'])
    else:
        raise CheckpointError('unknown campaign protocol')
    for key,value in expected.items():
        if plan.get(key) != value:
            raise CheckpointError(f'campaign specification mismatch: {key}')
    if set(plan) != set(expected) | {'campaign_id','identity','manifest','historical_cache'}:
        raise CheckpointError('unexpected campaign plan fields')
    if plan['identity'] != identity(plan['manifest']):
        raise CheckpointError('manifest identity mismatch')


def validate_row(row, case, backend, budget, plan):
    n = row['effective_iterations']
    if type(n) is not int or not 0 < n <= budget:
        raise CheckpointError('invalid iteration count')
    if any(type(row[f]) is not int or row[f]<0 for f in ('stalled_checks','stop_max','checks')):
        raise CheckpointError('invalid stopping counter type/range')
    if row['stalled_checks']>row['checks'] or row['stop_max']<1:
        raise CheckpointError('inconsistent stopping counters')
    stopped = row['stalled_checks'] > row['stop_max']
    # Exact existing pcc.py patience calculation, including Python round/ties.
    stop_max = max(1, round((case.nodes/(case.particles*case.k_nn))*round(plan['parameters']['es_chk']*.1)))
    if (row['mode'] != 'early_stopping' or row['backend'] != backend or row['max_iter'] != budget
            or row['stop_max'] != stop_max or row['checks'] != (n+9)//10
            or row['stop_reason'] != ('early_stop' if stopped else 'max_iter')
            or row['reached_max_iter'] != (n==budget) or row['stopped_before_cap'] != (n<budget)
            or (stopped and (n-1)%10) or (not stopped and n!=budget)):
        raise CheckpointError('invalid stopping metrics/protocol')
    expected = {'numpy':'pcc_numpy.pcc_propagate_numpy', 'numba':'pcc_numba.pcc_propagate_numba',
                'cython':'pcc_step.pcc_propagate'}[backend]
    if (row['backend_identity'] != expected or row['propagation_seed_requested'] != 0
            or row['propagation_seed_effective'] != (123456789 if backend=='cython' else 0)):
        raise CheckpointError('backend/RNG mismatch')
    if not math.isfinite(row['accuracy_unlabeled']) or not 0 <= row['accuracy_unlabeled'] <= 1:
        raise CheckpointError('invalid accuracy')
    if any(not math.isfinite(row[f]) or row[f] <= 0 for f in TIMINGS):
        raise CheckpointError('invalid timing')
    if abs(row['fit_predict_s']-sum(row[f] for f in TIMINGS[:-1])) > 1e-8:
        raise CheckpointError('timing boundaries disagree')
    if not math.isclose(row['propagation_s_per_iteration'],row['propagation_s']/n,rel_tol=1e-12):
        raise CheckpointError('invalid per-iteration timing')
    hashes = row['state_hashes']
    if set(hashes) != {'predictions','dominance','positions','strengths','distances','owndeg'}:
        raise CheckpointError('invalid state hash inventory')
    try:
        if any(not isinstance(h,str) or len(h)!=64 or int(h,16)<0 for h in hashes.values()):
            raise ValueError('hash')
    except ValueError as exc:
        raise CheckpointError('invalid state hash') from exc


class Controller:
    def __init__(self, store, measure_fn=None, manifest_fn=None, preparer=None):
        self.store = store
        self.measure = measure_fn or measure
        self.manifest = manifest_fn or environment_manifest
        self.preparer = preparer or FixturePreparer()
        self.fixture_cache = {}
        self.reference_states = {}

    @classmethod
    def create(cls, root, spec, historical_cache=None, **kwargs):
        manifest_fn = kwargs.get('manifest_fn') or environment_manifest
        config=spec if 'repetitions' not in spec else dict(protocol=spec['protocol'],
            specification_sha256=digest(spec),task_count=len(spec['tasks']))
        manifest = manifest_fn(config, list(BACKENDS))
        plan = dict(spec, campaign_id=uuid.uuid4().hex, identity=identity(manifest), manifest=manifest,
                    historical_cache=str(Path(historical_cache).resolve()) if historical_cache else None)
        validate_spec(plan)
        return cls(Store.create(root, plan), **kwargs)

    def verify_environment(self, plan):
        validate_spec(plan)
        current = self.manifest({}, list(BACKENDS))
        if identity(current) != plan['identity']:
            raise CheckpointError('source, binary, dependencies, hardware or environment changed')
        return current

    def fixture(self, case, plan, directory=None):
        directory = directory or self.store.root/'fixtures'/case.case_id
        seal = read_record(directory/'seal.json')
        actual = file_inventory(directory)
        actual.pop('seal.json',None)
        if seal['files'] != actual or set(actual) != {'fixture.json','arrays.npz'}:
            raise CheckpointError('fixture file checksum/inventory mismatch')
        if seal['plan_sha256'] != digest(plan) or seal['case'] != asdict(case):
            raise CheckpointError('fixture identity mismatch')
        metadata = json.loads((directory/'fixture.json').read_text(encoding='utf-8'))
        if metadata['case'] != asdict(case):
            raise CheckpointError('fixture case changed')
        with np.load(directory/'arrays.npz', allow_pickle=False) as stored:
            arrays = {name:stored[name] for name in stored.files}
        hashes = {name:array_hash(value) for name,value in arrays.items()}
        if hashes != metadata['hashes'] or hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest() != metadata['fixture_sha256']:
            raise CheckpointError('fixture arrays/identity changed')
        if (arrays['raw'].shape != (case.nodes,case.features) or arrays['x'].shape != (case.nodes,case.features)
                or arrays['truth'].shape != (case.nodes,) or len(np.unique(arrays['truth']))!=case.classes
                or not np.isfinite(arrays['raw']).all() or not np.isfinite(arrays['x']).all()
                or not np.array_equal(select_labels(arrays['truth'],case),arrays['labels'])):
            raise CheckpointError('fixture dimensions/data/labels invalid')
        if not case.synthetic:
            reference = FROZEN[case.name]
            for actual_hash, expected_hash in (
                (fingerprint(arrays['raw'],arrays['truth']),reference['raw_data_fingerprint']),
                (fingerprint(arrays['x'],arrays['truth']),reference['processed_data_fingerprint']),
                (fingerprint(arrays['neighbors'],arrays['degrees']),reference['graph_fingerprint'])):
                if actual_hash != expected_hash:
                    raise CheckpointError('frozen fixture fingerprint mismatch')
            validate_historical_labels(case,arrays['labels'],metadata['source'])
            if not metadata['source']['label_fingerprint_matches']:
                raise CheckpointError('unvalidated historical label mask')
            if any(not f.get('matches') or f['actual'] != f['expected']
                   for f in metadata['source']['frozen_fingerprints'].values()):
                raise CheckpointError('invalid original source fingerprints')
        return arrays, metadata

    def validate_result(self, result, task, plan):
        case = next(Case(**c) for c in plan['cases'] if Case(**c).case_id == task['case_id'])
        if case.case_id not in self.fixture_cache:
            self.fixture_cache[case.case_id] = self.fixture(case, plan)
        _, metadata = self.fixture_cache[case.case_id]
        row = result['row']
        if (result['fixture_sha256'] != metadata['fixture_sha256'] or row['fixture_sha256']!=result['fixture_sha256']
                or row['case_id']!=case.case_id or row['task_id']!=task['task_id']
                or row['session_id']!=result['session_id'] or row['order_index']!=task['order_index']
                or row['execution_order']!=task.get('execution_order',list(BACKENDS))
                or row['labeled_nodes']!=case.particles):
            raise CheckpointError('result task/fixture/order mismatch')
        validate_row(row,case,task['backend'],plan['max_iter'],plan)
        if 'repetitions' in plan:
            if row.get('repeat')!=task['measurement_index']:
                raise CheckpointError('result repetition mismatch')
            key=(case.case_id,task['backend'])
            value=(row['state_hashes'],row['effective_iterations'],row['stop_reason'],row['accuracy_unlabeled'])
            if self.reference_states.setdefault(key,value)!=value:
                raise CheckpointError('nonrepeatable within-backend state/stopping/accuracy')
        if len(result['warmup_files']) != 2:
            raise CheckpointError('require two session-specific warm-ups')
        states=[]
        for index,name in enumerate(result['warmup_files']):
            expected = f"sessions/{result['session_id']}/warmups/{task.get('warmup_task_id',task['task_id'])}-{index}.json"
            if name != expected:
                raise CheckpointError('invalid warm-up reference path')
            warm = read_record(self.store.root/name)
            if warm['fixture_sha256']!=result['fixture_sha256'] or warm['session_id']!=result['session_id']:
                raise CheckpointError('warm-up identity mismatch')
            validate_row(warm['row'],case,task['backend'],2,plan)
            states.append(warm['row']['state_hashes'])
        if states[0]!=states[1]:
            raise CheckpointError('warm-up states not repeatable')

    def audit(self, check_environment=True, tolerate_summaries=False):
        self.fixture_cache = {}
        self.reference_states = {}
        if any(p.is_symlink() for p in self.store.root.rglob('*')):
            raise CheckpointError('campaign symlinks refused')
        plan = self.store.plan()
        validate_spec(plan)
        if check_environment:
            self.verify_environment(plan)
        expected_cases={Case(**c).case_id:Case(**c) for c in plan['cases']}
        for directory in (self.store.root/'fixtures').iterdir():
            if directory.name not in expected_cases or not directory.is_dir():
                raise CheckpointError('unknown fixture directory')
            self.fixture_cache[directory.name]=self.fixture(expected_cases[directory.name],plan)
        state=self.store.audit(self.validate_result,tolerate_summaries)
        tasks={t['task_id']:t for t in plan['tasks']}
        for directory in (self.store.root/'sessions').iterdir():
            session=read_record(directory/'session.json')
            if session['session_id']!=directory.name or session['identity']!=plan['identity']:
                raise CheckpointError('orphan/session provenance mismatch')
            for warm_path in (directory/'warmups').iterdir():
                task_id, index=warm_path.stem.rsplit('-',1)
                if task_id not in tasks or index not in ('0','1') or warm_path.suffix!='.json':
                    raise CheckpointError('unknown warm-up record')
                task=tasks[task_id]
                case=next(Case(**c) for c in plan['cases'] if Case(**c).case_id==task['case_id'])
                warm=read_record(warm_path)
                if case.case_id not in self.fixture_cache or warm['fixture_sha256']!=self.fixture_cache[case.case_id][1]['fixture_sha256'] or warm['session_id']!=directory.name:
                    raise CheckpointError('orphan warm-up identity mismatch')
                validate_row(warm['row'],case,task['backend'],2,plan)
            if (directory/'seed_helper.json').exists():
                latency=read_record(directory/'seed_helper.json')['first_call_s']
                if not math.isfinite(latency) or latency<0:
                    raise CheckpointError('invalid seed helper latency')
        return state

    def partial_fixtures(self, case):
        return sorted((self.store.root/'staging').glob(f'fixture-{case.case_id}-*'))

    def prepare(self, case, plan):
        target=self.store.root/'fixtures'/case.case_id
        if target.exists():
            return self.fixture(case,plan)
        ignored = {}
        for event in self.store.events():
            if event['kind']=='fixture_retry':
                ignored.update(event['preserved'])
        partials=[p for p in self.partial_fixtures(case) if p.name not in ignored]
        if partials:
            # Fully sealed fixture publication interrupted: reuse, never rebuild.
            if len(partials)==1 and (partials[0]/'seal.json').exists():
                arrays,metadata=self.fixture(case,plan,partials[0])
                os.rename(partials[0],target)
                sync_directory(target.parent)
                return arrays,metadata
            raise CheckpointError(f'partial fixture {case.case_id}; explicit --retry-fixture and reason required')
        if 'repetitions' in plan and case.label_seed!=10:
            anchor=Case(**dict(asdict(case),label_seed=10))
            arrays,metadata=self.fixture(anchor,plan)
            before=time.perf_counter()
            labels=select_labels(arrays['truth'],case)
            elapsed=time.perf_counter()-before
            metadata=copy.deepcopy(metadata)
            validate_historical_labels(case,labels,metadata['source'])
            arrays=dict(arrays,labels=labels)
            hashes={name:array_hash(value) for name,value in arrays.items()}
            metadata.update(case=asdict(case),hashes=hashes,
                fixture_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                preparation=dict(reused=True,source_case_id=anchor.case_id),
                timings=dict(dataset_load_s=0.,graph_build_s=0.,preprocessing_and_labels_s=elapsed))
        else:
            arrays,metadata=self.preparer.prepare(case,historical_cache=plan['historical_cache'])
        stage=self.store.root/'staging'/f'fixture-{case.case_id}-{uuid.uuid4().hex}'
        self.store.fault('before_fixture_write')
        save_fixture(stage,arrays,metadata)
        for file in stage.iterdir():
            # Windows _commit needs a writable descriptor, even after save_fixture
            # has closed/flushed the file. Reopen without changing its bytes.
            with file.open('r+b') as stream:
                stream.flush()
                os.fsync(stream.fileno())
        self.store.fault('fixture_written')
        self.store.write(stage/'seal.json',dict(case=asdict(case),plan_sha256=digest(plan),files=file_inventory(stage)))
        self.store.fault('fixture_sealed')
        self.fixture(case,plan,stage)
        os.rename(stage,target)
        sync_directory(target.parent)
        self.store.fault('fixture_published')
        return arrays,metadata

    def retry_fixture(self, case_id, reason):
        state=self.audit()
        case=next((Case(**c) for c in state['plan']['cases'] if Case(**c).case_id==case_id),None)
        if not case or not reason.strip() or (self.store.root/'fixtures'/case_id).exists():
            raise CheckpointError('explicit partial-fixture retry only')
        partials=self.partial_fixtures(case)
        if not partials:
            raise CheckpointError('no partial fixture to retry')
        self.store.event('fixture_retry',case_id=case_id,reason=reason,
                         preserved={p.name:file_inventory(p) for p in partials})

    def summarize(self, reason=None):
        with self.store.lock():
            state=self.audit(tolerate_summaries=bool(reason))
            if state['ambiguous']:
                raise CheckpointError('ambiguous tasks must be resolved before summary generation')
            if state['summary_errors']:
                self.store.retire_summaries(state,reason)
            self.store.reconcile(state)
            return self.store.snapshot(state)

    def execute(self, retry_task=None, retry_fixture=None, reason=None, max_tasks=None, stage_seed=None):
        with self.store.lock():
            state=self.audit()
            full='repetitions' in state['plan']
            if full:
                if stage_seed not in range(10,20):
                    raise CheckpointError('full matrix execution requires explicit --stage-seed 10..19')
                earlier={Case(**c).case_id for c in state['plan']['cases'] if c['label_seed']<stage_seed}
                pending_ids=set(state['pending'])
                if any(t['case_id'] in earlier for t in state['plan']['tasks'] if t['task_id'] in pending_ids):
                    raise CheckpointError('previous seed stages are incomplete')
                selected={Case(**c).case_id for c in state['plan']['cases'] if c['label_seed']==stage_seed}
                eligible={t['task_id'] for t in state['plan']['tasks'] if t['case_id'] in selected}
                if retry_task and retry_task not in eligible:
                    raise CheckpointError('retry task outside selected stage')
                if retry_fixture and retry_fixture not in selected:
                    raise CheckpointError('retry fixture outside selected stage')
            else:
                if stage_seed is not None:
                    raise CheckpointError('stage selection requires full matrix protocol')
                eligible={t['task_id'] for t in state['plan']['tasks']}
            self.store.reconcile(state)
            if retry_task:
                self.store.authorize_retry(retry_task,reason or '',self.validate_result)
            if retry_fixture:
                self.retry_fixture(retry_fixture,reason or '')
            state=self.audit()
            if state['ambiguous']:
                raise CheckpointError(f"ambiguous attempts require explicit retry: {state['ambiguous']}")
            pending=set(state['pending']) & eligible
            if not pending:
                return self.store.snapshot(state)
            plan=state['plan']
            plan_sha=digest(plan)
            for backend in BACKENDS:
                expected_backend(backend)
            session_id=uuid.uuid4().hex
            directory=self.store.root/'sessions'/session_id
            (directory/'warmups').mkdir(parents=True)
            current=self.verify_environment(plan)
            session=dict(session_id=session_id,identity=plan['identity'],manifest=current,started_utc=utc(),stage_seed=stage_seed)
            self.store.write(directory/'session.json',session)
            self.store.event('session_start',session_id=session_id)
            start=time.perf_counter()
            completed=0
            warmed={}
            status='failed'
            try:
                before=time.perf_counter()
                seed_backend('numba',0)
                self.store.write(directory/'seed_helper.json',dict(first_call_s=time.perf_counter()-before))
                for task in plan['tasks']:
                    tid=task['task_id']
                    if tid not in pending:
                        continue
                    case=next(Case(**c) for c in plan['cases'] if Case(**c).case_id==task['case_id'])
                    if case.case_id not in self.fixture_cache:
                        self.fixture_cache[case.case_id]=self.prepare(case,plan)
                    arrays,meta=self.fixture_cache[case.case_id]
                    warm_id=task.get('warmup_task_id',tid)
                    if warm_id not in warmed:
                        warms=[]
                        for index in range(2):
                            warm=self.measure(arrays,case,task['backend'],2,True,0,steady_state=False)
                            validate_row(warm,case,task['backend'],2,plan)
                            relative=f'sessions/{session_id}/warmups/{warm_id}-{index}.json'
                            self.store.write(self.store.root/relative,dict(session_id=session_id,
                                fixture_sha256=meta['fixture_sha256'],row=warm))
                            warms.append(relative)
                        warmed[warm_id]=warms
                    warms=warmed[warm_id]
                    aid=uuid.uuid4().hex
                    self.store.fault('before_started')
                    self.store.event('started',task_id=tid,attempt_id=aid,session_id=session_id,warmup_files=warms)
                    self.store.fault('after_started')
                    entered=utc()
                    try:
                        row=self.measure(arrays,case,task['backend'],plan['max_iter'],True,0)
                        row.update(task_id=tid,case_id=case.case_id,session_id=session_id,
                            fixture_sha256=meta['fixture_sha256'],labeled_nodes=case.particles,
                            order_index=task['order_index'],execution_order=task.get('execution_order',list(BACKENDS)),
                            call_started_utc=entered,call_completed_utc=utc())
                        if full:
                            row['repeat']=task['measurement_index']
                        payload=dict(task_id=tid,attempt_id=aid,session_id=session_id,plan_sha256=plan_sha,
                            fixture_sha256=meta['fixture_sha256'],warmup_files=warms,row=row)
                        self.validate_result(payload,task,plan)
                        self.store.result(payload)
                    except Exception as exc:
                        self.store.event('failed',task_id=tid,attempt_id=aid,error=repr(exc))
                        raise
                    completed+=1
                    print(f"{case.case_id} {task['backend']}: {row['effective_iterations']} {row['stop_reason']}",flush=True)
                    if max_tasks is not None and completed>=max_tasks:
                        break
                state=self.audit()
                self.store.reconcile(state)
                summary=self.store.snapshot(state)
                status=summary['status']
                return summary
            finally:
                self.store.write(directory/'end.json',dict(session_id=session_id,ended_utc=utc(),
                    elapsed_s=time.perf_counter()-start,status=status,measured_tasks_completed=completed))
                self.store.event('session_end',session_id=session_id)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--action',choices=('plan','start','resume','status','unlock','summarize'),default='plan')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--historical-cache',type=Path)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--full-campaign',action='store_true',help='freeze all 3600 tasks; execution still requires an explicit stage')
    parser.add_argument('--stage-seed',type=int,choices=range(10,20))
    parser.add_argument('--iterations',type=int,default=3,help='synthetic validation only; historical cap is frozen')
    parser.add_argument('--retry-task')
    parser.add_argument('--retry-fixture')
    parser.add_argument('--reason')
    parser.add_argument('--lock-token')
    parser.add_argument('--max-tasks',type=int,help='stop cleanly after this many pending tasks; never changes their fit cap')
    args=parser.parse_args(argv)
    if args.action not in ('start','plan') and (args.synthetic or args.full_campaign or args.iterations!=3 or args.historical_cache):
        parser.error('resume/status use immutable saved configuration')
    if not args.synthetic and args.iterations!=3:
        parser.error('historical cap is frozen at 500000; --iterations is synthetic-only')
    if (args.retry_task or args.retry_fixture) and (args.action!='resume' or not args.reason):
        parser.error('explicit retry requires resume and --reason')
    if args.action=='plan':
        spec=full_specification(args.synthetic,args.iterations) if args.full_campaign else specification(args.synthetic,args.iterations)
        print(json.dumps(spec,indent=2))
        return
    if not args.root:
        parser.error('--root required')
    if args.action!='status' and not args.execute:
        parser.error('mutations require --execute')
    if args.max_tasks is not None and args.max_tasks<1:
        parser.error('--max-tasks must be positive')
    if args.action=='start':
        if args.retry_task or args.retry_fixture:
            parser.error('retry requires resume')
        if args.full_campaign and args.stage_seed is None:
            parser.error('full campaign start requires --stage-seed; no implicit later stages')
        spec=full_specification(args.synthetic,args.iterations) if args.full_campaign else specification(args.synthetic,args.iterations)
        controller=Controller.create(args.root,spec,args.historical_cache)
    else:
        controller=Controller(Store(args.root))
    if args.action=='unlock':
        controller.store.recover_lock(args.lock_token,args.reason or '')
        return
    if args.action=='status':
        state=controller.audit()
        print(json.dumps({k:state[k] for k in ('pending','ambiguous','authorized_retries','reconcile')},indent=2))
        return
    if args.action=='summarize':
        summary=controller.summarize(args.reason)
    else:
        summary=controller.execute(args.retry_task,args.retry_fixture,args.reason,args.max_tasks,args.stage_seed)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
