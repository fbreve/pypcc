"""Reviewed, resumable exploratory full-snapshot pilot. No real fits in preflight."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import uuid

from benchmarks.checkpoints import CheckpointError, Store, canonical, digest, durable_file, file_inventory, read_record
from benchmarks.fixtures import Case, FixturePreparer, file_hash, save_fixture
from benchmarks.historical import FROZEN
from pcc import ParticleCompetitionAndCooperation as PCC
from . import convergence_pilot as b
from . import convergence_stage_c as c
from .convergence_trajectories import observe_fit, metadata
from .mechanistic_snapshots import SnapshotConfig, estimates, check_budget, write_snapshots, read_snapshots

MILESTONES=(0,10,25,50,100,250,500,1000,2000,5000,10000,20000,40000,60000,80000,100000)
MAX_OBSERVATIONS=32


def specification(synthetic=False,iterations=7):
    spec=b.specification(synthetic,iterations)
    if synthetic:
        cases=[Case(name,n,f,cl,synthetic=True,k_nn=3,label_seed=seed)
               for name,n,f,cl in (('mechanism_a',24,2,2),('mechanism_b',30,3,3)) for seed in (10,13)]
    else:
        cases=[Case(name,FROZEN[name]['nodes'],FROZEN[name]['processed_features'],FROZEN[name]['classes'],label_seed=seed)
               for name in ('pageblocks','spambase') for seed in (10,13)]
        spec['milestones']=list(MILESTONES)
    spec.update(protocol='synthetic-mechanistic-pilot-v1' if synthetic else 'historical-mechanistic-pilot-v1',
                cases=[asdict(x) for x in cases],tasks=[dict(task_id=f't{3*i+j:04d}',case_id=case.case_id,backend=backend)
                for i,case in enumerate(cases) for j,backend in enumerate(b.BACKENDS)],
                max_observations=MAX_OBSERVATIONS,snapshot_config=asdict(SnapshotConfig()))
    return spec


def identity():
    result=b.execution_identity()
    root=b.ROOT
    result['mechanistic_sources']={name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in
        ('experiments/mechanistic_snapshots.py','experiments/mechanistic_pilot.py','docs/mechanistic_snapshots_protocol.md')}
    return result


class FrozenC1(b.Pilot):
    """Read-only audit of the completed old store, not permission to resume it."""
    def __init__(self,root):
        self.store=Store(Path(root).resolve())
        if self.store.plan().get('pilot_kind')!='convergence_stage_c':raise CheckpointError('not a frozen C1 store')

    def context(self):
        plan=self.store.plan();report=read_record(Path(plan['preflight'])/'report.json')
        c.validate_spec(report['specification'])
        if (report['specification']['part']!='c1' or report['specification']['synthetic'] or report!=plan['report']
                or digest(report)!=plan['report_sha256'] or plan['tasks']!=report['specification']['tasks']):
            raise CheckpointError('invalid frozen C1 binding')
        c.check_fixtures(plan['preflight'],report);c.verify_dependency(report['stage_b'])
        return plan,report

    def execute(self,**kwargs):raise CheckpointError('frozen C1 audit cannot execute fits')


def references(stage_b,stage_c1,analysis,spec,fixtures,execution):
    old_b=c.FrozenStageB(stage_b);old_c=FrozenC1(stage_c1)
    bs,cs=old_b.audit(),old_c.audit()
    if (len(bs['done'])!=12 or len(cs['done'])!=48 or any(s['recoverable'] or s['ambiguous'] for s in (bs,cs))):
        raise CheckpointError('B/C1 references are not complete and unambiguous')
    analysis=Path(analysis).resolve();manifest=json.loads((analysis/'manifest.json').read_text(encoding='utf-8'))
    if manifest['files']!={k:v for k,v in file_inventory(analysis).items() if k!='manifest.json'}:
        raise CheckpointError('joint analysis manifest differs')
    integrity=json.loads((analysis/'integrity.json').read_text(encoding='utf-8'))
    if (not integrity['passed'] or integrity['trajectories']!=60 or integrity['observations']!=600
            or integrity['stage_b_sha256']!=bs['plan']['report_sha256'] or integrity['c1_sha256']!=cs['plan']['report_sha256']):
        raise CheckpointError('joint analysis does not bind B/C1')
    scientific={(r['dataset'],r['seed'],r['backend']):r for r in integrity['provenance']}
    refs={};dependencies={str(analysis):file_inventory(analysis)}
    for pilot,state in ((old_b,bs),(old_c,cs)):
        dependencies[str(pilot.store.root)]=file_inventory(pilot.store.root)
        dependencies[state['plan']['preflight']]=file_inventory(Path(state['plan']['preflight']))
    for task in spec['tasks']:
        case=next(Case(**d) for d in spec['cases'] if Case(**d).case_id==task['case_id'])
        source,state=(old_b,bs) if case.label_seed==10 else (old_c,cs)
        old=next(t for t in state['plan']['tasks'] if (t['case_id'],t['backend'])==(case.case_id,task['backend']))
        path=source.store.root/'sessions'/state['last'][old['task_id']]/'trajectory'
        data=source.validate_bundle(path,old,state['report']);meta=data['metadata']
        if (file_inventory(path)!=scientific[(case.name,case.label_seed,task['backend'])]['files']
                or fixtures[case.case_id]['fixture_sha256']!=state['report']['fixtures'][case.case_id]['fixture_sha256']
                or meta['parameters']!=spec['parameters'] or meta['propagation_seed']!=spec['propagation_seeds'][task['backend']]):
            raise CheckpointError('reference input/parameter/seed differs')
        for name in ('pcc.py','pcc_numpy.py','pcc_numba.py','pcc_step.pyx','pcc_metrics.py'):
            if meta['source_hashes'][name]!=execution['source_hashes'][name]:raise CheckpointError('PCC algorithm source differs from reference')
        if any(meta[k]!=execution[k] for k in ('python','dependencies','platform')) or state['report']['identity']['cython_extension_sha256']!=execution['cython_extension_sha256']:
            raise CheckpointError('reference environment/binary differs; exact prefix comparison cannot be assumed')
        refs[task['task_id']]=dict(path=str(path),files=file_inventory(path))
    return dict(dependencies=dependencies,bundles=refs,stage_b_sha256=bs['plan']['report_sha256'],c1_sha256=cs['plan']['report_sha256'])


def resource(case,backend,spec):
    count=len(b.schedule_for(spec['parameters']['max_iter'],spec['milestones'],False,spec['max_observations']))
    result=estimates(case.nodes,case.classes,case.particles,count,backend)
    check_budget(SnapshotConfig(**spec['snapshot_config']),result)
    return dict(nodes=case.nodes,classes=case.classes,particles=case.particles,observations=count,**result)


def preflight(output,*,synthetic=False,iterations=7,fixture_cache=None,timing_summary=None,
              historical16=None,historical17=None,stage_b=None,stage_c1=None,analysis=None):
    spec=specification(synthetic,iterations)
    if not synthetic and any(x is None for x in (fixture_cache,timing_summary,historical16,historical17,stage_b,stage_c1,analysis)):
        raise ValueError('real preflight requires cached fixtures, timings, historical16/17, Stage B/C1 and joint analysis')
    output=c.safe_destination(output,[fixture_cache,timing_summary,historical16,historical17,stage_b,stage_c1,analysis])
    execution=identity();output.mkdir(parents=True,exist_ok=False);(output/'fixtures').mkdir()
    fixtures={};rows=[];preparer=FixturePreparer();started=time.perf_counter()
    for d in spec['cases']:
        case=Case(**d)
        arrays,info=preparer.prepare(case) if synthetic else b.load_fixture(Path(fixture_cache)/case.case_id,case)
        b.validate_fixture(case,arrays,info);save_fixture(output/'fixtures'/case.case_id,arrays,info)
        fixtures[case.case_id]=dict(fixture_sha256=info['fixture_sha256'],files=file_inventory(output/'fixtures'/case.case_id),
                                   array_hashes=info['hashes'],class_mapping=sorted(set(arrays['truth'].tolist())))
        for task in spec['tasks']:
            if task['case_id']==case.case_id:
                rows.append(dict(task,**resource(case,task['backend'],spec),graph_shape=list(arrays['neighbors'].shape),
                                 degree_max=int(arrays['degrees'].max()),directed_edges=int(arrays['degrees'].sum())))
    refs=None if synthetic else references(stage_b,stage_c1,analysis,spec,fixtures,execution)
    if refs:
        for path in refs['dependencies']:c.safe_destination(output,[path])
    history=None if synthetic else c.historical_check(historical16,historical17,spec,fixtures)
    report=dict(schema=1,specification=spec,identity=execution,fixtures=fixtures,resources=rows,references=refs,historical=history,
        run_count=12,total_iteration_budget=12*spec['parameters']['max_iter'],
        total_particle_updates=sum(r['particles']*spec['parameters']['max_iter'] for r in rows),
        total_snapshot_array_bytes=sum(r['snapshot_array_bytes'] for r in rows),
        aggregate_export_limit_bytes=12*spec['snapshot_config']['max_export_bytes'],
        wall_clock=b.timing_projection(timing_summary,spec,fixtures),execution_started=False,preparation_s=time.perf_counter()-started,
        limitations='Exploratory seeds selected after B/C1. Timing projection is not measured full-snapshot overhead or a runtime bound. Memory estimate excludes total process RSS/libraries/JIT. No visit/acceptance/occupancy/flip counters; no intra-fit resumption.')
    durable_file(output/'report.json',canonical(dict(schema=1,payload=report,sha256=digest(report)))+b'\n')
    return report


def verify_report(directory):
    directory=Path(directory);r=read_record(directory/'report.json');spec=r['specification']
    if spec!=specification(spec['synthetic'],spec['parameters']['max_iter']):raise CheckpointError('mechanistic specification changed')
    if r['identity']!=identity():raise CheckpointError('source/environment/binary differs from reviewed mechanistic preflight')
    expected=[]
    for desc in spec['cases']:
        case=Case(**desc);entry=r['fixtures'][case.case_id]
        arrays,info=b.load_fixture(directory/'fixtures'/case.case_id,case,entry['files'])
        if entry['fixture_sha256']!=info['fixture_sha256'] or entry['array_hashes']!=info['hashes'] or entry['class_mapping']!=sorted(set(arrays['truth'].tolist())):
            raise CheckpointError('mechanistic fixture identity changed')
        for task in spec['tasks']:
            if task['case_id']==case.case_id:
                expected.append(dict(task,**resource(case,task['backend'],spec),graph_shape=list(arrays['neighbors'].shape),degree_max=int(arrays['degrees'].max()),directed_edges=int(arrays['degrees'].sum())))
    if (r['resources']!=expected or r['run_count']!=12 or r['total_iteration_budget']!=12*spec['parameters']['max_iter']
            or r['total_particle_updates']!=sum(x['particles']*spec['parameters']['max_iter'] for x in expected)
            or r['total_snapshot_array_bytes']!=sum(x['snapshot_array_bytes'] for x in expected)
            or r['aggregate_export_limit_bytes']!=12*spec['snapshot_config']['max_export_bytes'] or r['execution_started'] is not False):
        raise CheckpointError('mechanistic resource totals/gates changed')
    if not spec['synthetic']:
        if (not isinstance(r['references'],dict) or set(r['references']['bundles'])!={t['task_id'] for t in spec['tasks']}
                or r['historical']['compatible'] is not True):
            raise CheckpointError('missing/inconsistent mechanistic reference inventory')
        for name,value in r['references']['dependencies'].items():
            if file_inventory(Path(name))!=value:raise CheckpointError('sealed B/C1 or analysis dependency changed')
        for name,value in r['historical']['files'].items():
            if file_hash(name)!=value:raise CheckpointError('historical reference changed')
        if b.timing_projection(r['wall_clock']['reference'],spec,r['fixtures'])!=r['wall_clock']:
            raise CheckpointError('mechanistic timing projection changed')
    return r


class MechanisticPilot(b.Pilot):
    def __init__(self,root):
        self.store=Store(b.output_path(root))
        if self.store.plan().get('pilot_kind')!='mechanistic_snapshot_pilot':raise CheckpointError('not a mechanistic pilot store')

    @classmethod
    def create(cls,root,preflight_directory):
        r=verify_report(preflight_directory);sources=[preflight_directory]
        if r['references']:
            sources+=list(r['references']['dependencies'])+r['historical']['roots']
        c.safe_destination(root,sources)
        plan=dict(pilot_kind='mechanistic_snapshot_pilot',campaign_id=uuid.uuid4().hex,tasks=r['specification']['tasks'],
                  report=r,report_sha256=digest(r),preflight=str(Path(preflight_directory).resolve()))
        Store.create(b.output_path(root),plan);return cls(root)

    def context(self):
        plan=self.store.plan();r=verify_report(plan['preflight'])
        if plan['report']!=r or plan['report_sha256']!=digest(r) or plan['tasks']!=r['specification']['tasks']:
            raise CheckpointError('mechanistic plan/preflight binding changed')
        return plan,r

    def validate_bundle(self,path,task,report):
        with read_snapshots(path) as data:
            payload=data['summary'];m=payload['metadata'];rich=data['metadata'];spec=report['specification']
            case=next(Case(**d) for d in spec['cases'] if Case(**d).case_id==task['case_id'])
            fixture=report['fixtures'][case.case_id]
            if (m['dataset']!=case.name or m['label_seed']!=case.label_seed or m['backend']!=task['backend']
                    or m['fixture_hashes']!={k:fixture['array_hashes'][k] for k in ('neighbors','degrees','labels','truth')}
                    or m['parameters']!=spec['parameters'] or m['milestones']!=spec['milestones'] or m['include_checks']
                    or m['propagation_seed']!=spec['propagation_seeds'][task['backend']] or m['k_nn']!=case.k_nn
                    or m['class_index_to_label']!=fixture['class_mapping'] or m['budget']['max_observations']!=spec['max_observations']
                    or m['budget']['max_bytes']!=spec['max_bytes'] or rich['limits']!=spec['snapshot_config']
                    or m['source_commit']!=report['identity']['source_commit'] or m['source_hashes']!=report['identity']['source_hashes']
                    or rich['source_hashes']['experiments/mechanistic_snapshots.py']!=report['identity']['mechanistic_sources']['experiments/mechanistic_snapshots.py']
                    or any(m[k]!=report['identity'][k] for k in ('python','platform','dependencies','protocol_sha256'))):
                raise CheckpointError('mechanistic bundle differs from immutable task')
            if report['references']:
                ref=report['references']['bundles'][task['task_id']]
                if file_inventory(Path(ref['path']))!=ref['files']:raise CheckpointError('prefix reference changed')
                old=b.read_output(ref['path'])
                observed={x['iteration']:x for x in payload['observations']}
                for row in old['observations']:
                    now=observed[row['iteration']]
                    if row['state_hashes']!=now['state_hashes'] or row['assignments']!=now['assignments'] or row['accuracy_unlabeled_retrospective']!=now['accuracy_unlabeled_retrospective']:
                        raise CheckpointError('instrumentation differs from frozen B/C1 common milestone')
        return payload

    def execute(self,*,reviewed_preflight=None,max_runs=12,retry_task=None,reason=None,fault=lambda point:None):
        if type(max_runs) is not int or not 0<=max_runs<=12:raise ValueError('max_runs must be 0..12')
        with self.store.lock():
            state=self.audit()
            if not state['report']['specification']['synthetic'] and reviewed_preflight!=state['plan']['report_sha256']:
                raise CheckpointError('real fit requires exact approved mechanistic preflight SHA')
            for tid in state['done']+state['recoverable']:self.commit_bundle(state,tid)
            if retry_task is not None:
                if retry_task not in state['ambiguous'] or not reason or not reason.strip():raise CheckpointError('explicit incomplete retry/reason required')
                self.store.event('pilot_retry',task_id=retry_task,attempt_id=state['last'][retry_task],reason=reason,
                    preserved=file_inventory(self.store.root/'sessions'/state['last'][retry_task]))
            state=self.audit()
            if state['ambiguous']:raise CheckpointError('partial fit requires explicit retry: '+', '.join(state['ambiguous']))
            spec=state['report']['specification'];launched=0
            for task in spec['tasks']:
                tid=task['task_id']
                if tid in state['done'] or launched>=max_runs:continue
                case=next(Case(**d) for d in spec['cases'] if Case(**d).case_id==task['case_id'])
                arrays,_=b.load_fixture(Path(state['plan']['preflight'])/'fixtures'/case.case_id,case,state['report']['fixtures'][case.case_id]['files'])
                aid=uuid.uuid4().hex;self.store.event('pilot_started',task_id=tid,attempt_id=aid);fault('started')
                directory=self.store.root/'sessions'/aid;directory.mkdir()
                model=PCC(impl=task['backend']);model.set_graph(arrays['neighbors'],arrays['degrees'],k_nn=case.k_nn)
                pred,trace,_=observe_fit(model,arrays['labels'],truth=arrays['truth'],milestones=spec['milestones'],
                    propagation_seed=spec['propagation_seeds'][task['backend']],max_observations=spec['max_observations'],max_bytes=spec['max_bytes'],
                    snapshot_config=SnapshotConfig(**spec['snapshot_config']),**spec['parameters'])
                b.validate_final_state(model,pred,arrays,trace);fault('fit_finished')
                write_snapshots(directory/'trajectory',trace,metadata(dataset=case.name,label_seed=case.label_seed,collector=trace))
                self.validate_bundle(directory/'trajectory',task,state['report']);fault('bundle_finished')
                self.commit_bundle(self.audit(),tid,fault);fault('committed');launched+=1;state=self.audit()
            return self.status()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    dry=commands.add_parser('preflight');dry.add_argument('--output',type=Path,required=True)
    dry.add_argument('--synthetic',action='store_true');dry.add_argument('--iterations',type=int)
    for name in ('fixture-cache','timing-summary','historical16','historical17','stage-b','stage-c1','analysis'):dry.add_argument('--'+name,type=Path)
    init=commands.add_parser('init');init.add_argument('--preflight',type=Path,required=True);init.add_argument('--root',type=Path,required=True)
    for name in ('run','status','recover-lock'):
        sub=commands.add_parser(name);sub.add_argument('--root',type=Path,required=True)
        if name=='run':
            sub.add_argument('--reviewed-preflight');sub.add_argument('--max-runs',type=int,default=12);sub.add_argument('--retry-task');sub.add_argument('--reason')
        elif name=='recover-lock':sub.add_argument('--token',required=True);sub.add_argument('--reason',required=True)
    args=vars(parser.parse_args(argv));command=args.pop('command')
    if command=='preflight':
        if args['iterations'] is not None and not args['synthetic']:parser.error('--iterations is synthetic only')
        args['iterations']=7 if args['iterations'] is None else args['iterations']
        report=preflight(**args);result=dict(report=report,report_sha256=digest(report))
    elif command=='init':result=MechanisticPilot.create(args['root'],args['preflight']).status()
    else:
        pilot=MechanisticPilot(args.pop('root'))
        if command=='run':result=pilot.execute(**args)
        elif command=='recover-lock':pilot.store.recover_lock(args['token'],args['reason']);result=pilot.status()
        else:result=pilot.status()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
