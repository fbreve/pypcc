"""Stage C preparation and bounded whole-fit resumption; real runs require review.

Stage B is an immutable dependency, never a writable destination. No downloads.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from benchmarks.checkpoints import CheckpointError, Store, canonical, digest, durable_file, file_inventory, read_record
from benchmarks.fixtures import Case, FixturePreparer, file_hash, save_fixture
from benchmarks.historical import FROZEN
from . import convergence_pilot as b


def specification(part, synthetic=False, iterations=7):
    if part not in ('c1', 'c2'):
        raise ValueError('part must be c1 or c2')
    spec = b.specification(synthetic, iterations)
    seeds = range(12, 20) if part == 'c1' else (10, 11)
    names = ('pageblocks', 'spambase') if part == 'c1' else ('banknote', 'sonar')
    if synthetic:
        cases = [Case(name, n, f, c, synthetic=True, k_nn=3, label_seed=seed)
                 for name, n, f, c in (('stage_c_a', 24, 2, 2), ('stage_c_b', 30, 3, 3)) for seed in seeds]
    else:
        cases = [Case(name, FROZEN[name]['nodes'], FROZEN[name]['processed_features'],
                      FROZEN[name]['classes'], label_seed=seed) for name in names for seed in seeds]
    spec.update(protocol=f"{'synthetic' if synthetic else 'historical'}-stage-{part}-v1", part=part,
                cases=[asdict(c) for c in cases], tasks=[dict(task_id=f't{3*i+j:04d}', case_id=c.case_id, backend=backend)
                for i,c in enumerate(cases) for j,backend in enumerate(b.BACKENDS)])
    return spec


def validate_spec(spec):
    if spec != specification(spec['part'], spec['synthetic'], spec['parameters']['max_iter']):
        raise CheckpointError('Stage C specification changed')


def identity():
    result = b.execution_identity()
    result['stage_c_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return result


def safe_destination(destination, sources):
    destination=b.output_path(destination)
    for source in sources:
        if source is not None:
            source=Path(source).resolve()
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                raise CheckpointError('Stage C destination overlaps a protected source')
    return destination


def check_fixtures(directory, report):
    rows=[]
    for desc in report['specification']['cases']:
        case=Case(**desc); entry=report['fixtures'][case.case_id]
        arrays,info=b.load_fixture(Path(directory)/'fixtures'/case.case_id, case, entry['files'])
        if (entry['fixture_sha256']!=info['fixture_sha256'] or entry['array_hashes']!=info['hashes']
                or entry['class_mapping']!=np.unique(arrays['truth']).tolist()):
            raise CheckpointError('fixture provenance mismatch')
        for task in report['specification']['tasks']:
            if task['case_id']==case.case_id:
                rows.append(dict(task, **b.resources(case,task['backend'],report['specification'],arrays)))
    if rows!=report['resources'] or not all(r['fits'] for r in rows):
        raise CheckpointError('resource estimates or observation budget changed')
    return rows


class FrozenStageB(b.Pilot):
    """Offline artifact audit: old environment is provenance, not today's runtime.

    This read-only adapter MUST NOT be used to execute an old campaign.
    """
    def context(self):
        plan=self.store.plan(); report=read_record(Path(plan['preflight'])/'report.json')
        b.validate_spec(report['specification'])
        if (digest(report)!=plan['report_sha256'] or report!=plan['report']
                or report['specification']['synthetic'] or plan['tasks']!=report['specification']['tasks']):
            raise CheckpointError('invalid frozen Stage B binding')
        check_fixtures(plan['preflight'],report)
        return plan,report

    def execute(self, **kwargs):
        raise CheckpointError('frozen Stage B audit is read-only')


def audit_stage_b(root, analysis):
    root,analysis=Path(root).resolve(),Path(analysis).resolve()
    state=FrozenStageB(root).audit()
    if len(state['done'])!=12 or state['recoverable'] or state['ambiguous']:
        raise CheckpointError('Stage B is incomplete')
    manifest=json.loads((analysis/'analysis_manifest.json').read_text(encoding='utf-8'))
    inventory=file_inventory(analysis)
    if manifest['outputs']!={k:v for k,v in inventory.items() if k!='analysis_manifest.json'}:
        raise CheckpointError('Stage B analysis inventory/checksum mismatch')
    integrity=json.loads((analysis/'integrity.json').read_text(encoding='utf-8'))
    if (integrity['status']!='passed' or integrity['complete']!=12 or integrity['observations']!=120
            or integrity['preflight_sha256']!=state['plan']['report_sha256']):
        raise CheckpointError('analysis does not bind complete Stage B')
    source_tasks={r['task']['task_id']:r for r in integrity['tasks']}
    if set(source_tasks)!={t['task_id'] for t in state['plan']['tasks']}:
        raise CheckpointError('analysis source task inventory mismatch')
    for name,value in integrity['historical_inputs'].items():
        if file_hash(name)!=value: raise CheckpointError('Stage B analysis historical input changed')
    for task in state['plan']['tasks']:
        tid=task['task_id']; path=root/'sessions'/state['last'][tid]/'trajectory'
        data=FrozenStageB(root).validate_bundle(path,task,state['report'])
        if source_tasks[tid]['files']!=file_inventory(path):
            raise CheckpointError('analysis source bundle differs from Stage B')
        case=Case(**next(d for d in state['report']['specification']['cases'] if Case(**d).case_id==task['case_id']))
        arrays,_=b.load_fixture(Path(state['plan']['preflight'])/'fixtures'/case.case_id,case)
        mask=arrays['labels']==-1; previous=None
        rows=data['observations']
        if [r['iteration'] for r in rows]!=list(b.MILESTONES) or sum(r['final'] for r in rows)!=1:
            raise CheckpointError('Stage B schedule differs')
        for row in rows:
            pred=np.array(data['metadata']['class_index_to_label'])[row['assignments']]
            changes=None if previous is None else int(np.count_nonzero(pred!=previous))
            if (float(np.mean(pred[mask]==arrays['truth'][mask]))!=row['accuracy_unlabeled_retrospective']
                    or changes!=row['class_changes'] or not np.array_equal(pred[~mask],arrays['labels'][~mask])):
                raise CheckpointError('Stage B retrospective diagnostics inconsistent')
            previous=pred
    # Seal immutable source artifacts; no assumption of current execution identity.
    return dict(root=str(root),analysis=str(analysis),preflight=str(Path(state['plan']['preflight'])),
                report_sha256=state['plan']['report_sha256'],complete=12,observations=120,
                files=file_inventory(root),analysis_files=inventory,
                preflight_files=file_inventory(Path(state['plan']['preflight'])))


def verify_dependency(dep):
    for path,key in [('root','files'),('analysis','analysis_files'),('preflight','preflight_files')]:
        if file_inventory(Path(dep[path]))!=dep[key]:
            raise CheckpointError('sealed Stage B dependency changed')


def historical_check(root16, root17, spec, fixtures):
    """Match all six technical repetitions; never run historical controllers."""
    root16,root17=Path(root16).resolve(),Path(root17).resolve()
    paths=[root16/'manifest.json',root16/'measurements.jsonl',root17/'plan.json',root17/'plan_identity.json']
    manifest=json.loads(paths[0].read_text(encoding='utf-8')); store=Store(root17); plan=store.plan()
    params={k:spec['parameters'][k] for k in ('p_grd','delta_v','deltap','dexp','es_chk')}
    if manifest['configuration']['parameters']!=params or plan['parameters']!=params or plan['max_iter']!=500000 or plan['propagation_seed']!=0:
        raise CheckpointError('historical parameters or stopping protocol mismatch')
    fixed=[json.loads(line) for line in paths[1].read_text(encoding='utf-8').splitlines() if line.strip()]
    groups=[]; plan_sha=digest(plan)
    for task in spec['tasks']:
        old=[r for r in fixed if (r['case_id'],r['backend'])==(task['case_id'],task['backend'])]
        tasks=[t for t in plan['tasks'] if (t['case_id'],t['backend'])==(task['case_id'],task['backend'])]
        selected=[]
        for t in tasks:
            path=root17/'results'/(t['task_id']+'.json'); rec=read_record(path); paths.append(path)
            if (rec['plan_sha256']!=plan_sha or rec['task_id']!=t['task_id']
                    or rec['row']['repeat']!=t['measurement_index']):
                raise CheckpointError('historical endpoint not bound to plan')
            selected.append(rec['row'])
        if len(old)!=6 or len(selected)!=6:
            raise CheckpointError('historical repetitions missing/duplicate')
        for rows in (old,selected):
            if sorted(r['repeat'] for r in rows)!=list(range(6)):
                raise CheckpointError('historical repetition inventory mismatch')
            for r in rows:
                if (r['fixture_sha256']!=fixtures[task['case_id']]['fixture_sha256']
                        or r['propagation_seed_effective']!=spec['propagation_seeds'][task['backend']]
                        or r['case_id']!=task['case_id'] or r['backend']!=task['backend']):
                    raise CheckpointError('historical fixture/backend/seed mismatch')
        if any(r['mode']!='fixed' or r['effective_iterations']!=10000 or r['max_iter']!=10000 for r in old):
            raise CheckpointError('PR16 is not fixed 10000 work')
        if any(r['mode']!='early_stopping' or r['max_iter']!=500000 for r in selected):
            raise CheckpointError('PR17 stopping policy differs')
        groups.append(dict(task,pr16_repetitions=6,pr17_repetitions=6,
                           pr17_iterations=sorted(set(r['effective_iterations'] for r in selected))))
    return dict(compatible=True,groups=groups,roots=[str(root16),str(root17)],files={str(p):file_hash(p) for p in paths},
                limitation='Matched frozen inputs/parameters/effective seeds, not new trajectory equality or RNG audit; PR17 uses early stopping and 500000 cap; environments/sources differ.')


def preflight(output, *, part, synthetic=False, iterations=7, fixture_cache=None,
              timing_summary=None, historical16=None, historical17=None, stage_b=None, analysis=None):
    spec=specification(part,synthetic,iterations)
    output=safe_destination(output,[fixture_cache,timing_summary,historical16,historical17,stage_b,analysis])
    dependency=None if synthetic else audit_stage_b(stage_b,analysis)
    if dependency is not None: safe_destination(output,[dependency['preflight']])
    if not synthetic and any(v is None for v in (fixture_cache,timing_summary,historical16,historical17)):
        raise ValueError('real preflight requires verified fixture cache, timings and both historical sources')
    execution=identity(); output.mkdir(parents=True,exist_ok=False); (output/'fixtures').mkdir()
    fixtures={}; rows=[]; preparer=FixturePreparer(); started=time.perf_counter()
    for desc in spec['cases']:
        case=Case(**desc)
        arrays,info=(preparer.prepare(case) if synthetic else b.load_fixture(Path(fixture_cache)/case.case_id,case))
        b.validate_fixture(case,arrays,info); save_fixture(output/'fixtures'/case.case_id,arrays,info)
        fixtures[case.case_id]=dict(fixture_sha256=info['fixture_sha256'],files=file_inventory(output/'fixtures'/case.case_id),
                                   array_hashes=info['hashes'],class_mapping=np.unique(arrays['truth']).tolist())
        for task in spec['tasks']:
            if task['case_id']==case.case_id: rows.append(dict(task,**b.resources(case,task['backend'],spec,arrays)))
    history=None if synthetic else historical_check(historical16,historical17,spec,fixtures)
    report=dict(schema=1,specification=spec,identity=execution,fixtures=fixtures,resources=rows,
                stage_b=dependency,historical=history,execution_started=False,run_count=len(rows),
                total_iteration_budget=len(rows)*spec['parameters']['max_iter'],
                total_particle_updates=sum(r['particles']*spec['parameters']['max_iter'] for r in rows),
                aggregate_export_limit_bytes=len(rows)*b.MAX_BYTES,
                observation_budget_fits=all(r['fits'] for r in rows),
                wall_clock=b.timing_projection(timing_summary,spec,fixtures),preparation_s=time.perf_counter()-started,
                storage_note='128 observations/64 MiB estimated diagnostic storage per run; excludes process RSS, libraries/JIT/fixtures and interrupted attempts. C1 adds 48 runs, never repeats Stage B seeds 10/11; C2 adds 12. Resume is whole-fit only.')
    durable_file(output/'report.json',canonical(dict(schema=1,payload=report,sha256=digest(report)))+b'\n')
    return report


def verify_report(directory):
    report=read_record(Path(directory)/'report.json'); spec=report['specification']; validate_spec(spec)
    if report['identity']!=identity():
        raise CheckpointError('source/environment/binary changed since Stage C preflight')
    rows=check_fixtures(directory,report); count=len(spec['tasks'])
    if (report['run_count']!=count or report['total_iteration_budget']!=count*spec['parameters']['max_iter']
            or report['total_particle_updates']!=sum(r['particles']*spec['parameters']['max_iter'] for r in rows)
            or report['aggregate_export_limit_bytes']!=count*b.MAX_BYTES
            or report['execution_started'] is not False or not report['observation_budget_fits']):
        raise CheckpointError('Stage C totals/gates changed')
    if not spec['synthetic']:
        verify_dependency(report['stage_b'])
        if not report['historical']['compatible']:
            raise CheckpointError('historical compatibility missing')
        for name,value in report['historical']['files'].items():
            if file_hash(name)!=value: raise CheckpointError('historical reference changed')
        timing=report['wall_clock']
        if b.timing_projection(timing['reference'],spec,report['fixtures'])!=timing:
            raise CheckpointError('timing projection changed')
    return report


class StageC(b.Pilot):
    def __init__(self, root):
        self.store=Store(b.output_path(root))
        if self.store.plan().get('pilot_kind')!='convergence_stage_c':
            raise CheckpointError('not a Stage C store; Stage B cannot be modified')

    @classmethod
    def create(cls,root,preflight_directory):
        report=verify_report(preflight_directory)
        sources=[preflight_directory]
        if report['stage_b'] is not None:
            sources += [report['stage_b'][k] for k in ('root','preflight','analysis')]
            sources += report['historical']['roots']
        safe_destination(root,sources)
        plan=dict(pilot_kind='convergence_stage_c',campaign_id=uuid.uuid4().hex,tasks=report['specification']['tasks'],
                  report=report,report_sha256=digest(report),preflight=str(Path(preflight_directory).resolve()))
        Store.create(b.output_path(root),plan); return cls(root)

    def context(self):
        plan=self.store.plan(); report=verify_report(plan['preflight'])
        if report!=plan['report'] or digest(report)!=plan['report_sha256'] or plan['tasks']!=report['specification']['tasks']:
            raise CheckpointError('Stage C plan/preflight binding changed')
        return plan,report

    def execute(self, *, max_runs=None, **kwargs):
        count=len(self.store.plan()['tasks'])
        if max_runs is None: max_runs=count
        if type(max_runs) is not int or not 0<=max_runs<=count:
            raise ValueError('max_runs outside Stage C task budget')
        # Reuse the tested Stage B transaction path, whose per-call cap is 12.
        # Boundaries are BETWEEN independent fits, never inside a trajectory.
        left=max_runs
        while True:
            batch=min(left,12)
            result=super().execute(max_runs=batch,**kwargs)
            left-=batch
            if left<=0 or result['complete']==count: return result
            kwargs.pop('retry_task',None); kwargs.pop('reason',None)

    def status(self):
        state=self.audit()
        return dict(report_sha256=state['plan']['report_sha256'],complete=len(state['done']),
                    total=len(state['plan']['tasks']),recoverable=state['recoverable'],ambiguous=state['ambiguous'],
                    pending=[t['task_id'] for t in state['plan']['tasks'] if t['task_id'] not in state['done']])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    dry=sub.add_parser('preflight'); dry.add_argument('--part',choices=('c1','c2'),required=True)
    dry.add_argument('--output',type=Path,required=True); dry.add_argument('--synthetic',action='store_true')
    dry.add_argument('--iterations',type=int,default=None)
    for arg in ('fixture-cache','timing-summary','historical16','historical17','stage-b','analysis'):
        dry.add_argument('--'+arg,type=Path)
    init=sub.add_parser('init'); init.add_argument('--root',type=Path,required=True); init.add_argument('--preflight',type=Path,required=True)
    for name in ('run','status','recover-lock'):
        cmd=sub.add_parser(name); cmd.add_argument('--root',type=Path,required=True)
        if name=='run':
            cmd.add_argument('--reviewed-preflight'); cmd.add_argument('--max-runs',type=int)
            cmd.add_argument('--retry-task'); cmd.add_argument('--reason')
        elif name=='recover-lock':
            cmd.add_argument('--token',required=True);cmd.add_argument('--reason',required=True)
    args=vars(parser.parse_args(argv)); command=args.pop('command')
    if command=='preflight':
        if args['iterations'] is not None and not args['synthetic']: parser.error('--iterations is synthetic only')
        args['iterations']=7 if args['iterations'] is None else args['iterations']
        report=preflight(**args); result=dict(report=report,report_sha256=digest(report))
    elif command=='init': result=StageC.create(args['root'],args['preflight']).status()
    else:
        pilot=StageC(args.pop('root'))
        if command=='run': result=pilot.execute(**args)
        elif command=='recover-lock':
            pilot.store.recover_lock(args['token'],args['reason']);result=pilot.status()
        else: result=pilot.status()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
