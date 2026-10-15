"""Read-only audit/report for the approved seed-10 stage; never runs PCC."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from itertools import permutations
import json
from pathlib import Path
import time

import numpy as np
from benchmarks.checkpoints import Store, read_record, digest, file_inventory
from benchmarks.resumable import Controller, full_specification

if not __debug__:
    raise RuntimeError('Strict audit requires Python assertion checks enabled')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seconds(start,end):
    return (datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds()


def stats(values):
    a=np.asarray(values,dtype=float)
    q=np.percentile(a,[25,50,75])
    return dict(median=float(q[1]),q1=float(q[0]),q3=float(q[2]),iqr=float(q[2]-q[0]),
                minimum=float(a.min()),maximum=float(a.max()),total=float(a.sum()))


def analyze(root,baseline,provenance):
    started=time.perf_counter()
    controller=Controller(Store(root))
    state=controller.audit(check_environment=False)
    plan=state['plan']
    assert plan['protocol']=='historical_early_stopping_full_v1'
    assert not plan['manifest']['repository']['dirty']
    assert digest(full_specification())==read('docs/benchmarks/early_stopping_full_plan/matrix.json')['specification_sha256']
    for relative,h in plan['identity']['source_hashes'].items():
        assert sha(provenance/'source_files'/relative)==h,relative
    extension=plan['identity']['cython_extension']
    assert sha(extension['path'])==extension['sha256']
    pilot=read('docs/benchmarks/early_stopping_pilot/results/raw/manifest.json')
    for relative,h in pilot['repository']['source_hashes'].items():
        if relative.startswith('pcc') or relative in ('benchmarks/run.py','benchmarks/fixtures.py','benchmarks/runner.py','benchmarks/historical.py'):
            assert plan['identity']['source_hashes'][relative]==h,relative
    cases={c['name']+'-labels'+str(c['label_seed']):c for c in plan['cases']}
    selected=[t for t in plan['tasks'] if cases[t['case_id']]['label_seed']==10]
    assert len(plan['tasks'])==3600 and len(plan['cases'])==200 and len(selected)==360
    assert [t['task_id'] for t in selected]==[f't{i:04d}' for i in range(360)]
    assert set(state['results'])=={t['task_id'] for t in selected}
    assert len(state['pending'])==3240 and not state['ambiguous'] and not state['reconcile']
    rows=[state['results'][t['task_id']]['row'] for t in selected]
    events=controller.store.events()
    starts=[e for e in events if e['kind']=='started']
    assert len(starts)==360 and {e['task_id'] for e in starts}==set(state['results'])
    assert all(cases[t['case_id']]['label_seed']==10 for t in selected)
    anomalies=[e for e in events if e['kind'] in ('failed','retry_authorized','fixture_retry','lock_recovered','reconciled','summary_retired')]
    assert not anomalies,anomalies
    assert not (root/'writer.lock').exists()
    groups=[]
    fixtures=[]
    for cid,case in cases.items():
        if case['label_seed']!=10:
            assert not (root/'fixtures'/cid).exists()
            continue
        actual=controller.fixture_cache[cid][1]
        previous=read(baseline/'fixtures'/cid/'fixture.json')
        assert actual['hashes']==previous['hashes'] and actual['fixture_sha256']==previous['fixture_sha256'],cid
        fixtures.append(dict(case_id=cid,fixture_sha256=actual['fixture_sha256'],hashes=actual['hashes'],
            timings=actual['timings'],matches_frozen_and_pr16=True))
        selected_rows=[r for r in rows if r['case_id']==cid]
        assert len(selected_rows)==18
        assert {tuple(r['execution_order']) for r in selected_rows}==set(permutations(('numpy','numba','cython')))
        for backend in ('numpy','numba','cython'):
            group=[r for r in selected_rows if r['backend']==backend]
            assert len(group)==6 and {r['repeat'] for r in group}==set(range(6))
            assert Counter(r['order_index'] for r in group)=={0:2,1:2,2:2}
            reference=(group[0]['state_hashes'],group[0]['effective_iterations'],group[0]['stop_reason'],group[0]['accuracy_unlabeled'])
            assert all((r['state_hashes'],r['effective_iterations'],r['stop_reason'],r['accuracy_unlabeled'])==reference for r in group)
            groups.append(dict(dataset=case['name'],backend=backend,repetitions=6,
                iterations=group[0]['effective_iterations'],stop_reason=group[0]['stop_reason'],
                cap_hits=sum(r['reached_max_iter'] for r in group),censored=sum(r['stop_reason']=='max_iter' for r in group),
                accuracy_percent=100*group[0]['accuracy_unlabeled'],state_hashes=group[0]['state_hashes'],
                propagation_s=stats([r['propagation_s'] for r in group]),fit_predict_s=stats([r['fit_predict_s'] for r in group]),
                task_ids=[r['task_id'] for r in group]))
    sessions=[]; warmups=[]
    for directory in sorted((root/'sessions').iterdir()):
        session=read_record(directory/'session.json')
        end=read_record(directory/'end.json')
        assert session['stage_seed']==10 and end['status']=='partial' and end['measured_tasks_completed']==360
        warmups.extend(read_record(p) for p in sorted((directory/'warmups').iterdir()))
        sessions.append(dict(start_utc=session['started_utc'],**end,
            seed_helper=read_record(directory/'seed_helper.json')))
    assert len(sessions)==1 and len(warmups)==120
    summary_events=[e for e in events if e['kind']=='summary']
    latest=read_record(root/'summaries'/summary_events[-1]['generation']/'summary.json')
    assert latest['status']=='partial' and latest['measurements']==360
    assert latest['stages']['10']==dict(expected=360,completed=360,status='complete')
    assert all(latest['stages'][str(seed)]['status']=='not_started' for seed in range(11,20))
    preserved=read(provenance/'prior-artifact-preservation.json')
    assert all(sha(p)==h for p,h in preserved.items())
    execution=read(provenance/'execution.json')
    assert execution['exit_code']==0 and execution['stop_reason'] is None and execution['runtime_limit'] is None
    completion={e['task_id']:e for e in events if e['kind']=='completed'}
    helper_wall=sum(seconds(r['call_started_utc'],r['call_completed_utc']) for r in rows)
    publication=[seconds(r['call_completed_utc'],completion[r['task_id']]['timestamp_utc']) for r in rows]
    assert all(t>=0 for t in publication)
    fit=sum(r['fit_predict_s'] for r in rows)
    warmfit=sum(w['row']['fit_predict_s'] for w in warmups)
    prep=sum(t for f in fixtures for t in f['timings'].values())
    seed_helper=sum(s['seed_helper']['first_call_s'] for s in sessions)
    session_elapsed=sum(s['elapsed_s'] for s in sessions)
    overhead=dict(measured_helper_wall_s=helper_wall,measured_helper_untimed_checks_s=helper_wall-fit,
        validation_and_result_publication_window_s=stats(publication),
        warmup_fit_s=warmfit,fixture_preparation_s=prep,seed_helper_first_call_s=seed_helper,
        controller_session_elapsed_s=session_elapsed,
        session_other_s=session_elapsed-fit-warmfit-prep-seed_helper,
        native_wall_outside_session_s=execution['wall_clock_s']-session_elapsed,
        native_wall_less_fit_warmup_preparation_seed_s=execution['wall_clock_s']-fit-warmfit-prep-seed_helper,
        residual_note='Residual includes untimed fit/fixture/row verification, graph copies/model construction/RNG seeding, checkpoint publication, ledger/session I/O, startup and final audit/summary. Publication windows include validation and result writes, but exclude subsequent completion-event fsync; pure filesystem cost is not isolated. Wall UTC spans and monotonic fit timers can differ slightly.')
    estimate=read('docs/benchmarks/early_stopping_full_plan/seed10_estimate.json')
    calibration_rows=[]
    for path in ('docs/benchmarks/early_stopping_pilot/results/raw/measurements.jsonl',
                 'docs/benchmarks/early_stopping_second_calibration/results/measurements.jsonl'):
        calibration_rows.extend(json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines())
    calibration_differences=[]
    for g in groups:
        earlier=next(r for r in calibration_rows if r['case_id']==g['dataset']+'-labels10' and r['backend']==g['backend'])
        changed=[field for field in ('state_hashes','effective_iterations','stop_reason','accuracy_unlabeled')
                 if earlier[field]!=next(r for r in rows if r['case_id']==earlier['case_id'] and r['backend']==earlier['backend'])[field]]
        if changed:
            calibration_differences.append(dict(dataset=g['dataset'],backend=g['backend'],fields=changed))
    scenarios=dict(observed_stopping=estimate['observed_stopping_accounted_s'],observed_rates_cap=estimate['observed_rate_cap_s'],
        historical_median_cap=estimate['historical_cap_s']['median'],historical_maximum_cap=estimate['historical_cap_s']['maximum'],conservative_cap=estimate['conservative_cap_s'])
    comparisons={name:dict(planned_s=v,actual_wall_s=execution['wall_clock_s'],actual_over_planned=execution['wall_clock_s']/v,
        difference_s=execution['wall_clock_s']-v) for name,v in scenarios.items()}
    timing_forecast_differences=[]
    for g in groups:
        forecast=next(v for v in estimate['groups'] if v['dataset']==g['dataset'] and v['backend']==g['backend'])
        timing_forecast_differences.append(dict(dataset=g['dataset'],backend=g['backend'],
            forecast_fit_s=forecast['stage_fit_s'],actual_fit_s=g['fit_predict_s']['total'],
            difference_s=g['fit_predict_s']['total']-forecast['stage_fit_s']))
    backend_summary=[]
    dataset_summary=[]
    for field,keys,output in [('backend',('numpy','numba','cython'),backend_summary),
                            ('dataset',[c['name'] for c in plan['cases'][:20]],dataset_summary)]:
        for key in keys:
            rr=[r for r in rows if (r['backend'] if field=='backend' else cases[r['case_id']]['name'])==key]
            gg=[g for g in groups if g[field]==key]
            output.append(dict(**{field:key},measurements=len(rr),iterations=sum(r['effective_iterations'] for r in rr),
                early_stops=sum(r['stop_reason']=='early_stop' for r in rr),censored=sum(r['stop_reason']=='max_iter' for r in rr),
                censor_fraction=sum(r['stop_reason']=='max_iter' for r in rr)/len(rr),
                propagation_total_s=sum(r['propagation_s'] for r in rr),fit_total_s=sum(r['fit_predict_s'] for r in rr),
                accuracy_macro_percent=float(np.mean([g['accuracy_percent'] for g in gg])),
                accuracy_spread_percentage_points=max(g['accuracy_percent'] for g in gg)-min(g['accuracy_percent'] for g in gg)))
    monitor=[json.loads(s) for s in (provenance/'monitor.jsonl').read_text().splitlines()]
    flat=[dict(dataset=cases[r['case_id']]['name'],seed=10,backend=r['backend'],repeat=r['repeat'],task_id=r['task_id'],
        iterations=r['effective_iterations'],reason=r['stop_reason'],cap_hit=r['reached_max_iter'],censored=r['stop_reason']=='max_iter',
        accuracy_percent=100*r['accuracy_unlabeled'],propagation_s=r['propagation_s'],fit_predict_s=r['fit_predict_s'],
        session_id=r['session_id'],fixture_sha256=r['fixture_sha256'],state_hashes=json.dumps(r['state_hashes'],sort_keys=True),
        execution_order=','.join(r['execution_order']),order_index=r['order_index']) for r in rows]
    return dict(audit='passed',stage_seed=10,source_commit=plan['manifest']['repository']['commit'],
        specification_sha256=digest(full_specification()),plan_sha256=digest(plan),measurements=360,warmups=120,fixtures=20,
        pending_unauthorized_tasks=3240,sessions=sessions,attempts=360,anomalies=anomalies,
        early_stops=sum(r['stop_reason']=='early_stop' for r in rows),censored=sum(r['stop_reason']=='max_iter' for r in rows),
        cap_hits=sum(r['reached_max_iter'] for r in rows),iterations=sum(r['effective_iterations'] for r in rows),
        propagation_s=sum(r['propagation_s'] for r in rows),fit_predict_s=fit,
        groups=groups,backend_summary=backend_summary,dataset_summary=dataset_summary,rows=flat,
        timing_variability_ranked=sorted([dict(dataset=g['dataset'],backend=g['backend'],
            fit_maximum_over_minimum=g['fit_predict_s']['maximum']/g['fit_predict_s']['minimum'],
            fit_range_s=g['fit_predict_s']['maximum']-g['fit_predict_s']['minimum'],
            fit_iqr_over_median=g['fit_predict_s']['iqr']/g['fit_predict_s']['median']) for g in groups],
            key=lambda v:v['fit_range_s'],reverse=True),
        earlier_calibration_differences=calibration_differences,
        fixture_audit=fixtures,overhead=overhead,execution=execution,planning_comparisons=comparisons,
        timing_forecast_differences=sorted(timing_forecast_differences,key=lambda v:abs(v['difference_s']),reverse=True),
        resources=dict(minimum_available_ram_bytes=min(m['available_ram_bytes'] for m in monitor),
            minimum_free_disk_bytes=min(m['free_disk_bytes'] for m in monitor),monitor_samples=len(monitor)),
        prior_files_verified=len(preserved),outliers_excluded=0,audit_duration_s=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root,args.baseline,args.provenance)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (args.output/'results.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['rows'][0])); writer.writeheader(); writer.writerows(result['rows'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','groups','fixture_audit')},indent=2))


if __name__=='__main__':
    main()
