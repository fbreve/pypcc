"""Read-only complete historical early-stopping audit and analysis; no PCC fits."""
import argparse
from collections import Counter
import csv
from itertools import permutations
import json
from pathlib import Path
import time
import numpy as np
from benchmarks.checkpoints import Store, read_record, digest, file_inventory
from benchmarks.resumable import Controller, full_specification
from tools.analyze_full_seed10 import read, sha, seconds, stats

if not __debug__: raise RuntimeError('Strict audit requires assertions enabled')
BACKENDS=('numpy','numba','cython')

def analyze(root,baseline,provenance):
    begin=time.perf_counter(); c=Controller(Store(root)); state=c.audit()
    plan=state['plan']; old=read('docs/benchmarks/early_stopping_full_seed11/results/analysis.json')
    assert digest(plan)==old['plan_sha256'] and digest(full_specification())==old['specification_sha256']
    assert set(state['results'])=={f't{i:04d}' for i in range(3600)}
    assert not state['pending'] and not state['ambiguous'] and not state['reconcile'] and not state['authorized_retries']
    assert not (root/'writer.lock').exists()
    assert all(sha(root/p)==h for p,h in read(provenance/'prior-checkpoints.json').items())
    assert all(sha(p)==h for p,h in read(provenance/'protected-artifacts.json').items())
    assert all(sha(provenance/'source_files'/p)==h for p,h in plan['identity']['source_hashes'].items())
    extension=plan['identity']['cython_extension']; assert sha(extension['path'])==extension['sha256']
    rows=[state['results'][f't{i:04d}']['row'] for i in range(3600)]
    cases={v['name']+'-labels'+str(v['label_seed']):v for v in plan['cases']}
    events=c.store.events(); starts=[e for e in events if e['kind']=='started']
    assert len(starts)==3600 and Counter(e['task_id'] for e in starts)=={f't{i:04d}':1 for i in range(3600)}
    assert not [e for e in events if e['kind'] in ('failed','retry_authorized','fixture_retry','lock_recovered','reconciled','summary_retired')]
    groups=[]; fixtures=[]
    for cid,case in cases.items():
        meta=c.fixture_cache[cid][1]; fixed=read(baseline/'fixtures'/cid/'fixture.json')
        assert meta['hashes']==fixed['hashes'] and meta['fixture_sha256']==fixed['fixture_sha256'],cid
        if case['label_seed']!=10:
            anchor=c.fixture_cache[case['name']+'-labels10'][1]
            assert meta['preparation']==dict(reused=True,source_case_id=case['name']+'-labels10')
            assert meta['timings']['dataset_load_s']==meta['timings']['graph_build_s']==0
            assert all(h==anchor['hashes'][k] for k,h in meta['hashes'].items() if k!='labels')
        fixtures.append(dict(case_id=cid,fixture_sha256=meta['fixture_sha256'],hashes=meta['hashes']))
        rr=[r for r in rows if r['case_id']==cid]
        assert len(rr)==18 and {tuple(r['execution_order']) for r in rr}==set(permutations(BACKENDS))
        for b in BACKENDS:
            gg=[r for r in rr if r['backend']==b]
            assert len(gg)==6 and {r['repeat'] for r in gg}==set(range(6)) and Counter(r['order_index'] for r in gg)=={0:2,1:2,2:2}
            ref=tuple(gg[0][k] for k in ('state_hashes','effective_iterations','stop_reason','accuracy_unlabeled'))
            assert all(tuple(r[k] for k in ('state_hashes','effective_iterations','stop_reason','accuracy_unlabeled'))==ref for r in gg)
            groups.append(dict(dataset=case['name'],seed=case['label_seed'],backend=b,iterations=gg[0]['effective_iterations'],stop_reason=gg[0]['stop_reason'],censored=gg[0]['stop_reason']=='max_iter',accuracy_percent=100*gg[0]['accuracy_unlabeled'],state_hashes=gg[0]['state_hashes'],propagation_s=stats([r['propagation_s'] for r in gg]),fit_predict_s=stats([r['fit_predict_s'] for r in gg]),task_ids=[r['task_id'] for r in gg]))
    stages=[read('docs/benchmarks/early_stopping_full_seed10/results/analysis.json'),old]
    for seed in range(12,20):
        stage=read(provenance/f'seed{seed}'/'audit.json')
        assert stage['audit']=='passed' and stage['cumulative_measurements']==(seed-9)*360
        assert all(sha(root/p)==h for p,h in read(provenance/f'seed{seed}'/'prior-checkpoints.json').items())
        stages.append(stage)
    sessions=[]; warmups=[]
    for directory in (root/'sessions').iterdir():
        s=read_record(directory/'session.json'); end=read_record(directory/'end.json'); w=[read_record(p) for p in (directory/'warmups').iterdir()]
        assert len(w)==120 and end['measured_tasks_completed']==360
        assert end['status']==('complete' if s['stage_seed']==19 else 'partial')
        assert not s['manifest']['repository']['dirty'] and s['identity']==plan['identity']
        sessions.append(dict(stage_seed=s['stage_seed'],manifest=s['manifest'],started_utc=s['started_utc'],**end)); warmups.extend(w)
    assert sorted(s['stage_seed'] for s in sessions)==list(range(10,20)) and len(warmups)==1200
    summary=read_record(root/'summaries'/next(e['generation'] for e in reversed(events) if e['kind']=='summary')/'summary.json')
    assert summary['measurements']==3600 and summary['status']=='complete'
    assert all(v==dict(expected=360,completed=360,status='complete') for v in summary['stages'].values())
    by_dataset_backend=[]
    for name in [v['name'] for v in plan['cases'][:20]]:
        for b in BACKENDS:
            gg=[g for g in groups if g['dataset']==name and g['backend']==b]; assert len(gg)==10
            rr=[r for r in rows if cases[r['case_id']]['name']==name and r['backend']==b]; assert len(rr)==60
            acc=[g['accuracy_percent'] for g in gg]; it=[g['iterations'] for g in gg]
            by_dataset_backend.append(dict(dataset=name,backend=b,label_masks=10,timing_repetitions_per_mask=6,measured_fits=60,early_stopping_masks=sum(not g['censored'] for g in gg),censored_masks=sum(g['censored'] for g in gg),censor_fraction=sum(g['censored'] for g in gg)/10,iterations=stats(it),accuracy_percent=dict(mean=float(np.mean(acc)),sample_sd=float(np.std(acc,ddof=1)),**stats(acc)),propagation_repetitions_s=stats([r['propagation_s'] for r in rr]),fit_repetitions_s=stats([r['fit_predict_s'] for r in rr]),propagation_seed_medians_s=stats([g['propagation_s']['median'] for g in gg]),fit_seed_medians_s=stats([g['fit_predict_s']['median'] for g in gg]),propagation_total_s=sum(r['propagation_s'] for r in rr),fit_total_s=sum(r['fit_predict_s'] for r in rr)))
    backend_summary=[]
    for b in BACKENDS:
        rr=[r for r in rows if r['backend']==b]; gg=[g for g in groups if g['backend']==b]
        backend_summary.append(dict(backend=b,measured_fits=len(rr),early_stops=sum(r['stop_reason']=='early_stop' for r in rr),censored=sum(r['stop_reason']=='max_iter' for r in rr),censor_fraction=sum(r['stop_reason']=='max_iter' for r in rr)/len(rr),iterations=sum(r['effective_iterations'] for r in rr),propagation_total_s=sum(r['propagation_s'] for r in rr),fit_total_s=sum(r['fit_predict_s'] for r in rr),accuracy_macro_percent=float(np.mean([g['accuracy_percent'] for g in gg])),accuracy_seed_macro_percent=stats([float(np.mean([g['accuracy_percent'] for g in gg if g['seed']==seed])) for seed in range(10,20)])))
    comparisons=[]
    for name in [v['name'] for v in plan['cases'][:20]]:
        for a,b in (('numpy','numba'),('numpy','cython'),('numba','cython')):
            aa=[next(g for g in groups if g['dataset']==name and g['backend']==a and g['seed']==s) for s in range(10,20)]
            bb=[next(g for g in groups if g['dataset']==name and g['backend']==b and g['seed']==s) for s in range(10,20)]
            delta=[y['accuracy_percent']-x['accuracy_percent'] for x,y in zip(aa,bb)]
            comparisons.append(dict(dataset=name,reference_backend=a,comparison_backend=b,accuracy_difference_pp=dict(mean=float(np.mean(delta)),sample_sd=float(np.std(delta,ddof=1)),**stats(delta)),ratio_reference_over_comparison_fit_medians=stats([x['fit_predict_s']['median']/y['fit_predict_s']['median'] for x,y in zip(aa,bb)]),note='Backend-specific trajectories/RNG and stopping iterations differ; ratios describe complete fits, not equal-work kernel speedups.'))
    permutations_summary=[]; position_summary=[]
    group_lookup={(g['dataset'],g['seed'],g['backend']):g for g in groups}
    def ratios(rr): return [r['fit_predict_s']/group_lookup[(cases[r['case_id']]['name'],cases[r['case_id']]['label_seed'],r['backend'])]['fit_predict_s']['median'] for r in rr]
    for b in BACKENDS:
        for order in permutations(BACKENDS):
            rr=[r for r in rows if r['backend']==b and tuple(r['execution_order'])==order]; assert len(rr)==200
            permutations_summary.append(dict(backend=b,execution_order=list(order),measured_fits=200,normalized_fit=stats(ratios(rr))))
        for pos in range(3):
            rr=[r for r in rows if r['backend']==b and r['order_index']==pos]; assert len(rr)==400
            position_summary.append(dict(backend=b,position=pos,measured_fits=400,normalized_fit=stats(ratios(rr))))
    completed={e['task_id']:e for e in events if e['kind']=='completed'}
    fit=sum(r['fit_predict_s'] for r in rows); prop=sum(r['propagation_s'] for r in rows)
    wall=sum(s['execution']['wall_clock_s'] for s in stages)
    publication=stats([seconds(r['call_completed_utc'],completed[r['task_id']]['timestamp_utc']) for r in rows])
    flat=[dict(dataset=cases[r['case_id']]['name'],seed=cases[r['case_id']]['label_seed'],backend=r['backend'],repeat=r['repeat'],task_id=r['task_id'],iterations=r['effective_iterations'],stop_reason=r['stop_reason'],cap_hit=r['reached_max_iter'],accuracy_percent=100*r['accuracy_unlabeled'],propagation_s=r['propagation_s'],fit_predict_s=r['fit_predict_s'],fixture_sha256=r['fixture_sha256'],state_hashes=json.dumps(r['state_hashes'],sort_keys=True),session_id=r['session_id'],execution_order=','.join(r['execution_order']),order_index=r['order_index']) for r in rows]
    result=dict(audit='passed',plan_sha256=digest(plan),specification_sha256=digest(full_specification()),original_runtime_source_commit=plan['manifest']['repository']['commit'],measurements=3600,warmups=1200,fixtures=200,fixture_backend_groups=600,attempts=3600,pending=0,early_stops=sum(r['stop_reason']=='early_stop' for r in rows),censored=sum(r['stop_reason']=='max_iter' for r in rows),cap_hits=sum(r['reached_max_iter'] for r in rows),iterations=sum(r['effective_iterations'] for r in rows),native_stage_wall_total_s=wall,propagation_total_s=prop,fit_total_s=fit,overhead=dict(native_wall_less_measured_fit_s=wall-fit,warmup_fit_s=sum(w['row']['fit_predict_s'] for w in warmups),validation_publication_window_s=publication,note='Residual includes warm-ups, preparations, seeding, validation, checkpoints and audits. Pure I/O is not separately timed; publication intervals overlap residual.'),backend_summary=backend_summary,dataset_backend_summary=by_dataset_backend,groups=groups,backend_comparisons=comparisons,permutation_summary=permutations_summary,position_summary=position_summary,stages=stages,sessions=sessions,fixture_audit=fixtures,rows=flat,timing_variability_ranked=sorted([dict(dataset=g['dataset'],seed=g['seed'],backend=g['backend'],fit_range_s=g['fit_predict_s']['maximum']-g['fit_predict_s']['minimum'],iqr_over_median=g['fit_predict_s']['iqr']/g['fit_predict_s']['median'],maximum_over_minimum=g['fit_predict_s']['maximum']/g['fit_predict_s']['minimum']) for g in groups],key=lambda v:v['fit_range_s'],reverse=True),outliers_excluded=0,anomalies=[],audit_duration_s=time.perf_counter()-begin)

    remaining_stages=stages[2:]
    gaps=[dict(after_seed=before['stage_seed'],before_seed=after['stage_seed'],
        elapsed_utc_s=seconds(before['execution']['end_utc'],after['execution']['start_utc']))
        for before,after in zip(remaining_stages,remaining_stages[1:])]
    result['remaining_sequence_runtime']=dict(
        native_wall_s=sum(s['execution']['wall_clock_s'] for s in remaining_stages),
        elapsed_utc_s=seconds(remaining_stages[0]['execution']['start_utc'],remaining_stages[-1]['execution']['end_utc']),
        between_stage_gaps=gaps,between_stage_gap_total_s=sum(g['elapsed_utc_s'] for g in gaps),
        scope='Seeds 12--19 start to final native exit, includes between-stage audits/preflight. Initial 720-result preflight and final post-stage audit/reporting excluded. UTC gaps are distinct from monotonic native process timings.')
    resources=[]
    for seed in range(12,20):
        monitor=[__import__('json').loads(line) for line in (provenance/f'seed{seed}'/'monitor.jsonl').read_text().splitlines()]
        assert all(v['stop_reason'] is None for v in monitor)
        resources.append(dict(seed=seed,minimum_available_ram_bytes=min(v['available_ram_bytes'] for v in monitor),minimum_free_disk_bytes=min(v['free_disk_bytes'] for v in monitor)))
    result['remaining_stage_resources']=resources
    result['timing_caveat']='Large timing changes/outliers are retained. Within a fixture/backend all six state hashes, stopping counts/reasons and accuracies agree. Seed order and elapsed machine time are confounded; CPU load, temperature, power and affinity were not instrumented in this frozen protocol, so timing causes cannot be established.'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--provenance',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=analyze(a.root,a.baseline,a.provenance); a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (a.output/'results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(result['rows'][0])); w.writeheader(); w.writerows(result['rows'])
    c=Controller(Store(a.root)); state=c.audit()
    with (a.output/'measurements.jsonl').open('w',encoding='utf-8') as f:
        for i in range(3600): f.write(json.dumps(state['results'][f't{i:04d}']['row'],sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('audit','measurements','early_stops','censored','native_stage_wall_total_s')}))

if __name__=='__main__': main()
