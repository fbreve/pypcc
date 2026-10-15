"""Read-only audit of seed 11 and all 720 completed full-campaign results."""
import argparse
from collections import Counter
import csv
import json
from itertools import permutations
from pathlib import Path
import time

import numpy as np
from benchmarks.checkpoints import Store, read_record, digest, file_inventory
from benchmarks.resumable import Controller, full_specification
from tools.analyze_full_seed10 import read, sha, seconds, stats

if not __debug__:
    raise RuntimeError('Strict audit requires Python assertion checks enabled')


def analyze(root,baseline,provenance):
    begin=time.perf_counter(); c=Controller(Store(root)); state=c.audit(check_environment=False)
    plan=state['plan']; old=read('docs/benchmarks/early_stopping_full_seed10/results/analysis.json')
    assert plan['protocol']=='historical_early_stopping_full_v1' and digest(plan)==old['plan_sha256']
    assert digest(full_specification())==old['specification_sha256']
    for relative,h in plan['identity']['source_hashes'].items():
        assert sha(provenance/'source_files'/relative)==h,relative
    extension=plan['identity']['cython_extension']; assert sha(extension['path'])==extension['sha256']
    assert len(state['results'])==720 and set(state['results'])=={f't{i:04d}' for i in range(720)}
    assert len(state['pending'])==2880 and not state['ambiguous'] and not state['reconcile'] and not state['authorized_retries']
    cases={v['name']+'-labels'+str(v['label_seed']):v for v in plan['cases']}
    tasks=[t for t in plan['tasks'] if cases[t['case_id']]['label_seed']==11]
    assert [t['task_id'] for t in tasks]==[f't{i:04d}' for i in range(360,720)]
    rows=[state['results'][t['task_id']]['row'] for t in tasks]
    all_rows=[state['results'][f't{i:04d}']['row'] for i in range(720)]
    events=c.store.events(); starts=[e for e in events if e['kind']=='started']
    assert len(starts)==720 and Counter(e['task_id'] for e in starts)=={f't{i:04d}':1 for i in range(720)}
    errors=[e for e in events if e['kind'] in ('failed','retry_authorized','fixture_retry','lock_recovered','reconciled','summary_retired')]
    assert not errors and not (root/'writer.lock').exists()
    original=read(provenance/'prior-checkpoint-preservation.json')
    assert all(sha(root/p)==h for p,h in original.items())
    preserved=read(provenance/'prior-artifact-preservation.json')
    assert all(sha(p)==h for p,h in preserved.items())
    groups=[]; fixtures=[]
    for cid,case in cases.items():
        if case['label_seed'] not in (10,11):
            assert not (root/'fixtures'/cid).exists()
            continue
        meta=c.fixture_cache[cid][1]; previous=read(baseline/'fixtures'/cid/'fixture.json')
        assert meta['hashes']==previous['hashes'] and meta['fixture_sha256']==previous['fixture_sha256'],cid
        if case['label_seed']==10: continue
        anchor=c.fixture_cache[case['name']+'-labels10'][1]
        assert meta['preparation']['reused'] and meta['preparation']['source_case_id']==case['name']+'-labels10'
        assert meta['timings']['dataset_load_s']==0 and meta['timings']['graph_build_s']==0
        assert all(h==anchor['hashes'][field] for field,h in meta['hashes'].items() if field!='labels')
        fixtures.append(dict(case_id=cid,fixture_sha256=meta['fixture_sha256'],hashes=meta['hashes'],
            timings=meta['timings'],preparation=meta['preparation'],matches_frozen_and_pr16=True))
        rr=[r for r in rows if r['case_id']==cid]; assert len(rr)==18
        assert {tuple(r['execution_order']) for r in rr}==set(permutations(('numpy','numba','cython')))
        for backend in ('numpy','numba','cython'):
            gg=[r for r in rr if r['backend']==backend]; assert len(gg)==6
            assert {r['repeat'] for r in gg}==set(range(6)) and Counter(r['order_index'] for r in gg)=={0:2,1:2,2:2}
            reference=(gg[0]['state_hashes'],gg[0]['effective_iterations'],gg[0]['stop_reason'],gg[0]['accuracy_unlabeled'])
            assert all((r['state_hashes'],r['effective_iterations'],r['stop_reason'],r['accuracy_unlabeled'])==reference for r in gg)
            groups.append(dict(dataset=case['name'],backend=backend,repetitions=6,
                iterations=gg[0]['effective_iterations'],stop_reason=gg[0]['stop_reason'],
                accuracy_percent=100*gg[0]['accuracy_unlabeled'],state_hashes=gg[0]['state_hashes'],
                cap_hits=sum(r['reached_max_iter'] for r in gg),censored=sum(r['stop_reason']=='max_iter' for r in gg),
                propagation_s=stats([r['propagation_s'] for r in gg]),fit_predict_s=stats([r['fit_predict_s'] for r in gg]),
                task_ids=[r['task_id'] for r in gg]))
    sessions=[]; warmups=[]; cumulative_warmups=0
    for directory in (root/'sessions').iterdir():
        session=read_record(directory/'session.json'); end=read_record(directory/'end.json')
        warms=[read_record(p) for p in (directory/'warmups').iterdir()]; cumulative_warmups+=len(warms)
        assert session['stage_seed'] in (10,11) and end['status']=='partial' and end['measured_tasks_completed']==360
        if session['stage_seed']==11:
            assert not session['manifest']['repository']['dirty']
            sessions.append(dict(start_utc=session['started_utc'],resume_commit=session['manifest']['repository']['commit'],**end,
                seed_helper=read_record(directory/'seed_helper.json')))
            warmups.extend(warms)
    assert len(sessions)==1 and len(warmups)==120 and cumulative_warmups==240
    last=next(e for e in reversed(events) if e['kind']=='summary')
    summary=read_record(root/'summaries'/last['generation']/'summary.json')
    assert summary['status']=='partial' and summary['measurements']==720
    assert all(summary['stages'][str(s)]==dict(expected=360,completed=360,status='complete') for s in (10,11))
    assert all(summary['stages'][str(s)]['status']=='not_started' for s in range(12,20))
    execution=read(provenance/'execution.json')
    assert execution['exit_code']==0 and execution['stop_reason'] is None and execution['runtime_limit'] is None
    completed={e['task_id']:e for e in events if e['kind']=='completed'}
    publication=[seconds(r['call_completed_utc'],completed[r['task_id']]['timestamp_utc']) for r in rows]
    helper=sum(seconds(r['call_started_utc'],r['call_completed_utc']) for r in rows)
    fit=sum(r['fit_predict_s'] for r in rows); prop=sum(r['propagation_s'] for r in rows)
    warmfit=sum(w['row']['fit_predict_s'] for w in warmups)
    prep=sum(t for f in fixtures for t in f['timings'].values())
    seed=sessions[0]['seed_helper']['first_call_s']; elapsed=sessions[0]['elapsed_s']
    overhead=dict(measured_helper_wall_s=helper,measured_helper_untimed_checks_s=helper-fit,
        validation_and_result_publication_window_s=stats(publication),warmup_fit_s=warmfit,
        fixture_preparation_s=prep,seed_helper_first_call_s=seed,controller_session_elapsed_s=elapsed,
        session_other_s=elapsed-fit-warmfit-prep-seed,native_wall_outside_session_s=execution['wall_clock_s']-elapsed,
        native_wall_less_fit_warmup_preparation_seed_s=execution['wall_clock_s']-fit-warmfit-prep-seed,
        note='Overlapping measures. Residual includes verification, model/graph copying and RNG seeding, checkpoint/session/ledger I/O and audits/summaries; pure filesystem I/O is not separately timed. UTC helper/publication spans and monotonic timers can differ slightly.')
    comparisons=[]
    for g in groups:
        prior=next(v for v in old['groups'] if v['dataset']==g['dataset'] and v['backend']==g['backend'])
        comparisons.append(dict(dataset=g['dataset'],backend=g['backend'],seed10_iterations=prior['iterations'],
            seed11_iterations=g['iterations'],iteration_change=g['iterations']-prior['iterations'],
            seed10_reason=prior['stop_reason'],seed11_reason=g['stop_reason'],
            seed10_accuracy_percent=prior['accuracy_percent'],seed11_accuracy_percent=g['accuracy_percent'],
            accuracy_change_percentage_points=g['accuracy_percent']-prior['accuracy_percent'],
            seed10_propagation_median_s=prior['propagation_s']['median'],seed11_propagation_median_s=g['propagation_s']['median'],
            seed10_fit_median_s=prior['fit_predict_s']['median'],seed11_fit_median_s=g['fit_predict_s']['median'],
            fit_median_ratio=g['fit_predict_s']['median']/prior['fit_predict_s']['median']))
    pilot=[json.loads(line) for line in Path('docs/benchmarks/early_stopping_pilot/results/raw/measurements.jsonl').read_text().splitlines()]
    observed_pilot=[r for r in pilot if r['case_id'].endswith('-labels11')]
    pilot_differences=[]
    for r in observed_pilot:
        current=next(v for v in rows if v['case_id']==r['case_id'] and v['backend']==r['backend'])
        changed=[f for f in ('state_hashes','effective_iterations','stop_reason','accuracy_unlabeled') if r[f]!=current[f]]
        if changed: pilot_differences.append(dict(case_id=r['case_id'],backend=r['backend'],fields=changed))
    backend_summary=[]; dataset_summary=[]
    for field,keys,output in [('backend',('numpy','numba','cython'),backend_summary),
                            ('dataset',[v['name'] for v in plan['cases'][:20]],dataset_summary)]:
        for key in keys:
            rr=[r for r in rows if (r['backend'] if field=='backend' else cases[r['case_id']]['name'])==key]
            gg=[g for g in groups if g[field]==key]
            output.append(dict(**{field:key},measurements=len(rr),iterations=sum(r['effective_iterations'] for r in rr),
                early_stops=sum(r['stop_reason']=='early_stop' for r in rr),censored=sum(r['stop_reason']=='max_iter' for r in rr),
                censor_fraction=sum(r['stop_reason']=='max_iter' for r in rr)/len(rr),
                propagation_total_s=sum(r['propagation_s'] for r in rr),fit_total_s=sum(r['fit_predict_s'] for r in rr),
                accuracy_macro_percent=float(np.mean([g['accuracy_percent'] for g in gg])),
                accuracy_spread_percentage_points=max(g['accuracy_percent'] for g in gg)-min(g['accuracy_percent'] for g in gg)))
    permutation_summary=[]
    for backend in ('numpy','numba','cython'):
        for position in range(3):
            rr=[r for r in rows if r['backend']==backend and r['order_index']==position]
            assert len(rr)==40
            ratios=[r['fit_predict_s']/next(g['fit_predict_s']['median'] for g in groups if g['dataset']==cases[r['case_id']]['name'] and g['backend']==backend) for r in rr]
            permutation_summary.append(dict(backend=backend,position=position,measurements=40,
                fit_over_fixture_backend_median=stats(ratios),
                note='Descriptive ratio, not independent samples or causal position effect; position and time/order are confounded.'))
    execution_permutation_summary=[]
    for order in permutations(('numpy','numba','cython')):
        for backend in ('numpy','numba','cython'):
            rr=[r for r in rows if tuple(r['execution_order'])==order and r['backend']==backend]
            assert len(rr)==20
            ratios=[r['fit_predict_s']/next(g['fit_predict_s']['median'] for g in groups if g['dataset']==cases[r['case_id']]['name'] and g['backend']==backend) for r in rr]
            execution_permutation_summary.append(dict(order=list(order),backend=backend,position=order.index(backend),
                measurements=20,fit_over_fixture_backend_median=stats(ratios)))
    baseline_summary=read(baseline/'summary.json')
    historical=[v for v in baseline_summary if int(v['case_id'].rsplit('labels',1)[1])>=12]
    assert len(historical)==480
    overhead_stage=(execution['wall_clock_s']-fit+old['execution']['wall_clock_s']-old['fit_predict_s'])/2
    median_cap=max_cap=0.
    for dataset in [v['name'] for v in plan['cases'][:20]]:
        for backend in ('numpy','numba','cython'):
            rates=[r['propagation_s']/r['effective_iterations'] for r in all_rows if cases[r['case_id']]['name']==dataset and r['backend']==backend]
            assert len(rates)==12
            median_cap+=float(np.median(rates))*500000*6*8
            max_cap+=max(rates)*500000*6*8
    historical_cap={f:sum(300*v['propagation_s'][f] for v in historical)+8*overhead_stage for f in ('median','maximum')}
    median_cap+=8*overhead_stage; max_cap+=8*overhead_stage
    remaining=dict(seeds=list(range(12,20)),measured_fits=2880,uninterrupted_warmups=960,
        measured_iteration_cap=1440000000,observed_stage_mean_wall_s=4*(old['execution']['wall_clock_s']+execution['wall_clock_s']),
        observed_stage_wall_range_s=[8*min(old['execution']['wall_clock_s'],execution['wall_clock_s']),8*max(old['execution']['wall_clock_s'],execution['wall_clock_s'])],
        observed_median_rate_cap_s=median_cap,observed_maximum_rate_cap_s=max_cap,
        historical_median_rate_cap_s=historical_cap['median'],historical_maximum_rate_cap_s=historical_cap['maximum'],
        conservative_cap_s=1.5*max(max_cap,historical_cap['maximum']),
        overhead_charge_per_stage_s=overhead_stage,
        limitations='Only two label masks observed per dataset. Other masks can change iterations/censoring/accuracy. Stage-wall extrapolation includes all outliers; normalized cap rates are state dependent. Historical fixed-work rates cover exactly seeds12–19. Headroom is a planning scenario, not a bound/cutoff. Audits grow with ledger/fixtures/summaries; future interruption/verification costs require allowance.')
    flat=[dict(dataset=cases[r['case_id']]['name'],seed=11,backend=r['backend'],repeat=r['repeat'],task_id=r['task_id'],
        iterations=r['effective_iterations'],reason=r['stop_reason'],cap_hit=r['reached_max_iter'],censored=r['stop_reason']=='max_iter',
        accuracy_percent=100*r['accuracy_unlabeled'],propagation_s=r['propagation_s'],fit_predict_s=r['fit_predict_s'],
        fixture_sha256=r['fixture_sha256'],state_hashes=json.dumps(r['state_hashes'],sort_keys=True),session_id=r['session_id'],
        execution_order=','.join(r['execution_order']),order_index=r['order_index']) for r in rows]
    monitor=[json.loads(line) for line in (provenance/'monitor.jsonl').read_text().splitlines()]
    return dict(audit='passed',stage_seed=11,original_campaign_source_commit=plan['manifest']['repository']['commit'],
        resume_source_commit=sessions[0]['resume_commit'],runtime_source_binary_environment_match_original=True,
        plan_sha256=digest(plan),specification_sha256=digest(full_specification()),measurements=360,warmups=120,fixtures=20,
        cumulative_measurements=720,cumulative_warmups=240,cumulative_fixtures=40,pending_unauthorized_tasks=2880,
        cumulative_iterations=sum(r['effective_iterations'] for r in all_rows),cumulative_early_stops=sum(r['stop_reason']=='early_stop' for r in all_rows),
        cumulative_censored=sum(r['stop_reason']=='max_iter' for r in all_rows),sessions=sessions,attempts=360,anomalies=errors,
        early_stops=sum(r['stop_reason']=='early_stop' for r in rows),censored=sum(r['stop_reason']=='max_iter' for r in rows),
        cap_hits=sum(r['reached_max_iter'] for r in rows),iterations=sum(r['effective_iterations'] for r in rows),
        propagation_s=prop,fit_predict_s=fit,groups=groups,rows=flat,backend_summary=backend_summary,dataset_summary=dataset_summary,
        seed_comparisons=comparisons,permutation_summary=permutation_summary,
        execution_permutation_summary=execution_permutation_summary,
        timing_variability_ranked=sorted([dict(dataset=g['dataset'],backend=g['backend'],fit_range_s=g['fit_predict_s']['maximum']-g['fit_predict_s']['minimum'],
            maximum_over_minimum=g['fit_predict_s']['maximum']/g['fit_predict_s']['minimum'],iqr_over_median=g['fit_predict_s']['iqr']/g['fit_predict_s']['median']) for g in groups],key=lambda v:v['fit_range_s'],reverse=True),
        original_checkpoint_files_unchanged=len(original),prior_artifact_files_unchanged=len(preserved),fixture_audit=fixtures,
        first_pilot_seed11_comparison_groups=12,first_pilot_seed11_differences=pilot_differences,
        overhead=overhead,execution=execution,remaining_runtime=remaining,outliers_excluded=0,
        resources=dict(minimum_available_ram_bytes=min(v['available_ram_bytes'] for v in monitor),minimum_free_disk_bytes=min(v['free_disk_bytes'] for v in monitor)),
        audit_duration_s=time.perf_counter()-begin)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','baseline','provenance','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); a=analyze(args.root,args.baseline,args.provenance)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(a,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (args.output/'results.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(a['rows'][0])); writer.writeheader(); writer.writerows(a['rows'])
    print(json.dumps({k:v for k,v in a.items() if k not in ('groups','rows','fixture_audit','seed_comparisons')},indent=2))


if __name__=='__main__': main()
