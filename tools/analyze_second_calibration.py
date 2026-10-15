"""Offline strict controller audit, original-fixture comparison and projections."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmarks.checkpoints import Store, read_record, file_inventory
from benchmarks.resumable import Controller, specification


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(root, baseline, first_pilot, execution):
    controller=Controller(Store(root))
    # Historical analysis is read-only: a later controller change must not
    # impersonate the measured execution identity. Validate archived source
    # bytes explicitly; live execution/resume still demands an exact identity.
    state=controller.audit(check_environment=False)
    for relative,expected_hash in state['plan']['manifest']['repository']['source_hashes'].items():
        snapshot=root.parent/'source_files'/relative
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=expected_hash:
            raise ValueError(f'measured source snapshot missing or changed: {relative}')
    extension=state['plan']['manifest']['cython_extension']
    if hashlib.sha256(Path(extension['path']).read_bytes()).hexdigest()!=extension['sha256']:
        raise ValueError('measured Cython binary changed; restore the identified artifact for audit')
    if state['pending'] or state['ambiguous'] or state['reconcile']:
        raise ValueError('campaign incomplete or unreconciled')
    if state['plan']['protocol']!='remaining_historical_seed10':
        raise ValueError('not the approved second calibration')
    rows=[state['results'][t['task_id']]['row'] for t in state['plan']['tasks']]
    if len(rows)!=48:
        raise ValueError('require exactly 48 results')
    for case in state['plan']['cases']:
        case_id=f"{case['name']}-labels10"
        actual=controller.fixture_cache[case_id][1]
        expected=read(baseline/'fixtures'/case_id/'fixture.json')
        if actual['fixture_sha256']!=expected['fixture_sha256'] or actual['hashes']!=expected['hashes']:
            raise ValueError(f'PR #16 fixture differs: {case_id}')
    events=controller.store.events()
    attempts=sum(e['kind']=='started' for e in events)
    retry_events=[e for e in events if e['kind']=='retry_authorized']
    if attempts!=48+len(retry_events):
        raise ValueError('attempt/retry inventory inconsistent')
    sessions=[]
    warmups=[]
    for directory in sorted((root/'sessions').iterdir()):
        session=read_record(directory/'session.json')
        end=read_record(directory/'end.json') if (directory/'end.json').exists() else dict(
            session_id=directory.name,status='interrupted',ended_utc=None,elapsed_s=None)
        warms=[read_record(p) for p in sorted((directory/'warmups').iterdir())]
        warmups.extend(warms)
        sessions.append(dict(start_utc=session['started_utc'],
                             **end,seed_helper=read_record(directory/'seed_helper.json')
                             if (directory/'seed_helper.json').exists() else None))
    if len(warmups)<96:
        raise ValueError('missing warm-up inventory')
    fixture_preparation={cid:meta['timings'] for cid,(_,meta) in controller.fixture_cache.items()}
    first=read(first_pilot/'analysis.json')
    original_manifest=read(first_pilot/'raw/manifest.json')
    source_hashes=state['plan']['manifest']['repository']['source_hashes']
    for name,expected_hash in original_manifest['repository']['source_hashes'].items():
        if name.startswith('pcc') or name in ('benchmarks/run.py','benchmarks/fixtures.py','benchmarks/runner.py','benchmarks/historical.py'):
            if source_hashes[name]!=expected_hash:
                raise ValueError(f'production or existing harness source changed: {name}')
    if state['plan']['manifest']['repository']['dirty']:
        raise ValueError('calibration source was not clean at launch')
    first_groups=first['groups']
    baseline_summary=read(baseline/'summary.json')
    charges=first['full_campaign']['overhead_charges_s']
    # 16 single-mask observations x ten masks x six repeats. Four first-pilot
    # datasets have two masks: sum their medians x five masks x six repeats.
    observed_projection=60*sum(r['propagation_s'] for r in rows)+30*sum(r['propagation_s_median'] for r in first_groups)
    observed_fit_projection=60*sum(r['fit_predict_s'] for r in rows)+30*sum(r['fit_predict_s_median'] for r in first_groups)
    # Repeats are reset deterministic trajectories, not independent samples.
    actual_rate_cap=60*sum(r['propagation_s']/r['effective_iterations']*500000 for r in rows)
    actual_rate_cap+=30*sum(r['propagation_s_median']/r['iterations']*500000 for r in first_groups)
    first_raw=[json.loads(line) for line in (first_pilot/'raw/measurements.jsonl').read_text().splitlines()]
    first_rates={}
    for r in first_raw:
        key=(r['case_id'],r['backend'])
        first_rates[key]=max(first_rates.get(key,0),r['propagation_s_per_iteration'])
    maximum_observed_rate_cap=60*sum(r['propagation_s_per_iteration']*500000 for r in rows)+30*sum(v*500000 for v in first_rates.values())
    historical_cap=300*sum(r['propagation_s']['median'] for r in baseline_summary)+charges
    historical_max_cap=300*sum(r['propagation_s']['maximum'] for r in baseline_summary)+charges
    projections=dict(observed_stopping_propagation_s=observed_projection,
        observed_stopping_fit_s=observed_fit_projection,
        observed_stopping_accounted_s=observed_projection+charges,
        actual_rate_median_cap_s=actual_rate_cap+charges,
        actual_rate_maximum_cap_s=maximum_observed_rate_cap+charges,
        historical_median_cap_s=historical_cap,historical_maximum_cap_s=historical_max_cap,
        conservative_cap_s=1.5*max(historical_max_cap,maximum_observed_rate_cap+charges),
        historical_additional_charges_s=charges,
        limitations='Single seed for 16 datasets and two seeds for four; other masks remain unobserved. Stopping-based forecast includes censored cap outcomes as capped work, not predicted convergence. Fixed and measured per-iteration rates are state dependent. 50% headroom is a scenario, not a cutoff or bound. Startup, fsync and restart costs remain uncertain; overhead charges retain original baseline for comparability.')
    table=[]
    for row in rows:
        table.append(dict(dataset=row['case_id'].removesuffix('-labels10'),seed=10,backend=row['backend'],
            iterations=row['effective_iterations'],reason=row['stop_reason'],cap_hit=row['reached_max_iter'],
            censored=row['stop_reason']=='max_iter',accuracy_percent=100*row['accuracy_unlabeled'],
            propagation_s=row['propagation_s'],fit_s=row['fit_predict_s'],
            session_id=row['session_id'],task_id=row['task_id']))
    preserved=read(execution.parent/'prior-artifact-preservation.json')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in preserved.items()):
        raise ValueError('prior artifact changed')
    accuracy_differences=[]
    for case in state['plan']['cases']:
        name=case['name']
        values={r['backend']:r['accuracy_percent'] for r in table if r['dataset']==name}
        accuracy_differences.append(dict(dataset=name,accuracy_percent=values,
            spread_percentage_points=max(values.values())-min(values.values())))
    return dict(audit='passed',source_commit=state['plan']['manifest']['repository']['commit'],
        accuracy_differences=accuracy_differences,
        measurements=48,warmups=len(warmups),attempts=attempts,retry_events=retry_events,sessions=sessions,rows=table,
        actual_iterations=sum(r['effective_iterations'] for r in rows),
        cap_hits=sum(r['reached_max_iter'] for r in rows),
        censored=sum(r['stop_reason']=='max_iter' for r in rows),
        early_stops=sum(r['stop_reason']=='early_stop' for r in rows),
        propagation_s=sum(r['propagation_s'] for r in rows),fit_s=sum(r['fit_predict_s'] for r in rows),
        warmup_fit_s=sum(w['row']['fit_predict_s'] for w in warmups),
        fixture_preparation=fixture_preparation,execution=read(execution),full_campaign=projections,
        prior_files_verified=len(preserved),outliers_excluded=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--first-pilot',type=Path,default=Path('docs/benchmarks/early_stopping_pilot/results'))
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root,args.baseline,args.first_pilot,args.execution)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (args.output/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','fixture_preparation')},indent=2))


if __name__=='__main__':
    main()
