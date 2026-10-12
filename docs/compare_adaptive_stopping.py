"""Frozen retrospective comparison using only authenticated public PR #8 files."""
import csv
import argparse
import os
import gzip
import hashlib
import io
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.patience_archive import PublishedArtifacts, sha
from experiments.patience_results import audit_checks
from experiments.adaptive_stopping import original_event, classification_event, record_stagnation

ROOT=Path('docs/benchmarks/adaptive_stopping')
SOURCE=Path('docs/benchmarks/stopping_patience')


def write_csv(name, rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def describe(values):
    return dict(mean=statistics.mean(values) if values else None,
        sd=statistics.stdev(values) if len(values)>1 else (0 if values else None),
        median=statistics.median(values) if values else None,
        minimum=min(values) if values else None,maximum=max(values) if values else None)


def summaries(rows, dataset=False):
    groups={}
    for r in rows:
        key=(r['dataset'],r['configuration']) if dataset else (r['configuration'],)
        groups.setdefault(key,[]).append(r)
    out=[]
    for key,rr in groups.items():
        first=rr[0];r={k:first[k] for k in ('configuration','candidate','schedule','window','budget','is_primary')}
        if dataset:r['dataset']=first['dataset']
        r.update(cases=len(rr),stopped=sum(x['status']=='stopped' for x in rr),
            censored=sum(x['status']=='right_censored' for x in rr),
            observed_stop_accuracy=sum(x['stop_accuracy'] is not None for x in rr),
            event_cap_accuracy_available=sum(x['event_cap_accuracy'] is not None for x in rr))
        for field in ('observation_iteration','iterations_saved','iteration_difference_vs_A',
                      'accuracy_difference_vs_full','accuracy_difference_vs_A'):
            values=[x[field] for x in rr if x[field] is not None]
            r[field+'_n']=len(values)
            r.update({field+'_'+k:v for k,v in describe(values).items()})
        out.append(r)
    return out


def run():
    protocol_bytes=(ROOT/'proposed_protocol.json').read_bytes().replace(b'\r\n',b'\n')
    p=json.loads(protocol_bytes);freeze=json.loads((ROOT/'freeze.json').read_text())
    assert sha(protocol_bytes)==freeze['protocol_sha256']
    committed=subprocess.check_output(['git','show',freeze['protocol_commit']+':'+(ROOT/'proposed_protocol.json').as_posix()])
    assert sha(committed.replace(b'\r\n',b'\n'))==freeze['protocol_sha256']
    assert sha((SOURCE/'artifacts_manifest.json').read_bytes())==p['artifacts_manifest_sha256']
    inventory=json.loads((SOURCE/'inventory.json').read_text())['datasets']
    historical_hashes={(r['dataset'],int(r['seed'])):r for r in csv.DictReader((SOURCE/'archive_manifest.csv').open())}
    rows=[];baseline_matches=0;total_checks=0
    with PublishedArtifacts(SOURCE) as published:
        for info in inventory:
            for seed in p['development']['seeds']:
                path=f"trajectories/{info['name']}/{seed}/"
                complete=json.loads(published.read(path+'complete.json'))
                samples=list(csv.DictReader(io.StringIO(published.read(path+'samples.csv').decode())))
                packed=published.read(path+'checks.csv.gz');raw=gzip.decompress(packed)
                historical=historical_hashes[info['name'],seed]
                assert sha(packed)==historical['gzip_sha256'] and sha(raw)==historical['raw_sha256']
                checks=list(csv.DictReader(io.StringIO(raw.decode())))
                legacy=next(r for r in complete['rows'] if r['multiplier']==1)
                T=legacy['original_threshold'];cap=complete['iterations'];U=info['nodes']-info['labeled_nodes']
                audit=audit_checks(checks,T,[1.],cap);total_checks+=audit['checks']
                A=original_event(checks,T)
                assert (A['iteration'] if A else None)==legacy['stop_iteration']
                assert complete['seed']==seed and complete['dataset']==info['name'] and cap==500000
                baseline_matches+=1
                accuracies={int(r['iteration']):float(r['accuracy_unlabeled']) for r in samples}
                # Decisions receive only observable time/counts, never truth or accuracy.
                signals=[dict(iteration=int(r['iteration']),changed_nodes=int(r['changed_nodes'])) for r in samples]
                events=[('A','original',0,0,True,A)]
                for regular,schedule in ((True,'hybrid_regular'),(False,'all_observations_secondary')):
                    for candidate in ('B','C'):
                        for W in p['sensitivity']['windows']:
                            for budget in p['sensitivity']['sampled_transition_budgets']:
                                event=classification_event(signals,checks,W,budget,5000,U,T,joint=candidate=='C',regular=regular)
                                primary=regular and W==p['primary'][candidate]['window'] and budget==p['primary'][candidate]['sampled_transition_budget']
                                events.append((candidate,schedule,W,budget,primary,event))
                for W in p['sensitivity']['windows']:
                    for budget in p['sensitivity']['cumulative_record_gain_budgets']:
                        event=record_stagnation(checks,W,budget,5000)
                        primary=W==p['primary']['D']['window'] and budget==p['primary']['D']['cumulative_record_gain_budget']
                        events.append(('D','record_stagnation_check_grid',W,budget,primary,event))
                final=legacy['final_accuracy_unlabeled'];a_time=A['iteration'] if A else cap
                a_accuracy=accuracies[a_time]
                for candidate,schedule,W,budget,primary,event in events:
                    t=event['iteration'] if event else None;observed=t if t is not None else cap
                    acc=accuracies.get(observed)
                    if candidate=='C' and event:assert A and t>=A['iteration']
                    row=dict(dataset=info['name'],seed=seed,configuration=f'{candidate}:{schedule}:W{W}:q{budget:g}',
                        candidate=candidate,schedule=schedule,window=W,budget=budget,is_primary=primary,
                        status='stopped' if event else 'right_censored',stop_iteration=t,observation_iteration=observed,
                        iterations_saved=cap-observed,iteration_difference_vs_A=observed-a_time,
                        stop_accuracy=acc if event else None,event_cap_accuracy=acc,
                        accuracy_status='exact_saved_stop' if event and acc is not None else
                            ('unobserved_stop_missing' if event else 'full_cap_reference_not_stop'),
                        full_accuracy=final,A_event_cap_accuracy=a_accuracy,
                        accuracy_difference_vs_full=acc-final if acc is not None else None,
                        accuracy_difference_vs_A=acc-a_accuracy if acc is not None else None,
                        original_threshold=T,unknown_nodes=U,
                        window_start=event.get('window_start') if event else None,
                        window_end=event.get('window_end') if event else None,
                        actual_window_span=(event['window_end']-event['window_start']) if event and 'window_start' in event else None,
                        transition_count=event.get('transition_count') if event else None,
                        transition_rate=event.get('transition_rate') if event else None,
                        intermediate_observations=event.get('intermediate_observations') if event else None,
                        counter=event.get('counter') if event else None,
                        counter_iteration=event.get('counter_iteration') if event else None,
                        counter_age=event.get('counter_age') if event else None,
                        record_gain_sum=event.get('record_gain_sum') if event else None)
                    rows.append(row)
                print(f'Replayed {info["name"]} seed {seed}',flush=True)
    assert baseline_matches==200 and total_checks==10000000 and len(rows)==6800
    write_csv('individual.csv',rows)
    overall=summaries(rows);write_csv('overall_summary.csv',overall);write_csv('dataset_summary.csv',summaries(rows,True))
    primary=[r for r in rows if r['is_primary']]
    matched=[]
    for name in [i['name'] for i in inventory]:
        for seed in p['development']['seeds']:
            case=[r for r in primary if r['dataset']==name and r['seed']==seed]
            if all(r['event_cap_accuracy'] is not None for r in case):matched.extend(case)
    if matched:write_csv('primary_common_accuracy_cases.csv',matched)
    metadata=dict(protocol_commit=freeze['protocol_commit'],protocol_sha256=freeze['protocol_sha256'],
        artifacts_manifest_sha256=p['artifacts_manifest_sha256'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha(Path(__file__).read_bytes().replace(b'\r\n',b'\n')),
        observer_sha256=sha(Path('experiments/adaptive_stopping.py').read_bytes().replace(b'\r\n',b'\n')),
        archive_trajectories=200,check_rows_audited=total_checks,baseline_exact_matches=baseline_matches,
        configurations=34,individual_results=len(rows),primary_common_accuracy_cases=len(matched)//4,
        production_kernels_imported=any(k in sys.modules for k in ('pcc_numba','pcc_step','pcc')),
        propagation_executed=False,accuracy_imputed=False,parameters_selected=False)
    (ROOT/'validation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report(overall,metadata,rows)


def report(overall,metadata,rows):
    lines=['# Frozen exploratory adaptive stopping comparison','',
        f"Protocol commit `{metadata['protocol_commit']}`; SHA-256 `{metadata['protocol_sha256']}`.",'',
        'All 200 previously examined PR #8 cases; no new propagation, parameter selection or independent validation. [Frozen protocol](proposed_protocol.json) and [registration](freeze.json).','',
        'Primary B/C use hybrid observation: periodic decisions every 1000 iterations with all archived intermediate transitions summed once. Secondary B/C decide at every published observation. Neither implies stability between observations. D is the **record-stagnation criterion**, not general dominance stability.','',
        '## Primary results','',
        '| Candidate | Stops / censored | Mean event/cap iteration ± SD | Mean iterations saved (%) | Exact stop accuracy available | Event/cap accuracy difference vs full, mean ± SD (pp); n |',
        '|---|---:|---:|---:|---:|---:|']
    for r in overall:
        if not r['is_primary']:continue
        diff=r['accuracy_difference_vs_full_mean'];sd=r['accuracy_difference_vs_full_sd']
        accuracy=f'{100*diff:+.3f} ± {100*sd:.3f}; {r["accuracy_difference_vs_full_n"]}' if diff is not None else 'missing; 0'
        lines.append(f'| {r["candidate"]} | {r["stopped"]} / {r["censored"]} | {r["observation_iteration_mean"]:,.1f} ± {r["observation_iteration_sd"]:,.1f} | {100*r["iterations_saved_mean"]/500000:.2f} | {r["observed_stop_accuracy"]} / {r["stopped"]} | {accuracy} |')
    lines+=['','Censored cases contribute cap observations and zero iteration savings, not invented stop times. Cap reference accuracy is identified separately from actual stop accuracy. D missing stop accuracies remain blank; available-subset differences must not be interpreted as a full-cohort comparison.','',
        f"There are **{metadata['primary_common_accuracy_cases']} cases** with exact event/cap accuracy available for all four primary policies. Common-case rows are published when nonempty; exclusions can be audited against individual.csv. Each dataset has ten cases, so full-cohort rows weight datasets equally; missing-accuracy subsets need not.",'',
        '## Complete sensitivity and secondary results','',
        '| Configuration | Stops / censored | Mean event/cap iteration | Exact stop accuracies | Paired difference vs full (pp); n |',
        '|---|---:|---:|---:|---:|']
    for r in overall:
        diff=r['accuracy_difference_vs_full_mean'];value=f'{100*diff:+.3f}; {r["accuracy_difference_vs_full_n"]}' if diff is not None else 'missing; 0'
        lines.append(f'| {r["configuration"]} | {r["stopped"]} / {r["censored"]} | {r["observation_iteration_mean"]:,.1f} | {r["observed_stop_accuracy"]} | {value} |')
    lines+=['','[Individual results](individual.csv), [dataset statistics](dataset_summary.csv), and [overall statistics](overall_summary.csv) retain sample SD, median, range, paired differences vs original A and full reference, window endpoints/counts and counter ages for every configuration. No winning grid cell is selected.','',
        '## Interpretation and limits','',
        'C cannot stop before A because its conjunction preserves original patience; its additional classification evidence may delay or censor stopping. B measures sampled transition counts with multiplicity, including observed changes followed by reversions. D can stop while classifications or dominance continue moving below their earlier record; small cumulative record gains are not evidence of general dominance stability.','',
        'Iterations saved are retrospective processing-budget differences, **not measured time gains**. Archived full-run timings cannot recover candidate prefix wall times or observation overhead. Accuracy comparisons require exact saved instants; no interpolation, future checkpoint substitution or outcome-based threshold adjustments are used.','',
        'The primary hybrid includes legacy-event information causally generated by the old observers. It is not equivalent to monitoring only periodic classifications. Secondary timing differs by decision opportunities and by the window-start convention; report actual spans rather than implying equal windows.','',
        f"All {metadata['check_rows_audited']:,} recorded checks were independently audited; A matched all {metadata['baseline_exact_matches']} archived original events/censors. [Validation metadata](validation.json). Synthetic tests cover boundaries, cumulative small gains, censoring, counter resets, intermediate reversions and prefix invariance. No kernels or local checkpoints are required.",'']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,help='Local ZIP directory; overrides PYPCC_STOPPING_ARTIFACTS')
    parser.add_argument('--download',action='store_true',help='Explicitly fetch missing Release ZIPs with pinned hash verification')
    args=parser.parse_args()
    if args.artifact_dir is not None:
        os.environ['PYPCC_STOPPING_ARTIFACTS']=str(args.artifact_dir)
    if args.download:
        from experiments.stopping_artifact_store import ensure
        directory=Path(os.environ.get('PYPCC_STOPPING_ARTIFACTS','.cache/stopping_artifacts'))
        manifest=json.loads((SOURCE/'artifacts_manifest.json').read_text())
        for entry in manifest['archives']:
            ensure(entry,directory,download=True)
    run()
