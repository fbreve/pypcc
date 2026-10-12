"""Audit/archive completed phase-2 results and generate descriptive figures.

Run after --full finishes. No propagation or stopping rule is changed here.
"""
import csv
import gzip
import hashlib
import json
import pickle
import statistics
from pathlib import Path

import numpy as np
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.patience_results import audit_checks, validate_rows

ROOT = Path('docs/benchmarks/stopping_patience')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


fixed = SimpleNamespace(write_csv=write_csv)
study = SimpleNamespace(protocol=protocol)


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def archive_checks(path):
    destination=path.with_suffix('.csv.gz')
    source_hash=digest_file(path)
    # Deterministic gzip: no timestamp or source filename in the header.
    with destination.open('wb') as output:
        with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as compressed:
            with path.open('rb') as source:
                while chunk:=source.read(1024*1024):compressed.write(chunk)
    with gzip.open(destination,'rb') as archived:
        h=hashlib.sha256()
        while chunk:=archived.read(1024*1024):h.update(chunk)
    if h.hexdigest()!=source_hash:raise AssertionError('Archive round-trip mismatch')
    return dict(raw_sha256=source_hash,gzip_sha256=digest_file(destination),
                raw_bytes=path.stat().st_size,gzip_bytes=destination.stat().st_size)


def final_state(directory,result):
    from experiments.stopping_patience import STATE_INDICES
    with (directory/'recovery.pkl').open('rb') as stream:state=pickle.load(stream)
    a=state['args'];dom=a[11];n,c=dom.shape
    if state['iteration']!=500000 or state['runtime']!=result['runtime']:
        raise AssertionError('Final checkpoint budget/runtime mismatch')
    expected,_=state['verification_snapshots'][500000]
    assert all(a[i].dtype==x.dtype and a[i].shape==x.shape and a[i].tobytes()==x.tobytes()
               for i,x in zip(STATE_INDICES,expected))
    assert hashlib.sha256(repr(state['rng']).encode()).hexdigest()==result['rng_sha256']
    known=np.flatnonzero(a[2]!=-1)
    assert np.array_equal(dom[known],np.eye(c)[a[2][known]])
    assert a[10].dtype==np.uint8 and np.all(a[10][known,np.arange(len(known))]==0)
    assert np.all((a[7]>=0)&(a[7]<n)) and np.all((a[8]>=0)&(a[8]<c))
    assert np.isfinite(dom).all() and np.isfinite(a[9]).all() and np.isfinite(a[12]).all()
    assert a[9].min()>=-1e-9 and a[9].max()<=1+1e-9
    error=float(np.max(np.abs(dom.sum(axis=1)-1)))
    assert error<=1e-9 and dom.min()>=-1e-9 and dom.max()<=1+1e-9
    return dict(dominance_min=float(dom.min()),dominance_max=float(dom.max()),
                dominance_sum_max_abs_error=error, labeled_preserved=True,
                positions_valid=True,distances_uint8=True, home_distances_zero=True,
                strength_min=float(a[9].min()),strength_max=float(a[9].max()),
                own_degree_finite=True, final_snapshot_bytes_equal=True,final_rng_hash_equal=True)


def collect_deep():
    import study_stopping_patience as study
    study.ROOT = ROOT
    p=study.protocol();inventory=json.loads((ROOT/'inventory.json').read_text())['datasets']
    if study.protocol_hash()!=json.loads((ROOT/'inventory.json').read_text())['protocol_sha256']:
        raise AssertionError('Frozen protocol changed')
    all_rows=[];cases=[];references=[];verification=[];diagnostics=[];archives=[];invariants=[]
    for info in inventory:
        for seed in p['seeds']:
            directory=ROOT/'trajectories'/info['name']/str(seed)
            result=json.loads((directory/'complete.json').read_text())
            assert result['iterations']==p['max_iter'] and result['checks']==50000
            assert [r['multiplier'] for r in result['rows']]==p['multipliers']
            assert result['dataset']==info['name'] and result['seed']==seed
            with (directory/'checks.csv').open(newline='') as stream:
                audit=audit_checks(csv.DictReader(stream),result['rows'][0]['original_threshold'],p['multipliers'],p['max_iter'])
            validate_rows(result['rows'],audit,info['nodes']-info['labeled_nodes'],p['max_iter'])
            audits={k:v for k,v in audit.items() if k!='events'}
            diagnostics.append(dict(dataset=info['name'],seed=seed,**audits))
            invariants.append(dict(dataset=info['name'],seed=seed,**final_state(directory,result)))
            for point in result['verification']:
                assert point['state_equal'] and point['rng_equal']
                assert point['original_stop_matches'] in (None,True)
                verification.append(dict(dataset=info['name'],seed=seed,**point))
            cases.append(dict(dataset=info['name'],seed=seed,iterations=result['iterations'],
                checks=result['checks'],samples=result['samples'],observed_trajectory_s=result['elapsed_s'],
                checkpoint_io_s=result['checkpoint_io_s'], final_accuracy_unlabeled=result['rows'][0]['final_accuracy_unlabeled'],
                final_correct_nodes=result['rows'][0]['final_correct_nodes'],rng_sha256=result['rng_sha256']))
            for row in result['rows']:
                row.update(nodes=info['nodes'],classes=info['classes'],labeled_nodes=info['labeled_nodes'],
                    labeled_fraction=info['labeled_fraction'],connected_components=info['connected_components'],
                    degree_mean=info['degree_mean'],degree_max=info['degree_max'],measured_full_trajectory_s=result['elapsed_s'])
                all_rows.append(row)
            samples=read_csv(directory/'samples.csv')
            for iteration in p['intermediate_iterations']:
                value=next(r for r in samples if int(r['iteration'])==iteration)
                references.append(dict(dataset=info['name'],seed=seed,**value))
            archives.append(dict(dataset=info['name'],seed=seed,**archive_checks(directory/'checks.csv')))
            print(f"Audited and archived {info['name']} seed {seed}",flush=True)
    assert len(cases)==200 and len(all_rows)==1400 and len(references)==800
    # Existing --full output must match the independently reconstructed case table.
    existing=read_csv(ROOT/'individual.csv')
    assert len(existing)==1400
    for actual,expected in zip(existing,all_rows):
        for key,value in expected.items():
            if isinstance(value,(float,int)):
                assert float(actual[key])==value,(key,actual[key],value)
            else:assert actual[key]==('' if value is None else str(value))
    for name,rows in [('trajectory_times',cases),('intermediate',references),('full_verification',verification),
                      ('record_diagnostics',diagnostics),('archive_manifest',archives),('final_invariants',invariants)]:
        fixed.write_csv(ROOT/(name+'.csv'),rows)
    aggregate=[]
    for multiplier in p['multipliers']:
        rows=[r for r in all_rows if r['multiplier']==multiplier]
        # Exactly ten seeds/dataset: the 200 trajectory rows already weight datasets equally.
        row=dict(multiplier=multiplier,trajectories=len(rows),stopped=sum(r['status']=='stopped' for r in rows),
                 censored=sum(r['status']=='right_censored' for r in rows),
                 lost_accuracy=sum(r['correct_nodes_difference']<0 for r in rows),
                 same_accuracy=sum(r['correct_nodes_difference']==0 for r in rows),
                 higher_accuracy=sum(r['correct_nodes_difference']>0 for r in rows))
        for field in ('observation_iteration','iterations_saved','accuracy_unlabeled','accuracy_difference','correct_nodes_difference'):
            values=[r[field] for r in rows]
            row.update({field+'_'+stat:value for stat,value in describe(values).items()})
        aggregate.append(row)
    fixed.write_csv(ROOT/'overall_trajectory_summary.csv',aggregate)
    metadata=dict(approval_url='https://github.com/fbreve/pypcc/pull/8#issuecomment-6115498944',
        environment=study.environment(),protocol_sha256=study.protocol_hash(),full_trajectories=200,
        executed_iterations=sum(r['iterations'] for r in cases), original_checks=sum(r['checks'] for r in cases),
        measured_observed_trajectory_s=sum(r['observed_trajectory_s'] for r in cases),
        checkpoint_io_s=sum(r['checkpoint_io_s'] for r in cases),monolithic_checkpoint_comparisons=len(verification),
        original_first_stop_comparisons=sum(r['original_stop_matches'] is True for r in verification),
        raw_check_csv_bytes=sum(r['raw_bytes'] for r in archives), gzip_check_csv_bytes=sum(r['gzip_bytes'] for r in archives),
        maximum_dominance_sum_error=max(r['dominance_sum_max_abs_error'] for r in invariants),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'full_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    figures(all_rows,references,inventory)
    write_report(metadata,aggregate,cases,references,diagnostics,all_rows,inventory)
    return metadata,aggregate


def describe(values):
    return dict(mean=statistics.mean(values),median=statistics.median(values),
                sd=statistics.stdev(values),minimum=min(values),maximum=max(values))


def write_report(metadata,aggregate,cases,references,diagnostics,rows,inventory):
    original=[r for r in rows if r['multiplier']==1.]
    final_mean=statistics.mean(r['final_accuracy_unlabeled'] for r in cases)
    retry=json.loads((ROOT/'phase2_retry_execution.json').read_text())
    previous=json.loads((ROOT/'phase2_execution.json').read_text(encoding='utf-8-sig'))
    samples=read_csv(ROOT/'phase2_process_samples.csv')
    peak=max(int(r['peak_working_set_bytes']) for r in samples)
    lines=['# Full PCC stopping-patience study: paired retrospective results','',
      '**Phase 2 completed: 200 continuous logical trajectories, 20 frozen datasets, seeds 10–19, each exactly 500,000 iterations. PR remains draft.**','',
      '[Explicit authorization](https://github.com/fbreve/pypcc/pull/8#issuecomment-6115498944). The [phase-1 feasibility report](report.md), inventory, protocol and pilot results are preserved. No production kernel, public API, graph construction, labeling, stopping default or multiplier was changed. Seeds were examined previously and are not independent validation. No dataset was replaced or subsampled, and no best multiplier was selected per dataset.','',
      '## Overall paired outcomes','',
      f"The full-budget reference accuracy, averaged with equal dataset weights, is **{100*final_mean:.3f}%**. The table below includes all 200 cases per multiplier. Censored cases contribute their cap observation, not an invented stopping time; their accuracy is the full-budget reference. Accuracy difference is event/cap minus full 500k, in percentage points (pp). Iteration savings are retrospective and **not measured wall-clock speedups**.",'',
      '| Multiplier | Stopped / censored | Mean observation iteration | Median | Iterations saved (%) | Mean accuracy (%) | Paired accuracy difference mean ± SD (pp) | Difference range (pp) | Worse / same / better cases |',
      '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['multiplier']:g} | {r['stopped']} / {r['censored']} | {r['observation_iteration_mean']:,.0f} | {r['observation_iteration_median']:,.0f} | {100*r['iterations_saved_mean']/500000:.2f} | {100*r['accuracy_unlabeled_mean']:.3f} | {100*r['accuracy_difference_mean']:+.3f} ± {100*r['accuracy_difference_sd']:.3f} | {100*r['accuracy_difference_minimum']:+.2f} .. {100*r['accuracy_difference_maximum']:+.2f} | {r['lost_accuracy']} / {r['same_accuracy']} / {r['higher_accuracy']} |")
    lines+=['','Each dataset has ten trajectories, so the 200 equally weighted rows also give each dataset equal weight. This table describes the trajectory distribution (sample SD); [overall_equal_dataset_summary.csv](overall_equal_dataset_summary.csv) separately describes the distribution of the 20 dataset means. Neither SD is an uncertainty interval from independent validation. Mean/median/SD/min/max and stop/censor rates are retained in CSV.','',
      '![Paired distributions](full_distributions.png)','',
      '## Dataset outcomes at original patience (1.0)','',
      'Unknown-node micro accuracy is evaluated only retrospectively. Correct-node difference is reported separately from accuracy difference; its scale depends on dataset size. Positive differences mean the hypothetical stopped prediction was better than the later 500k prediction, not that later propagation improved.','',
      '| Dataset | Original stopped / censored | Mean stop/cap iteration | Full 500k accuracy mean ± SD (%) | Original event/cap accuracy (%) | Paired difference mean ± SD (pp) | Correct-node difference mean [min, max] | Observed trajectory time mean ± SD (s) |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for info in inventory:
        name=info['name'];rr=[r for r in original if r['dataset']==name];tt=[r for r in cases if r['dataset']==name]
        full=[r['final_accuracy_unlabeled'] for r in tt];diff=[r['accuracy_difference'] for r in rr];counts=[r['correct_nodes_difference'] for r in rr];times=[r['observed_trajectory_s'] for r in tt]
        stopped=sum(r['status']=='stopped' for r in rr)
        lines.append(f"| {name} | {stopped} / {10-stopped} | {statistics.mean(r['observation_iteration'] for r in rr):,.0f} | {100*statistics.mean(full):.2f} ± {100*statistics.stdev(full):.2f} | {100*statistics.mean(r['accuracy_unlabeled'] for r in rr):.2f} | {100*statistics.mean(diff):+.3f} ± {100*statistics.stdev(diff):.3f} | {statistics.mean(counts):+.1f} [{min(counts):+d}, {max(counts):+d}] | {statistics.mean(times):.3f} ± {statistics.stdev(times):.3f} |")
    lines+=['','All seven multipliers per dataset/seed are in [individual.csv](individual.csv), [dataset_summary.csv](dataset_summary.csv) and the heatmap below. Do not interpret cap times as uncensored stop-time estimates. The mathematically unavoidable Spambase 4.0 censoring is retained in all ten seeds.','',
      '![All frozen datasets and multipliers](full_heatmaps.png)','',
      '![Accuracy and iteration tradeoffs](paired_tradeoffs.png)','',
      '## Records, long patience and subsequent changes','',
      'Checks exactly reproduce the original serial mean-maximum-dominance reduction and phase 1, 11, 21, …; strict improvements reset the counter, and eligibility requires counter > threshold. Larger multipliers delay the first eligible event or censor it. All multipliers share one continuous trajectory; their hypothetical stops never affect propagation. The full reference is always 500k, even after every predicate fires.','',
      '| Dataset | Required original stalled checks | Mean total record resets | Mean records after 30k | Mean last record iteration | Mean longest stalled run (checks) | Mean sampled nodes changing after original stop | Mean sampled nodes returning to stop label |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for info in inventory:
        name=info['name'];dd=[r for r in diagnostics if r['dataset']==name];rr=[r for r in original if r['dataset']==name]
        # Post-stop columns describe observed stops only; censored cases have no post-stop interval.
        observed=[r for r in rr if r['status']=='stopped']
        changed=f"{statistics.mean(r['sampled_ever_changed_nodes'] for r in observed):.1f}" if observed else 'n/a'
        returned=f"{statistics.mean(r['sampled_returned_nodes'] for r in observed):.1f}" if observed else 'n/a'
        lines.append(f"| {name} | {rr[0]['original_threshold']+1} | {statistics.mean(r['resets'] for r in dd):.1f} | {statistics.mean(r['late_records'] for r in dd):.1f} | {statistics.mean(r['last_reset_iteration'] for r in dd):,.0f} | {statistics.mean(r['longest_stall'] for r in dd):,.0f} | {changed} | {returned} |")
    stopped=[r for r in original if r['status']=='stopped']
    changed=sum(r['sampled_ever_changed_nodes']>0 for r in stopped)
    returned=sum(r['sampled_returned_nodes']>0 for r in stopped)
    lines+=['',f"Among the {len(stopped)} observed original stops, **{changed} cases have subsequently changing nodes at sampled observations**, and **{returned} cases have sampled returns to the stop-label prediction**. Endpoint classification differences and full-matrix dominance norms are retained separately in individual.csv. A return demonstrates instability at observed times; zero endpoint changes or zero sampled changes do not establish interval-wide stability.",'',
      'The record log is exact at every original check; classification sampling is every 1000 iterations plus exact events and reference points. Thus small record improvements can be described without claiming that no classification changed between checks. [record_diagnostics.csv](record_diagnostics.csv) separates record frequency/magnitude from available uninterrupted stalled-check runs. Its <=1e-12 gain category is descriptive only and is never a stopping tolerance. Late records, long required runs and graph/label structure are distinct explanations, not interchangeable labels for slow stopping.','',
      '![Intermediate retrospective accuracies](intermediate_accuracy.png)','',
      '## Structure and descriptive strata','',
      'The fixed formula is approximately T = 20 / labeled_fraction at k=10 and es_chk=2000, before integer rounding. Therefore associations between patience and labeled fraction partly follow algebraically from the protocol rather than from learned convergence. Size, class count, dimensionality, imbalance and connected components vary together; this study does not isolate causal effects. [stratum_equal_dataset_summary.csv](stratum_equal_dataset_summary.csv) reports binary/multiclass, fixed size groups and labeled-fraction groups with equal dataset weights. Small groups and heterogeneous datasets limit generalization.','',
      '## Actual cost and execution incidents','',
      f"Summed recorded trajectory wall time including dominance observation, prediction checkpoints, CSV and recovery writes: **{metadata['measured_observed_trajectory_s']:.2f} s ({metadata['measured_observed_trajectory_s']/60:.2f} min)**. Checkpoint-write time accounts for {metadata['checkpoint_io_s']:.2f} s. These are measured full trajectories, not iteration-based speedup estimates. Per-case timings are in [trajectory_times.csv](trajectory_times.csv); SD reflects seeds/execution conditions, not replicated timing of each seed.",'',
      f"Two commands were interrupted by transient Windows WinError 5 during atomic checkpoint replacement, after 14 and 85 completed cases. The unmodified resumable workflow retained completed cases and exact saved states. An optional I/O-only launcher then added bounded retries (40 attempts, 50ms delay) around checkpoint writes. The final invocation reused 85 cases and finished in **{retry['wall_s']:.2f} s**, including setup, verification and plot generation; {len(retry['retries'])} retry episodes were resolved. The preceding resumed command took {previous['resumed_command_wall_s']:.2f} s before interruption. Initial-attempt total command wall time was not instrumented, so these command times are not a complete end-to-end total. Downtime/lost work after the last checkpoint and the last saved checkpoint I/O timing can be omitted by recovery; no scientific trajectory is reconstructed from independent restarted prefixes.",'',
      f"Largest sampled process high-water mark: {peak/2**20:.1f} MiB ([phase2_process_samples.csv](phase2_process_samples.csv)); sampling does not guarantee capture of the final post-plot process peak. Check logs occupy {metadata['raw_check_csv_bytes']/2**20:.1f} MiB uncompressed and **{metadata['gzip_check_csv_bytes']/2**20:.1f} MiB** as deterministic lossless CSV gzip archives. [archive_manifest.csv](archive_manifest.csv) records raw/compressed sizes and SHA-256 checksums. Raw CSVs and trusted local recovery pickles remain available locally and are ignored by git; compressed CSVs, sample CSVs, complete JSON and verification JSON are preserved inside four ZIPs. Phase-1 feasibility artifacts were not regenerated.",'',
      '## Correctness and reproducibility','',
      f"All **{metadata['executed_iterations']:,} executed iterations** and **{metadata['original_checks']:,} checks** are accounted for. Independent streaming replay of every recorded check verified strict records, resets, check phase, all seven first events and censoring. Accuracy/count identities and iteration savings were checked in all 1400 result rows. There are **{metadata['monolithic_checkpoint_comparisons']} successful unchanged-monolithic state/RNG comparisons**, including 30k and 500k for every case and **{metadata['original_first_stop_comparisons']} original first-stop comparisons**. See [full_verification.csv](full_verification.csv). Exact state comparisons use no floating tolerance; this is within the same sequential Numba implementation, not an equality demand across backends.",'',
      f"All 200 final checkpoints passed finite/bounded dominance and strength checks, dominance conservation, preservation of labeled nodes, valid positions, uint8 distances/home distances, finite own-degree and saved final-state/RNG identity checks. Maximum dominance-row sum error was {metadata['maximum_dominance_sum_error']:.3g} ([final_invariants.csv](final_invariants.csv)); a 1e-9 invariant-validation tolerance does not alter stopping or propagation.",'',
      'Environment, observer source versions and approved-protocol hashes are in [full_metadata.json](full_metadata.json). Sources/versions and data/graph/all labeling fingerprints remain in inventory.json. Recovery uses the version-locked private Numba MT API; wrapper retries never draw numbers or mutate state. Deterministic fault injection verifies retry state/RNG/CSV equivalence and bounded failures. The original propagation kernels and public API remain unchanged.','',
      'Regenerate statistics and figures from the public ZIPs without checkpoints or propagation:','',
      '```powershell','conda run -n pypcc-dev python docs/reproduce_stopping_patience.py',
      '```','',
      'Four deterministic ZIPs preserve the 800 original Git blobs and paths. artifacts_manifest.json records each member and ZIP SHA-256; archive_manifest.csv preserves the original gzip/raw log hashes. Public reproduction independently replays recorded checks and reconstructs statistics, figures and this report; it neither loads recovery.pkl nor imports production kernels. The state/RNG comparisons and final invariants above are historical audit evidence, not newly performed state comparisons. Optional --deep-audit uses trusted local checkpoints separately. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands, packaging and limits. No long experiment is required by CI.','',
      '## Observations, hypotheses and limitations','',
      '**Observations:** stopping/censor rates, paired accuracy losses/gains, subsequent sampled reversions and dominance endpoint differences are reported separately. Patience changes first-event timing monotonically on the shared trajectory; accuracy is not guaranteed to improve monotonically with more iterations. Forced Spambase 4.0 censoring is a budget/formula consequence.','',
      '**Hypotheses to discuss:** datasets with frequent late records may repeatedly restart patience even when sampled classifications change little; large thresholds can delay eligibility even after the last record. Neither dominance records nor larger patience by themselves establish meaningful accuracy progress. Descriptive group associations cannot establish graph/label causes.','',
      '**Limitations:** reused seeds, 20 heterogeneous related tasks (Digits/PenDigits share a handwriting domain), only three labels/class, one fixed parameter/graph protocol, cap censoring and sparse classification sampling. Full-matrix norms describe event-to-final endpoint differences, not total motion. Truth is never used for stopping; the 500k result is a reference, not an optimum or ground-truth convergence guarantee. Operational timing includes I/O and recovery incidents but excludes verification from trajectory timing.','',
      '**No new default or tuned criterion is proposed.** Review the fixed-grid sensitivity and dataset-specific variation before deciding on another experiment. PR #8 remains draft and master is unchanged.','']
    (ROOT/'full_report.md').write_text('\n'.join(lines),encoding='utf-8')


def figures(rows,references,inventory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[i['name'] for i in inventory];multipliers=study.protocol()['multipliers']
    fig,axes=plt.subplots(1,3,figsize=(15,9))
    fields=['stop_rate','accuracy_difference','iterations_saved']
    titles=['Stopping rate (%)','Accuracy difference vs 500k (percentage points)','Iterations saved (% of 500k)']
    for ax,field,title in zip(axes,fields,titles):
        data=[]
        for name in names:
            line=[]
            for mult in multipliers:
                values=[r for r in rows if r['dataset']==name and r['multiplier']==mult]
                line.append(100*statistics.mean(r['status']=='stopped' for r in values) if field=='stop_rate' else
                    statistics.mean(r[field] for r in values)*(100 if field=='accuracy_difference' else 100/500000))
            data.append(line)
        kwargs=dict(cmap='viridis',vmin=0,vmax=100)
        if field=='accuracy_difference':
            limit=max(np.abs(data).max(),.01);kwargs=dict(cmap='RdBu_r',vmin=-limit,vmax=limit)
        picture=ax.imshow(data,aspect='auto',**kwargs)
        ax.set_xticks(range(7),[str(m) for m in multipliers]);ax.set_xlabel('Frozen multiplier');ax.set_title(title)
        ax.set_yticks(range(20),names if ax is axes[0] else [])
        fig.colorbar(picture,ax=ax,shrink=.7)
    fig.suptitle('Ten paired reused seeds/dataset; censoring retained at cap; no multiplier selection')
    fig.tight_layout();fig.savefig(ROOT/'full_heatmaps.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,5,figsize=(17,11))
    for ax,name in zip(axes.flat,names):
        for seed in range(10,20):
            values=[r for r in references if r['dataset']==name and r['seed']==seed]
            ax.plot([int(r['iteration']) for r in values],[100*float(r['accuracy_unlabeled']) for r in values],'.-',alpha=.4,lw=.8)
        ax.set_title(name);ax.grid(alpha=.2)
        ax.set_xlim(0,510000);ax.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
    for ax in axes[-1]:ax.set_xlabel('Continuous trajectory iteration')
    for ax in axes[:,0]:ax.set_ylabel('Unlabeled accuracy (%)')
    fig.suptitle('Retrospective intermediate accuracy: 30k / 100k / 200k / 500k, ten reused seeds')
    fig.tight_layout();fig.savefig(ROOT/'intermediate_accuracy.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].boxplot([[r['observation_iteration'] for r in rows if r['multiplier']==m] for m in multipliers],tick_labels=multipliers)
    axes[0].set_yscale('log');axes[0].set_ylabel('Stop/censor observation iteration (log)')
    axes[1].boxplot([[100*r['accuracy_difference'] for r in rows if r['multiplier']==m] for m in multipliers],tick_labels=multipliers)
    axes[1].axhline(0,color='black',lw=.8);axes[1].set_ylabel('Event/cap accuracy minus full 500k (pp)')
    for ax in axes:ax.set_xlabel('Frozen multiplier');ax.grid(alpha=.2)
    fig.suptitle('200 paired cases per multiplier; censored times are cap observations')
    fig.tight_layout();fig.savefig(ROOT/'full_distributions.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    from reproduce_stopping_patience import main
    main()
