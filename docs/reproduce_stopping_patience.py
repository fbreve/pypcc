"""Regenerate published results without propagation, kernels or local checkpoints."""
import argparse
import csv
import gzip
import io
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.patience_archive import PublishedArtifacts, sha
from experiments.patience_results import audit_checks, validate_rows
from experiments.patience_summary import summaries
import report_stopping_patience as report


def generate(source, output):
    source, output = source.resolve(), output.resolve()
    if source == output or output in source.parents or source in output.parents:
        raise ValueError('Output must be separate from published input')
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source/'protocol.json').read_text())
    inventory = json.loads((source/'inventory.json').read_text())
    if sha((source/'protocol.json').read_bytes().replace(b'\r\n', b'\n')) != inventory['protocol_sha256']:
        raise ValueError('Frozen protocol mismatch')
    rows, cases, references, diagnostics, verification = [], [], [], [], []
    archive_rows = report.read_csv(source/'archive_manifest.csv')
    hashes = {(r['dataset'], int(r['seed'])):r for r in archive_rows}
    with PublishedArtifacts(source) as artifacts:
        if len(artifacts.entries) != 800:
            raise ValueError('Expected 800 artifacts')
        for info in inventory['datasets']:
            for seed in protocol['seeds']:
                directory = f"trajectories/{info['name']}/{seed}/"
                result = json.loads(artifacts.read(directory+'complete.json'))
                if (result['dataset'], result['seed'], result['iterations'], result['checks']) != (info['name'], seed, 500000, 50000):
                    raise ValueError('Trajectory identity/budget mismatch')
                if [r['multiplier'] for r in result['rows']] != protocol['multipliers']:
                    raise ValueError('Multiplier mismatch')
                packed = artifacts.read(directory+'checks.csv.gz')
                h = hashes[info['name'], seed]
                raw = gzip.decompress(packed)  # bounded to one trajectory, not all 10M checks
                if sha(packed) != h['gzip_sha256'] or sha(raw) != h['raw_sha256']:
                    raise ValueError('Historical check-log hash mismatch')
                audit = audit_checks(csv.DictReader(io.StringIO(raw.decode())),
                    result['rows'][0]['original_threshold'], protocol['multipliers'], protocol['max_iter'])
                validate_rows(result['rows'], audit, info['nodes']-info['labeled_nodes'], protocol['max_iter'])
                diagnostics.append(dict(dataset=info['name'], seed=seed, **{k:v for k,v in audit.items() if k!='events'}))
                recorded = json.loads(artifacts.read(directory+'verification.json'))
                if recorded != result['verification']:
                    raise ValueError('Recorded verification mismatch')
                verification.extend(dict(dataset=info['name'],seed=seed,**p) for p in recorded)
                cases.append(dict(dataset=info['name'],seed=seed,iterations=result['iterations'],checks=result['checks'],
                    samples=result['samples'],observed_trajectory_s=result['elapsed_s'],checkpoint_io_s=result['checkpoint_io_s'],
                    final_accuracy_unlabeled=result['rows'][0]['final_accuracy_unlabeled'],
                    final_correct_nodes=result['rows'][0]['final_correct_nodes'],rng_sha256=result['rng_sha256']))
                for r in result['rows']:
                    r.update(nodes=info['nodes'],classes=info['classes'],labeled_nodes=info['labeled_nodes'],
                        labeled_fraction=info['labeled_fraction'],connected_components=info['connected_components'],
                        degree_mean=info['degree_mean'],degree_max=info['degree_max'],measured_full_trajectory_s=result['elapsed_s'])
                    rows.append(r)
                samples = list(csv.DictReader(io.StringIO(artifacts.read(directory+'samples.csv').decode())))
                for iteration in protocol['intermediate_iterations']:
                    value = next(r for r in samples if int(r['iteration']) == iteration)
                    references.append(dict(dataset=info['name'],seed=seed,**value))
                print(f"Reconstructed {info['name']} seed {seed}", flush=True)
    datasets, overall = summaries(rows)
    strata = dict(binary=[r for r in rows if r['classes']==2],multiclass=[r for r in rows if r['classes']>2],
        nodes_lt1000=[r for r in rows if r['nodes']<1000],nodes_1000_4999=[r for r in rows if 1000<=r['nodes']<5000],
        nodes_ge5000=[r for r in rows if r['nodes']>=5000],labeled_lt1pct=[r for r in rows if r['labeled_fraction']<.01],
        labeled_ge1pct=[r for r in rows if r['labeled_fraction']>=.01])
    grouped=[]
    for name, values in strata.items():
        if values:
            _, g = summaries(values)
            grouped.extend(dict(stratum=name,**r) for r in g)
    aggregate=[]
    for multiplier in protocol['multipliers']:
        rr=[r for r in rows if r['multiplier']==multiplier]
        r=dict(multiplier=multiplier,trajectories=len(rr),stopped=sum(x['status']=='stopped' for x in rr),
            censored=sum(x['status']=='right_censored' for x in rr),lost_accuracy=sum(x['correct_nodes_difference']<0 for x in rr),
            same_accuracy=sum(x['correct_nodes_difference']==0 for x in rr),higher_accuracy=sum(x['correct_nodes_difference']>0 for x in rr))
        for field in ('observation_iteration','iterations_saved','accuracy_unlabeled','accuracy_difference','correct_nodes_difference'):
            r.update({field+'_'+k:v for k,v in report.describe([x[field] for x in rr]).items()})
        aggregate.append(r)
    tables=dict(individual=rows,dataset_summary=datasets,overall_equal_dataset_summary=overall,
        stratum_equal_dataset_summary=grouped,overall_trajectory_summary=aggregate,trajectory_times=cases,
        intermediate=references,record_diagnostics=diagnostics,full_verification=verification)
    for name, values in tables.items():
        report.write_csv(output/(name+'.csv'),values)
    # Immutable provenance and historical deep-audit evidence: copy, never claim a fresh state/RNG audit.
    for name in ('protocol.json','inventory.json','full_metadata.json','final_invariants.csv','archive_manifest.csv',
                 'phase2_execution.json','phase2_retry_execution.json','phase2_process_samples.csv','report.md',
                 'REPRODUCIBILITY.md'):
        shutil.copyfile(source/name,output/name)
    report.ROOT=output
    report.figures(rows,references,inventory['datasets'])
    paired_plot(datasets, output)
    report.write_report(json.loads((source/'full_metadata.json').read_text()),aggregate,cases,references,diagnostics,rows,inventory['datasets'])
    evidence=dict(trajectories=len(cases),rows=len(rows),checks=sum(r['checks'] for r in cases),
        propagation_executed=False,local_checkpoints_used=False,state_rng_audit='historical evidence only',
        artifacts_manifest_sha256=sha((source/'artifacts_manifest.json').read_bytes()))
    (output/'reproduction_validation.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


def paired_plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for name in dict.fromkeys(r['dataset'] for r in rows):
        values=sorted((r for r in rows if r['dataset']==name),key=lambda r:r['multiplier'])
        axes[0].plot([r['observation_iteration_mean'] for r in values],[r['accuracy_difference_mean'] for r in values],'.-',label=name)
        axes[1].plot([r['multiplier'] for r in values],[r['observation_iteration_mean'] for r in values],'.-')
    axes[0].set(xlabel='Mean stop/censor observation iteration',ylabel='Accuracy difference vs 500000')
    axes[1].set(xlabel='Patience multiplier',ylabel='Mean stop/censor observation iteration')
    axes[0].legend(fontsize=6,ncol=2)
    fig.suptitle('Descriptive paired results: censored cases observed at cap, not estimated stop times')
    fig.tight_layout();fig.savefig(output/'paired_tradeoffs.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/benchmarks/stopping_patience'))
    parser.add_argument('--output',type=Path,default=Path('.cache/stopping_patience_reproduced'))
    parser.add_argument('--deep-audit',action='store_true',help='Optional trusted local checkpoints; writes legacy audit outputs')
    args=parser.parse_args()
    if args.deep_audit:
        report.ROOT=args.source
        report.collect_deep()
    else:
        print(json.dumps(generate(args.source,args.output),indent=2))


if __name__=='__main__':
    main()
