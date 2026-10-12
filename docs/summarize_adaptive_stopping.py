"""Paired schedule diagnostics and figures from already published comparison CSVs."""
import csv
import json
import statistics
from pathlib import Path

ROOT=Path('docs/benchmarks/adaptive_stopping')


def main():
    rows=list(csv.DictReader((ROOT/'individual.csv').open()))
    index={(r['dataset'],r['seed'],r['candidate'],r['window'],r['budget'],r['schedule']):r for r in rows}
    pairs=[]
    for r in rows:
        if r['schedule']!='hybrid_regular':continue
        s=index[r['dataset'],r['seed'],r['candidate'],r['window'],r['budget'],'all_observations_secondary']
        primary,secondary=int(r['observation_iteration']),int(s['observation_iteration'])
        assert secondary<=primary
        pairs.append(dict(dataset=r['dataset'],seed=r['seed'],candidate=r['candidate'],window=r['window'],budget=r['budget'],
            primary_observation_iteration=primary,secondary_observation_iteration=secondary,
            iteration_difference=primary-secondary,primary_status=r['status'],secondary_status=s['status'],
            primary_event_cap_accuracy=r['event_cap_accuracy'],secondary_event_cap_accuracy=s['event_cap_accuracy'],
            accuracy_difference=float(r['event_cap_accuracy'])-float(s['event_cap_accuracy']),
            primary_window_span=r['actual_window_span'],secondary_window_span=s['actual_window_span']))
    with (ROOT/'primary_secondary_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    overview=list(csv.DictReader((ROOT/'overall_summary.csv').open()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in overview if r['is_primary']=='True']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    names=[r['candidate'] for r in selected]
    axes[0].bar(names,[float(r['observation_iteration_mean']) for r in selected])
    axes[0].set_ylabel('Mean stop/cap iteration (all 200 cases)')
    axes[1].bar(names,[int(r['stopped'])/200 for r in selected]);axes[1].set_ylim(0,1)
    axes[1].set_ylabel('Observed stop fraction')
    axes[2].bar(names,[int(r['observed_stop_accuracy'])/max(int(r['stopped']),1) for r in selected]);axes[2].set_ylim(0,1)
    axes[2].set_ylabel('Exact accuracy availability among stops')
    fig.suptitle('Exploratory primary policies; D = record stagnation; no measured time speedup')
    fig.tight_layout();fig.savefig(ROOT/'primary_comparison.png',dpi=150);plt.close(fig)
    blocks=['## Paired primary-versus-secondary schedules','',
        '| Candidate / W / budget | Cases with later periodic decision | Mean iteration difference | Mean paired accuracy difference (pp) |',
        '|---|---:|---:|---:|']
    for candidate in ('B','C'):
        for W in (1000,3000,5000):
            for budget in (0,.001):
                rr=[r for r in pairs if r['candidate']==candidate and int(r['window'])==W and float(r['budget'])==budget]
                blocks.append(f'| {candidate} / {W} / {budget:g} | {sum(r["iteration_difference"]>0 for r in rr)} / 200 | {statistics.mean(r["iteration_difference"] for r in rr):.1f} | {100*statistics.mean(r["accuracy_difference"] for r in rr):+.3f} |')
    blocks += ['','Additional decision opportunities cannot postpone the secondary first stop because it includes every primary decision time. Window spans can differ; [all 2400 paired rows](primary_secondary_pairs.csv) retain those spans. No schedule or parameter is selected as a winner.','',
        '![Primary comparisons and missing stop accuracy](primary_comparison.png)','',
        'Reproduce from repository root using Windows pypcc-dev:','',
        '```powershell','conda run -n pypcc-dev python docs/compare_adaptive_stopping.py',
        'conda run -n pypcc-dev python docs/summarize_adaptive_stopping.py','```','',
        'The first command replays archives; it does not run propagation. The second reads result CSVs only. Both preserve the frozen protocol and source archives.','']
    report=(ROOT/'report.md').read_text(encoding='utf-8').split('\n## Paired primary-versus-secondary schedules')[0]
    notes=(ROOT/'review_notes.md').read_text(encoding='utf-8')
    (ROOT/'report.md').write_text(report+'\n'+'\n'.join(blocks)+'\n'+notes,encoding='utf-8')


if __name__=='__main__':main()
