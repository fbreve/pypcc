"""Retrospective candidate analysis; new Wine reference runs, no production changes."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import benchmark_backends as fixed
from benchmark_backend_convergence import prepare_cases
from diagnose_digits_stopping import initial_arguments, predictions, trace
from tests.test_sequential_safety import next_random


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def wine_trajectories(output):
    baseline, cases = prepare_cases(Path('docs/benchmarks/backend_comparison.json'))
    old = read_csv(Path('docs/benchmarks/convergence_30000/backend_convergence.csv'))
    checks, snapshots, summaries = [], [], []
    for case in (case for case in cases if case['dataset'] == 'Wine'):
        initial = initial_arguments(case, baseline['protocol'], 30000)
        warm = copy.deepcopy(initial)
        fixed.pcc_numba.pcc_propagate_numba(*warm[:22], 2, True, *warm[24:])
        fixed.pcc_numba._pcc_calc_mmpot_sequential(initial[11], initial[0].shape[0], initial[5])
        next_random()
        signatures = tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures)
        # Continue original particle dynamics even after the shadow stopping test fires.
        final, rng, ch, ss, summary = trace(initial, case, 30000, 500, stop_at_original=False)
        assert signatures == tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures)
        mono = copy.deepcopy(initial)
        mono[23] = False  # Existing API: uninterrupted reference, not a new criterion.
        fixed.seed_numba(case['seed'])
        fixed.pcc_numba.pcc_propagate_numba(*mono)
        for i in (7, 9, 10, 11, 12):
            np.testing.assert_array_equal(final[i], mono[i])
        assert rng == float(next_random())
        # Independently verify the original-stop prefix, including RNG.
        prefix, prefix_rng, _, _, prefix_info = trace(initial, case, 30000, 500)
        original = copy.deepcopy(initial)
        fixed.seed_numba(case['seed'])
        info = fixed.pcc_numba.pcc_propagate_numba(*original, return_info=True)
        for i in (7, 9, 10, 11, 12):
            np.testing.assert_array_equal(prefix[i], original[i])
        assert prefix_rng == float(next_random())
        assert info['effective_iterations'] == summary['first_original_stop']
        expected = next(r for r in old if r['dataset'] == 'Wine' and
                        r['configuration'] == 'numba_sequential' and int(r['seed']) == case['seed'])
        assert info['effective_iterations'] == int(expected['effective_iterations'])
        assert fixed.fingerprint(predictions(prefix)) == expected['prediction_sha256']
        summary.update(full_trajectory_state_rng_verified=True, original_prefix_state_rng_verified=True,
                       original_accuracy=float(expected['accuracy_unlabeled']),
                       data_sha256=baseline['datasets']['Wine']['data_sha256'],
                       graph_sha256=fixed.fingerprint(case['edges'], case['degrees']),
                       labels_sha256=fixed.fingerprint(case['labels']))
        checks.extend(ch)
        snapshots.extend(ss)
        summaries.append(summary)
    for name, rows in (('wine_checks', checks), ('wine_snapshots', snapshots), ('wine_summary', summaries)):
        fixed.write_csv(output/(name+'.csv'), rows)
    return checks, snapshots


# Fixed exploratory grid. Candidates never inspect true labels or accuracy.
CANDIDATES = [
    ('labels_exact', 0., None),
    ('labels_0.1pct', .001, None),
    ('labels_0.5pct', .005, None),
    ('metric_1e-3', None, .001),
    ('metric_5e-4', None, .0005),
    ('hybrid_0.1pct_1e-3', .001, .001),
    ('hybrid_exact_1e-3', 0., .001),
    ('original_or_metric_5e-4', None, .0005),
    ('original_or_hybrid_0.1pct_1e-3', .001, .001),
]


def evaluate(dataset, checks, snapshots, original_rows):
    result, windows = [], []
    for seed in range(5):
        ch = sorted((r for r in checks if int(r['seed']) == seed), key=lambda r:int(r['iteration']))
        ss = sorted((r for r in snapshots if int(r['seed']) == seed), key=lambda r:int(r['iteration']))
        baseline_accuracy = float(ss[-1]['accuracy_unlabeled'])
        old = next(r for r in original_rows if r['dataset'] == dataset and
                   r['configuration'] == 'numba_sequential' and int(r['seed']) == seed)
        original_iter = int(old['effective_iterations'])
        seed_windows = []
        for previous, current in zip(ss, ss[1:]):
            start, end = int(previous['iteration']), int(current['iteration'])
            if end-start != 500:
                continue  # Do not treat the shorter final interval as a full window.
            # Inclusive endpoints and ALL original scalar checks in this window.
            metrics = [float(r['metric']) for r in ch if start <= int(r['iteration']) <= end]
            assert len(metrics) == 51
            seed_windows.append(dict(dataset=dataset, seed=seed, start_iteration=start,
                                     iteration=end, changed_predictions=int(current['changed_predictions']),
                                     changed_fraction_unlabeled=float(current['changed_fraction_unlabeled']),
                                     metric_range=max(metrics)-min(metrics),
                                     accuracy_unlabeled=float(current['accuracy_unlabeled'])))
        windows.extend(seed_windows)
        for name, label_tolerance, metric_tolerance in CANDIDATES:
            streak, selected = 0, None
            for window in seed_windows:
                qualifies = ((label_tolerance is None or window['changed_fraction_unlabeled'] <= label_tolerance)
                             and (metric_tolerance is None or window['metric_range'] <= metric_tolerance))
                streak = streak+1 if qualifies else 0
                if window['iteration'] >= 5001 and streak >= 3:
                    selected = window
                    break
            stop = selected['iteration'] if selected else 30000
            accuracy = selected['accuracy_unlabeled'] if selected else baseline_accuracy
            source = 'candidate' if selected else 'cap'
            triggered = selected is not None
            if name.startswith('original_or_') and original_iter < 30000 and original_iter <= stop:
                stop, accuracy = original_iter, float(old['accuracy_unlabeled'])
                source, triggered = 'original', True
            result.append(dict(dataset=dataset, seed=seed, candidate=name,
                               label_tolerance=label_tolerance, metric_tolerance=metric_tolerance,
                               minimum_iteration=5001, window_iterations=500, patience_windows=3,
                               triggered=triggered, trigger_source=source, estimated_stop_iteration=stop,
                               saved_iterations=30000-stop, saved_fraction=(30000-stop)/30000,
                               accuracy_at_stop=accuracy, accuracy_full_30000=baseline_accuracy,
                               accuracy_difference_pp=100*(accuracy-baseline_accuracy),
                               original_stop_iteration=original_iter,
                               iterations_difference_vs_original=stop-original_iter,
                               accuracy_original=float(old['accuracy_unlabeled']),
                               accuracy_difference_vs_original_pp=100*(accuracy-float(old['accuracy_unlabeled'])),
                               analysis='retrospective_existing_digits' if dataset=='Digits' else 'retrospective_new_wine_reference'))
    return result, windows


def main(args):
    if platform.system() != 'Windows' or not fixed.pcc_step.__file__.endswith('.pyd'):
        raise RuntimeError('requires Windows pypcc-dev with native Cython available')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_wine:
        wc, ws = (read_csv(args.output_dir/(name+'.csv')) for name in ('wine_checks','wine_snapshots'))
    else:
        wc, ws = wine_trajectories(args.output_dir)
    digits = Path('docs/benchmarks/digits_stopping_diagnostic')
    dc, ds = (read_csv(digits/(name+'.csv')) for name in ('checks','snapshots'))
    original_rows = read_csv(Path('docs/benchmarks/convergence_30000/backend_convergence.csv'))
    all_rows, all_windows = [], []
    for dataset, checks, snapshots in (('Digits',dc,ds), ('Wine',wc,ws)):
        rows, windows = evaluate(dataset, checks, snapshots, original_rows)
        all_rows.extend(rows)
        all_windows.extend(windows)
    fixed.write_csv(args.output_dir/'candidate_runs.csv', all_rows)
    fixed.write_csv(args.output_dir/'windows.csv', all_windows)
    summaries = []
    for dataset in ('Digits','Wine'):
        for name, _, _ in CANDIDATES:
            rows = [r for r in all_rows if r['dataset']==dataset and r['candidate']==name]
            item = dict(dataset=dataset,candidate=name,triggered_runs=sum(r['triggered'] for r in rows))
            item['original_fallback_runs'] = sum(r['trigger_source']=='original' for r in rows)
            for key in ('estimated_stop_iteration','saved_fraction','accuracy_difference_pp',
                        'accuracy_difference_vs_original_pp','iterations_difference_vs_original'):
                values = [r[key] for r in rows]
                item[key+'_mean'], item[key+'_std'] = float(np.mean(values)),float(np.std(values,ddof=1))
            item['worst_accuracy_difference_pp'] = min(r['accuracy_difference_pp'] for r in rows)
            item['worst_accuracy_difference_vs_original_pp'] = min(r['accuracy_difference_vs_original_pp'] for r in rows)
            summaries.append(item)
    fixed.write_csv(args.output_dir/'candidate_summary.csv', summaries)
    metadata = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    numpy=np.__version__,numba=fixed.numba.__version__,python=platform.python_version(),
                    platform=platform.platform(), minimum_iteration=5001, window_iterations=500, patience_windows=3,
                    candidates=CANDIDATES, wine='new continuous 30000 reference, early_stop=False; original stop observed separately',
                    digits='previous registered continuous trajectory; no new Digits execution',
                    digits_inputs_sha256={name:hashlib.sha256((digits/(name+'.csv')).read_bytes()).hexdigest()
                                          for name in ('checks','snapshots')},
                    wine_inputs_sha256={name:hashlib.sha256((args.output_dir/(name+'.csv')).read_bytes()).hexdigest()
                                        for name in ('wine_checks','wine_snapshots')})
    (args.output_dir/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,dataset in zip(axes,('Digits','Wine')):
        for name,_,_ in CANDIDATES:
            rows=[r for r in all_rows if r['dataset']==dataset and r['candidate']==name]
            ax.scatter([100*r['saved_fraction'] for r in rows],[r['accuracy_difference_pp'] for r in rows],label=name,s=25,alpha=.7)
        ax.axhline(0,color='black',linestyle='--')
        ax.set_title(dataset)
        ax.set_xlabel('Saved iterations vs 30000 (%)')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Accuracy change vs full 30000 (percentage points)')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir/'candidate_tradeoffs.png',dpi=160)
    plt.close(fig)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('docs/benchmarks/stopping_candidates'))
    parser.add_argument('--reuse-wine',action='store_true',help='reuse Wine trajectories already collected in output-dir')
    main(parser.parse_args())
