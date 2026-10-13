"""Optional continuous-trajectory diagnostics; production kernels are unchanged."""
import argparse
import copy
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import benchmark_backends as fixed
from benchmark_backend_convergence import prepare_cases
from tests.test_sequential_safety import next_random


def initial_arguments(case, protocol, cap):
    model = fixed.PCC(impl='numba')
    model.set_graph(case['edges'], case['degrees'], k_nn=protocol['k_nn'])
    captured = []
    def capture(*args):
        captured.extend(copy.deepcopy(args))
    model._get_backend_fn = lambda: capture
    parameters = {k: protocol[k] for k in ('p_grd', 'delta_v', 'deltap', 'dexp', 'es_chk')}
    model.fit_predict(case['labels'], max_iter=cap, early_stop=True, **parameters)
    return captured


def predictions(args):
    labels = args[2].copy()
    unknown = labels == -1
    labels[unknown] = np.argmax(args[11][unknown], axis=1)
    return labels


def trace(initial, case, cap, spacing, stop_at_original=True):
    # Validate once; keep the same buffers, particle state and thread-local RNG.
    args = copy.deepcopy(initial)
    max_degree = fixed.pcc_numba.validate_step_inputs(*args[:3], *args[5:13])
    fixed.pcc_numba.validate_buffers(args[5], max_degree, *args[15:22])
    fixed.seed_numba(case['seed'])
    kernel = fixed.pcc_numba._pcc_propagate_numba_sequential
    calc = fixed.pcc_numba._pcc_calc_mmpot_sequential
    n, c = args[0].shape[0], args[5]
    unknown = case['labels'] == -1
    previous_labels = predictions(args)
    best = previous_metric = 0.0
    stalled = resets = longest_stall = 0
    since_resets = since_checks = 0
    checks, snapshots = [], []
    current, last_snapshot = 0, 0
    first_original_stop = None
    start = time.perf_counter()
    while current < cap:
        target = min(cap, 1 if current == 0 else current + 10)
        # early_stop=False suppresses only the local block's counter. The original
        # metric and strict stop predicate are evaluated below on the global schedule.
        kernel(*args[:22], target-current, False, args[24], args[25], False)
        current = target
        metric = float(calc(args[11], n, c))
        is_check = (current-1) % 10 == 0
        improvement = max(0., metric-best) if is_check else 0.
        improved = is_check and metric > best
        if is_check:
            if improved:
                best, stalled = metric, 0
                resets += 1
                since_resets += 1
            else:
                stalled += 1
                longest_stall = max(longest_stall, stalled)
            since_checks += 1
            checks.append(dict(seed=case['seed'], iteration=current, check=len(checks)+1,
                               metric=metric, delta_from_previous_check=metric-previous_metric,
                               best_metric=best, best_improvement=improvement,
                               reset=int(improved), stalled_checks=stalled,
                               cumulative_resets=resets))
            previous_metric = metric
        stopped = is_check and stalled > args[25]
        if stopped and first_original_stop is None:
            first_original_stop = current
        if current == 1 or current-last_snapshot >= spacing or (stopped and stop_at_original) or current == cap:
            predicted = predictions(args)
            changed = int(np.count_nonzero(predicted != previous_labels))
            snapshots.append(dict(seed=case['seed'], iteration=current,
                                  interval_iterations=current-last_snapshot,
                                  original_check=int(is_check), metric=metric,
                                  best_metric=best, stalled_checks=stalled,
                                  cumulative_resets=resets, interval_resets=since_resets,
                                  interval_checks=since_checks,
                                  reset_frequency=since_resets/since_checks if since_checks else 0.,
                                  changed_predictions=changed,
                                  changed_fraction_unlabeled=changed/int(unknown.sum()),
                                  accuracy_unlabeled=float(np.mean(predicted[unknown] == case['truth'][unknown]))))
            previous_labels, last_snapshot = predicted, current
            since_resets = since_checks = 0
        if stopped and stop_at_original:
            break
    elapsed = time.perf_counter()-start
    return args, float(next_random()), checks, snapshots, dict(
        seed=case['seed'], effective_iterations=current, checks=len(checks),
        stop_reason='early_stop' if stopped and stop_at_original else 'max_iter', stop_max=args[25],
        first_original_stop=first_original_stop,
        stalled_checks=stalled, longest_stall_checks=longest_stall,
        reset_count=resets, diagnostic_s=elapsed,
        final_accuracy_unlabeled=snapshots[-1]['accuracy_unlabeled'])


def plot(checks, snapshots, output):
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for seed in range(5):
        ch = [r for r in checks if r['seed'] == seed]
        ss = [r for r in snapshots if r['seed'] == seed]
        x = [r['iteration'] for r in ss]
        axes[0].plot(x, [r['metric'] for r in ss], label=f'seed {seed}')
        axes[1].plot(x, [r['changed_predictions'] for r in ss])
        axes[2].plot(x, [r['accuracy_unlabeled'] for r in ss])
        axes[3].plot([r['iteration'] for r in ch], [r['stalled_checks'] for r in ch])
    for ax, label in zip(axes, ('Mean maximum dominance', 'Changed predictions since previous snapshot',
                               'Unlabeled accuracy', 'Stalled checks')):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].legend(ncol=5)
    axes[-1].axhline(1199, linestyle='--', color='black', label='Stopping counter: 1199')
    axes[-1].legend()
    axes[-1].set_xlabel('Actual iteration')
    fig.suptitle('Digits: continuous Numba sequential trajectories (original PCC)')
    fig.tight_layout()
    fig.savefig(output/'trajectories.png', dpi=160)
    fig.savefig(output/'trajectories.svg')
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for seed in range(5):
        ss = [r for r in snapshots if r['seed'] == seed and r['iteration'] >= 20001]
        x = [r['iteration'] for r in ss]
        for ax, key in zip(axes, ('metric', 'changed_predictions', 'accuracy_unlabeled')):
            ax.plot(x, [r[key] for r in ss], marker='.', label=f'seed {seed}')
    for ax, label in zip(axes, ('Mean maximum dominance', 'Changed predictions', 'Unlabeled accuracy')):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].legend(ncol=5)
    axes[-1].set_xlabel('Actual iteration')
    fig.suptitle('Late trajectory: near-stable labels while dominance grows')
    fig.tight_layout()
    fig.savefig(output/'late_trajectories.png', dpi=160)
    fig.savefig(output/'late_trajectories.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for seed in range(5):
        ch = [r for r in checks if r['seed'] == seed and r['reset'] and r['iteration'] > 1]
        ss = [r for r in snapshots if r['seed'] == seed]
        axes[0].scatter([r['iteration'] for r in ch], [r['best_improvement'] for r in ch],
                        s=3, alpha=.5, label=f'seed {seed}')
        axes[1].plot([r['iteration'] for r in ss], [r['reset_frequency'] for r in ss])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Improvement over running best (log)')
    axes[0].legend(ncol=5)
    axes[1].set_ylabel('Reset fraction in snapshot interval')
    axes[1].set_xlabel('Actual iteration')
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output/'improvements.png', dpi=160)
    fig.savefig(output/'improvements.svg')
    plt.close(fig)


def main(args):
    if platform.system() != 'Windows' or not fixed.pcc_step.__file__.endswith('.pyd'):
        raise RuntimeError('requires Windows pypcc-dev with native Cython')
    baseline, cases = prepare_cases(Path('docs/benchmarks/backend_comparison.json'))
    cases = [case for case in cases if case['dataset'] == 'Digits']
    protocol = baseline['protocol']
    all_checks, all_snapshots, summaries = [], [], []
    old_rows = list(csv.DictReader(Path(
        'docs/benchmarks/convergence_30000/backend_convergence.csv').open()))
    # Warm both return signatures, metric and RNG observer before timing.
    first = initial_arguments(cases[0], protocol, args.iterations)
    for flag in (False, True):
        warm = copy.deepcopy(first)
        fixed.pcc_numba.pcc_propagate_numba(*warm[:22], 2, True, *warm[24:], return_info=flag)
    fixed.pcc_numba._pcc_calc_mmpot_sequential(first[11], first[0].shape[0], first[5])
    next_random()
    signatures = tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures)
    for case in cases:
        initial = initial_arguments(case, protocol, args.iterations)
        final, rng, checks, snapshots, summary = trace(initial, case, args.iterations, args.spacing)
        assert signatures == tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures), 'JIT during diagnosis'
        timings = []
        for flag in (False, True):
            mono = copy.deepcopy(initial)
            fixed.seed_numba(case['seed'])
            begin = time.perf_counter()
            info = fixed.pcc_numba.pcc_propagate_numba(*mono, return_info=flag)
            timings.append(time.perf_counter()-begin)
            for index in (7, 9, 10, 11, 12):
                np.testing.assert_array_equal(final[index], mono[index])
            assert rng == float(next_random()), 'continuous trajectory consumed different RNG'
            if flag:
                for key in ('effective_iterations', 'checks', 'stop_reason', 'stop_max', 'stalled_checks'):
                    assert info[key] == summary[key]
            else:
                assert info is None
        if args.iterations == 30000:
            old = next(r for r in old_rows if r['dataset'] == 'Digits' and
                       r['configuration'] == 'numba_sequential' and int(r['seed']) == case['seed'])
            assert fixed.fingerprint(predictions(final)) == old['prediction_sha256']
        summary.update(monolithic_false_s=timings[0], monolithic_true_s=timings[1],
                       diagnostic_overhead_ratio=summary['diagnostic_s']/timings[1],
                       continuous_state_rng_verified=True,
                       data_sha256=baseline['datasets']['Digits']['data_sha256'],
                       graph_sha256=fixed.fingerprint(case['edges'],case['degrees']),
                       labels_sha256=fixed.fingerprint(case['labels']))
        if args.iterations >= 20001:
            late = [row for row in snapshots if row['iteration'] > 20001]
            improvements = [row['best_improvement'] for row in checks
                            if row['iteration'] > 20001 and row['reset']]
            base = next(row for row in checks if row['iteration'] == 20001)
            summary.update(late_reset_count=len(improvements),
                           late_snapshot_count=len(late),
                           late_zero_change_snapshots=sum(row['changed_predictions'] == 0 for row in late),
                           late_max_changed_predictions=max((row['changed_predictions'] for row in late), default=0),
                           late_mean_changed_predictions=float(np.mean([row['changed_predictions'] for row in late])) if late else 0.,
                           late_metric_gain=snapshots[-1]['metric']-base['metric'],
                           late_improvement_min=min(improvements, default=0.),
                           late_improvement_median=float(np.median(improvements)) if improvements else 0.,
                           late_improvement_max=max(improvements, default=0.),
                           late_improvements_below_1e_minus12=sum(value < 1e-12 for value in improvements))
        all_checks.extend(checks)
        all_snapshots.extend(snapshots)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (('checks', all_checks), ('snapshots', all_snapshots), ('summary', summaries)):
        fixed.write_csv(args.output_dir/(name+'.csv'), rows)
    metadata = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    protocol=dict(protocol, early_stop=True, max_iter=args.iterations),
                    snapshot_spacing=args.spacing, numpy=np.__version__, numba=fixed.numba.__version__,
                    matplotlib=matplotlib.__version__, python=platform.python_version(),
                    platform=platform.platform(), executable=__import__('sys').executable,
                    method='continuous compiled blocks; original metric/counter outside unchanged kernels')
    (args.output_dir/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    plot(all_checks, all_snapshots, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=30000)
    parser.add_argument('--spacing',type=int,default=500)
    parser.add_argument('--output-dir',type=Path,default=Path('docs/benchmarks/digits_stopping_diagnostic'))
    args=parser.parse_args()
    if args.iterations < 1 or args.spacing < 10 or args.spacing % 10:
        parser.error('positive iterations and snapshot spacing divisible by ten required')
    main(args)
