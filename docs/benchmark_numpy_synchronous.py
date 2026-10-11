"""Optional, reproducible Wine/Digits comparison; excluded from regression CI.

Run from the repository root:
    python docs/benchmark_numpy_synchronous.py --output results.json
"""
import argparse
import csv
import hashlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sklearn
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcc import ParticleCompetitionAndCooperation
import pcc_numpy


class PropagationProbe:
    """Benchmark-only observation; restore both patched functions after each run."""
    def __init__(self):
        self.backend = pcc_numpy.pcc_propagate_numpy
        self.step = pcc_numpy.pcc_step_numpy
        self.mean = np.mean
        self.result = None

    def __call__(self, *args, **kwargs):
        params = inspect.signature(self.backend).bind(*args, **kwargs).arguments
        iterations = checks = stalled = last_improvement = 0
        peak = last_value = 0.

        def count_step(*step_args, **step_kwargs):
            nonlocal iterations
            value = self.step(*step_args, **step_kwargs)
            iterations += 1
            return value

        def observe_mean(*mean_args, **mean_kwargs):
            nonlocal checks, stalled, peak, last_value, last_improvement
            value = self.mean(*mean_args, **mean_kwargs)
            # Observe the actual stopping statistic, without recomputing it.
            assert (iterations - 1) % 10 == 0
            checks += 1
            last_value = float(value)
            if value > peak:
                peak = float(value)
                stalled = 0
                last_improvement = iterations
            else:
                stalled += 1
            return value

        with patch.object(pcc_numpy, 'pcc_step_numpy', count_step), \
                patch.object(np, 'mean', observe_mean):
            start = time.perf_counter()
            returned = self.backend(*args, **kwargs)
            elapsed = time.perf_counter() - start
        stopped = bool(params['early_stop'] and stalled > params['stop_max'])
        assert iterations <= params['max_iter']
        if params['early_stop']:
            assert checks == (iterations + 9) // 10
            if stopped:
                assert (iterations - 1) % 10 == 0
                assert stalled == params['stop_max'] + 1
                # First eligible checkpoint after the last strict improvement.
                assert iterations == last_improvement + 10 * stalled
            else:
                assert iterations == params['max_iter']
        else:
            assert iterations == params['max_iter'] and checks == 0
        self.result = dict(seconds=elapsed, effective_iterations=iterations,
                           stop_reason='early_stop' if stopped else 'max_iter',
                           reached_max_iter=iterations == params['max_iter'],
                           stop_max=int(params['stop_max']), checks=checks,
                           stalled_checks=stalled, last_improvement_iteration=last_improvement,
                           final_mmpot=last_value, peak_mmpot=peak, stopping_audit_passed=True)
        return returned


def benchmark(seeds, iterations, repeats, early_stop=False, es_chk=2000):
    rows = []
    executions = []
    fingerprints = []
    for name, load in (('Wine', load_wine), ('Digits', load_digits)):
        data = load()
        x = StandardScaler().fit_transform(data.data)
        truth = data.target
        graph = ParticleCompetitionAndCooperation(impl='numpy')
        graph.build_graph(x, k_nn=10)
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            known = np.concatenate([
                rng.choice(np.flatnonzero(truth == label), 3, replace=False)
                for label in np.unique(truth)])
            observed = np.full(len(truth), -1, dtype=np.int64)
            observed[known] = truth[known]
            fingerprints.append(dict(dataset=name, seed=seed,
                                     graph_sha256=hashlib.sha256(graph.neib_list.tobytes() +
                                                                 graph.neib_qt.tobytes()).hexdigest(),
                                     labels_sha256=hashlib.sha256(observed.tobytes()).hexdigest()))
            models = {mode: ParticleCompetitionAndCooperation(impl='numpy', update_mode=mode)
                      for mode in ('sequential', 'synchronous')}
            for model in models.values():
                model.set_graph(graph.neib_list, graph.neib_qt)
                np.random.seed(seed)
                model.fit_predict(observed, max_iter=2, early_stop=False)
            probes = {mode: PropagationProbe() for mode in models}
            for mode, model in models.items():
                model._get_backend_fn = lambda mode=mode: probes[mode]
            times = {mode: [] for mode in models}
            iteration_counts = {mode: [] for mode in models}
            predictions = {}
            for repeat in range(repeats):
                order = list(models) if (seed + repeat) % 2 == 0 else list(reversed(models))
                for mode in order:
                    model = models[mode]
                    np.random.seed(seed)
                    predicted = model.fit_predict(observed, max_iter=iterations,
                                                  early_stop=early_stop, es_chk=es_chk)
                    measurement = probes[mode].result.copy()
                    times[mode].append(measurement['seconds'])
                    iteration_counts[mode].append(measurement['effective_iterations'])
                    np.testing.assert_array_equal(predicted[known], truth[known])
                    dom = model.node.dominance
                    # Sequential arithmetic can exceed 1 by a few ulps.
                    assert np.isfinite(dom).all() and np.all((dom >= -1e-12) & (dom <= 1. + 1e-12))
                    np.testing.assert_allclose(dom.sum(axis=1), 1., atol=1e-12)
                    if mode in predictions:
                        np.testing.assert_array_equal(predicted, predictions[mode])
                    predictions[mode] = predicted.copy()
                    measurement.update(dataset=name, seed=seed, repeat=repeat, mode=mode,
                                       order_index=order.index(mode),
                                       accuracy=float(np.mean(predicted[observed == -1] ==
                                                              truth[observed == -1])))
                    executions.append(measurement)
                    print(json.dumps(measurement), flush=True)
                mismatches = int(np.count_nonzero(predictions['sequential'] !=
                                                  predictions['synchronous']))
                for execution in executions[-2:]:
                    execution['prediction_mismatches'] = mismatches
            row = dict(dataset=name, seed=seed,
                       mismatches=int(np.count_nonzero(predictions['sequential'] !=
                                                       predictions['synchronous'])))
            for mode in models:
                row[mode] = dict(seconds=float(np.median(times[mode])),
                                 accuracy=float(np.mean(predictions[mode][observed == -1] ==
                                                        truth[observed == -1])),
                                 timings=times[mode], effective_iterations=iteration_counts[mode])
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for dataset in ('Wine', 'Digits'):
        selected = [row for row in rows if row['dataset'] == dataset]
        item = {}
        for mode in ('sequential', 'synchronous'):
            acc = np.array([row[mode]['accuracy'] for row in selected])
            item[mode] = dict(mean_seconds=float(np.mean([row[mode]['seconds'] for row in selected])),
                              mean_accuracy=float(acc.mean()),
                              std_accuracy=float(acc.std(ddof=1)))
            seed_times = np.array([row[mode]['seconds'] for row in selected])
            seed_iterations = np.array([row[mode]['effective_iterations'][0] for row in selected])
            assert all(len(set(row[mode]['effective_iterations'])) == 1 for row in selected)
            runs = [run for run in executions if run['dataset'] == dataset and run['mode'] == mode]
            item[mode].update(std_seconds=float(seed_times.std(ddof=1)),
                              mean_iterations=float(seed_iterations.mean()),
                              std_iterations=float(seed_iterations.std(ddof=1)),
                              mean_execution_seconds=float(np.mean([run['seconds'] for run in runs])),
                              std_execution_seconds=float(np.std([run['seconds'] for run in runs], ddof=1)),
                              reached_max_iter=sum(run['reached_max_iter'] for run in runs))
        differences = np.array([row['synchronous']['accuracy'] - row['sequential']['accuracy']
                                for row in selected])
        item.update(speedup=item['sequential']['mean_seconds'] / item['synchronous']['mean_seconds'],
                    mean_accuracy_difference=float(differences.mean()),
                    std_accuracy_difference=float(differences.std(ddof=1)),
                    synchronous_wins=int(np.sum(differences > 0)),
                    ties=int(np.sum(differences == 0)),
                    synchronous_losses=int(np.sum(differences < 0)),
                    seeds_with_prediction_differences=sum(row['mismatches'] > 0 for row in selected))
        ratios = np.array([row['sequential']['seconds'] / row['synchronous']['seconds']
                           for row in selected])
        time_differences = np.array([row['synchronous']['seconds'] - row['sequential']['seconds']
                                    for row in selected])
        mismatches = np.array([row['mismatches'] for row in selected])
        item.update(mean_paired_speedup=float(ratios.mean()),
                    std_paired_speedup=float(ratios.std(ddof=1)),
                    mean_paired_seconds_difference=float(time_differences.mean()),
                    std_paired_seconds_difference=float(time_differences.std(ddof=1)),
                    mean_prediction_mismatches=float(mismatches.mean()),
                    std_prediction_mismatches=float(mismatches.std(ddof=1)))
        summary[dataset] = item
    return dict(environment=dict(python=sys.version, platform=platform.platform(),
                                 numpy=np.__version__, sklearn=sklearn.__version__),
                protocol=dict(seeds=seeds, iterations=iterations, repeats=repeats,
                              k_nn=10, labels_per_class=3, p_grd=.5, delta_v=.1,
                              deltap=1., dexp=2., early_stop=early_stop, es_chk=es_chk,
                              timing='propagation only, with benchmark-only observation'),
                fingerprints=fingerprints, executions=executions, rows=rows, summary=summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--csv', type=Path, help='Optional per-execution propagation measurements')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--es-chk', type=int, default=2000)
    args = parser.parse_args()
    if args.seeds < 2 or args.iterations < 1 or args.repeats < 1:
        parser.error('require seeds >= 2, iterations >= 1, repeats >= 1')
    result = benchmark(args.seeds, args.iterations, args.repeats, args.early_stop, args.es_chk)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if args.csv:
        with args.csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['executions'][0]))
            writer.writeheader()
            writer.writerows(result['executions'])
    print(json.dumps(result['summary'], indent=2))
