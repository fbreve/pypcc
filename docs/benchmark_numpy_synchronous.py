"""Optional, reproducible Wine/Digits comparison; excluded from regression CI.

Run from the repository root:
    python docs/benchmark_numpy_synchronous.py --output results.json
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcc import ParticleCompetitionAndCooperation


def benchmark(seeds, iterations, repeats):
    rows = []
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
            models = {mode: ParticleCompetitionAndCooperation(impl='numpy', update_mode=mode)
                      for mode in ('sequential', 'synchronous')}
            for model in models.values():
                model.set_graph(graph.neib_list, graph.neib_qt)
                np.random.seed(seed)
                model.fit_predict(observed, max_iter=2, early_stop=False)
            times = {mode: [] for mode in models}
            predictions = {}
            for repeat in range(repeats):
                order = list(models) if (seed + repeat) % 2 == 0 else list(reversed(models))
                for mode in order:
                    model = models[mode]
                    np.random.seed(seed)
                    start = time.perf_counter()
                    predicted = model.fit_predict(observed, max_iter=iterations, early_stop=False)
                    times[mode].append(time.perf_counter() - start)
                    np.testing.assert_array_equal(predicted[known], truth[known])
                    dom = model.node.dominance
                    # Sequential arithmetic can exceed 1 by a few ulps.
                    assert np.isfinite(dom).all() and np.all((dom >= -1e-12) & (dom <= 1. + 1e-12))
                    np.testing.assert_allclose(dom.sum(axis=1), 1., atol=1e-12)
                    if mode in predictions:
                        np.testing.assert_array_equal(predicted, predictions[mode])
                    predictions[mode] = predicted.copy()
            row = dict(dataset=name, seed=seed,
                       mismatches=int(np.count_nonzero(predictions['sequential'] !=
                                                       predictions['synchronous'])))
            for mode in models:
                row[mode] = dict(seconds=float(np.median(times[mode])),
                                 accuracy=float(np.mean(predictions[mode][observed == -1] ==
                                                        truth[observed == -1])),
                                 timings=times[mode])
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
        differences = np.array([row['synchronous']['accuracy'] - row['sequential']['accuracy']
                                for row in selected])
        item.update(speedup=item['sequential']['mean_seconds'] / item['synchronous']['mean_seconds'],
                    mean_accuracy_difference=float(differences.mean()),
                    std_accuracy_difference=float(differences.std(ddof=1)),
                    synchronous_wins=int(np.sum(differences > 0)),
                    ties=int(np.sum(differences == 0)),
                    synchronous_losses=int(np.sum(differences < 0)),
                    seeds_with_prediction_differences=sum(row['mismatches'] > 0 for row in selected))
        summary[dataset] = item
    return dict(environment=dict(python=sys.version, platform=platform.platform(),
                                 numpy=np.__version__, sklearn=sklearn.__version__),
                protocol=dict(seeds=seeds, iterations=iterations, repeats=repeats,
                              k_nn=10, labels_per_class=3, p_grd=.5, delta_v=.1,
                              deltap=1., dexp=2., early_stop=False), rows=rows, summary=summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.seeds < 2 or args.iterations < 1 or args.repeats < 1:
        parser.error('require seeds >= 2, iterations >= 1, repeats >= 1')
    result = benchmark(args.seeds, args.iterations, args.repeats)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))
