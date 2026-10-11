"""Optional fixed-budget backend benchmark; run from the repository root.

    conda run -n pypcc-dev python docs/benchmark_backends.py

No backend is patched. Every measured run executes the same iteration budget.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import numba
import Cython
import scipy
import sklearn
import setuptools
from numba import njit
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcc_numba
import pcc_numpy
import pcc_step
from pcc import ParticleCompetitionAndCooperation as PCC

CONFIGURATIONS = {
    'numpy_sequential': ('numpy', 'sequential', pcc_numpy.pcc_propagate_numpy),
    'numba_sequential': ('numba', 'sequential', pcc_numba.pcc_propagate_numba),
    'cython_sequential': ('cython', 'sequential', pcc_step.pcc_propagate),
    'numpy_synchronous_experimental': ('numpy', 'synchronous', pcc_numpy.pcc_propagate_numpy),
}
HYPERPARAMETERS = dict(p_grd=.5, delta_v=.1, deltap=1., dexp=2.,
                       early_stop=False, es_chk=2000)


@njit
def seed_numba(seed):
    np.random.seed(seed)


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def processor_name():
    if platform.system() == 'Windows':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(configuration, edges, degrees, labels, truth, seed, iterations, k_nn):
    impl, mode, expected_backend = CONFIGURATIONS[configuration]
    # Seed outside the timer; NumPy and Numba have independent RNG states.
    np.random.seed(seed)
    if impl == 'numba':
        seed_numba(seed)
    start = time.perf_counter()
    model = PCC(impl=impl, update_mode=mode)
    model.set_graph(edges, degrees, k_nn=k_nn)
    backend = model._get_backend_fn()
    if backend is not expected_backend:
        raise RuntimeError(f'{configuration}: unexpected backend fallback')
    measured = {}

    def timed_backend(*args, **kwargs):
        if mode == 'synchronous':
            kwargs['update_mode'] = mode
        begin = time.perf_counter()
        measured['preparation_s'] = begin - start
        result = backend(*args, **kwargs)
        end = time.perf_counter()
        measured['propagation_s'] = end - begin
        measured['propagation_end'] = end
        return result

    # Only wrap the public call boundary; the production loop/RNG is unchanged.
    model._get_backend_fn = lambda: timed_backend
    predicted = model.fit_predict(labels, max_iter=iterations, **HYPERPARAMETERS)
    end = time.perf_counter()
    measured['postprocessing_s'] = end - measured.pop('propagation_end')
    measured['total_fit_s'] = end - start
    unknown = labels == -1
    np.testing.assert_array_equal(predicted[~unknown], truth[~unknown])
    dominance = model.node.dominance
    assert np.isfinite(dominance).all()
    assert np.all((dominance >= -1e-12) & (dominance <= 1. + 1e-12))
    np.testing.assert_allclose(dominance.sum(axis=1), 1., atol=1e-12)
    assert np.all((model.part.curnode >= 0) & (model.part.curnode < len(truth)))
    measured.update(accuracy_unlabeled=float(np.mean(predicted[unknown] == truth[unknown])),
                    effective_iterations=iterations, stop_reason='max_iter',
                    early_stop=False, es_chk=2000,
                    iteration_count_source='fixed_budget_no_early_stop',
                    rng_seed_effective=123456789 if impl == 'cython' else seed,
                    prediction_sha256=fingerprint(predicted))
    return measured, predicted


def summarize(rows):
    summary = []
    for dataset in dict.fromkeys(row['dataset'] for row in rows):
        baseline = {row['seed']: row for row in rows
                    if row['dataset'] == dataset and row['configuration'] == 'numpy_sequential'}
        for configuration in CONFIGURATIONS:
            selected = [row for row in rows
                        if row['dataset'] == dataset and row['configuration'] == configuration]
            item = dict(dataset=dataset, configuration=configuration, runs=len(selected))
            for field in ('preparation_s', 'propagation_s', 'postprocessing_s', 'total_fit_s',
                          'accuracy_unlabeled', 'effective_iterations',
                          'prediction_mismatches_vs_numpy_sequential'):
                values = np.array([row[field] for row in selected])
                item[field + '_mean'] = float(values.mean())
                item[field + '_std'] = float(values.std(ddof=1))
            baseline_time = np.mean([row['propagation_s'] for row in baseline.values()])
            item['propagation_speedup_ratio_of_means'] = float(baseline_time / item['propagation_s_mean'])
            paired = [baseline[row['seed']]['propagation_s'] / row['propagation_s'] for row in selected]
            item['paired_speedup_mean'] = float(np.mean(paired))
            item['paired_speedup_std'] = float(np.std(paired, ddof=1))
            differences = [row['accuracy_unlabeled'] - baseline[row['seed']]['accuracy_unlabeled']
                           for row in selected]
            item['paired_accuracy_difference_mean'] = float(np.mean(differences))
            item['paired_accuracy_difference_std'] = float(np.std(differences, ddof=1))
            summary.append(item)
    return summary


def benchmark(args):
    if platform.system() != 'Windows' or not pcc_step.__file__.endswith('.pyd'):
        raise RuntimeError('This protocol requires Windows and a native Cython .pyd')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / 'backend_comparison'
    metadata = dict(
        environment=dict(python=sys.version, executable=sys.executable,
                         conda_prefix=os.environ.get('CONDA_PREFIX'),
                         platform=platform.platform(), processor=processor_name(),
                         logical_cpus=os.cpu_count(), cython_binary=pcc_step.__file__,
                         cython_binary_sha256=hashlib.sha256(Path(pcc_step.__file__).read_bytes()).hexdigest(),
                         versions={module.__name__: module.__version__ for module in
                                   (np, numba, Cython, scipy, sklearn, setuptools)}),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        protocol=dict(seeds=args.seeds, repetitions=len(args.seeds), max_iter=args.iterations,
                      k_nn=args.k_nn, labels_per_class=3, **HYPERPARAMETERS,
                      standardization='StandardScaler on full transductive feature matrix',
                      split='three known nodes per class; all remaining nodes evaluated',
                      order='cyclic rotation of four configurations by dataset index + repetition',
                      std='sample standard deviation, ddof=1',
                      cython_rng='backend resets xorshift32 to 123456789 on each propagation call',
                      synchronous='experimental variant, not sequentially equivalent'),
        datasets={}, warmups=[])
    # Compile the seeding helper separately; do not charge it to a fit or propagation.
    begin = time.perf_counter()
    seed_numba(0)
    metadata['numba_seed_helper_first_call_s'] = time.perf_counter() - begin
    rows = []
    for dataset_index, (name, load) in enumerate((('Wine', load_wine), ('Digits', load_digits))):
        begin = time.perf_counter()
        data = load()
        truth = np.asarray(data.target, dtype=np.int64)
        loaded = time.perf_counter()
        x = StandardScaler().fit_transform(data.data)
        scaled = time.perf_counter()
        graph = PCC(impl='numpy')
        graph.build_graph(x, k_nn=args.k_nn, n_jobs=1)
        built = time.perf_counter()
        edges, degrees = graph.neib_list, graph.neib_qt
        info = dict(nodes=len(truth), features=x.shape[1], classes=len(np.unique(truth)),
                    data_load_s=loaded-begin, standardization_s=scaled-loaded,
                    graph_build_s=built-scaled, shared_preparation_s=built-begin,
                    max_degree=int(degrees.max()), data_sha256=fingerprint(x, truth),
                    graph_sha256=fingerprint(edges, degrees), splits=[])
        metadata['datasets'][name] = info
        for repeat, seed in enumerate(args.seeds):
            rng = np.random.default_rng(seed)
            known = np.concatenate([rng.choice(np.flatnonzero(truth == cls), 3, replace=False)
                                    for cls in np.unique(truth)])
            labels = np.full(len(truth), -1, dtype=np.int64)
            labels[known] = truth[known]
            split_hash = fingerprint(labels)
            info['splits'].append(dict(seed=seed, known_indices=known.tolist(), labels_sha256=split_hash))
            if repeat == 0:
                for configuration in CONFIGURATIONS:
                    for call in ('first', 'warm'):
                        values, _ = run(configuration, edges, degrees, labels, truth, seed, 2, args.k_nn)
                        metadata['warmups'].append(dict(dataset=name, configuration=configuration,
                                                       call=call, **values))
                        print(json.dumps(dict(warmup=True, dataset=name, configuration=configuration,
                                              call=call, **values)), flush=True)
            order = list(CONFIGURATIONS)
            offset = (dataset_index + repeat) % len(order)
            order = order[offset:] + order[:offset]
            predictions, batch = {}, []
            for index, configuration in enumerate(order):
                signatures = tuple(pcc_numba._pcc_propagate_numba_sequential.signatures)
                values, predicted = run(configuration, edges, degrees, labels, truth,
                                        seed, args.iterations, args.k_nn)
                if configuration == 'numba_sequential':
                    assert signatures and signatures == tuple(
                        pcc_numba._pcc_propagate_numba_sequential.signatures), 'JIT during measured run'
                predictions[configuration] = predicted
                row = dict(dataset=name, configuration=configuration, repeat=repeat,
                           seed=seed, order_index=index, nodes=len(truth),
                           labeled_nodes=len(known), unlabeled_nodes=len(truth)-len(known),
                           data_sha256=info['data_sha256'], graph_sha256=info['graph_sha256'],
                           labels_sha256=split_hash, shared_preparation_s=info['shared_preparation_s'],
                           k_nn=args.k_nn, p_grd=.5, delta_v=.1, deltap=1., dexp=2., **values)
                batch.append(row)
                print(json.dumps(row), flush=True)
            for row in batch:
                row['prediction_mismatches_vs_numpy_sequential'] = int(np.count_nonzero(
                    predictions[row['configuration']] != predictions['numpy_sequential']))
            rows.extend(batch)
            # Keep completed executions recoverable even if a later execution fails.
            write_csv(prefix.with_suffix('.csv'), rows)
            prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    summary = summarize(rows)
    write_csv(args.output_dir / 'backend_comparison_summary.csv', summary)
    write_csv(args.output_dir / 'backend_comparison_warmups.csv', metadata['warmups'])
    metadata['summary'] = summary
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/benchmarks'))
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--k-nn', type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1 or len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('require iterations >= 1 and at least two distinct seeds')
    benchmark(args)
