"""Convergence benchmark using the fixed-budget benchmark's exact input cases."""
import argparse
import csv
import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler

from benchmark_backends import PCC, fingerprint
import benchmark_backends as fixed


def prepare_cases(baseline_path):
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    protocol = baseline['protocol']
    assert protocol['seeds'] == [0, 1, 2, 3, 4]
    cases = []
    for name, load in (('Wine', load_wine), ('Digits', load_digits)):
        begin = time.perf_counter()
        data = load()
        truth = np.asarray(data.target, dtype=np.int64)
        x = StandardScaler().fit_transform(data.data)
        graph = PCC(impl='numpy')
        graph.build_graph(x, k_nn=protocol['k_nn'], n_jobs=1)
        setup_s = time.perf_counter() - begin
        saved = baseline['datasets'][name]
        assert fingerprint(x, truth) == saved['data_sha256'], 'dataset differs from PR #4'
        assert fingerprint(graph.neib_list, graph.neib_qt) == saved['graph_sha256'], 'graph differs from PR #4'
        for split in saved['splits']:
            labels = np.full(len(truth), -1, dtype=np.int64)
            known = np.array(split['known_indices'], dtype=np.int64)
            labels[known] = truth[known]
            assert fingerprint(labels) == split['labels_sha256'], 'initial labels differ from PR #4'
            stop_max = max(1, round((len(truth) / (len(known) * protocol['k_nn']))
                                    * round(protocol['es_chk'] * .1)))
            cases.append(dict(dataset=name, seed=split['seed'], truth=truth,
                              edges=graph.neib_list, degrees=graph.neib_qt,
                              labels=labels, stop_max=stop_max,
                              shared_dataset_setup_s=setup_s,
                              earliest_possible_stop=1 + 10 * (stop_max + 1)))
    return baseline, cases


def run(configuration, case, protocol, iterations):
    impl, mode, expected_backend = fixed.CONFIGURATIONS[configuration]
    np.random.seed(case['seed'])
    if impl == 'numba':
        fixed.seed_numba(case['seed'])
    begin = time.perf_counter()
    model = PCC(impl=impl, update_mode=mode)
    model.set_graph(case['edges'], case['degrees'], k_nn=protocol['k_nn'])
    backend = model._get_backend_fn()
    assert backend is expected_backend, 'unexpected backend fallback'
    measurement = {}

    def timed_backend(*args, **kwargs):
        kwargs['return_info'] = True
        if mode == 'synchronous':
            kwargs['update_mode'] = mode
        start = time.perf_counter()
        preparation_s = start - begin
        info = backend(*args, **kwargs)
        end = time.perf_counter()
        measurement.update(preparation_s=preparation_s, propagation_s=end-start,
                           propagation_end=end, **info)

    model._get_backend_fn = lambda: timed_backend
    parameters = {key: protocol[key] for key in ('p_grd', 'delta_v', 'deltap', 'dexp', 'es_chk')}
    predicted = model.fit_predict(case['labels'], max_iter=iterations, early_stop=True, **parameters)
    end = time.perf_counter()
    measurement.update(postprocessing_s=end-measurement.pop('propagation_end'), total_fit_s=end-begin)
    count, checks = measurement['effective_iterations'], measurement['checks']
    assert 0 < count <= iterations
    assert checks == (count + 9) // 10
    assert measurement['stop_max'] == case['stop_max']
    if measurement['stop_reason'] == 'early_stop':
        assert (count - 1) % 10 == 0
        assert measurement['stalled_checks'] == case['stop_max'] + 1
        assert count >= case['earliest_possible_stop']
    else:
        assert count == iterations and measurement['stalled_checks'] <= case['stop_max']
    if iterations < case['earliest_possible_stop']:
        assert measurement['stop_reason'] == 'max_iter'
    unknown = case['labels'] == -1
    np.testing.assert_array_equal(predicted[~unknown], case['truth'][~unknown])
    dominance = model.node.dominance
    assert np.isfinite(dominance).all()
    assert np.all((dominance >= -1e-12) & (dominance <= 1. + 1e-12))
    np.testing.assert_allclose(dominance.sum(axis=1), 1., atol=1e-12)
    assert np.all((model.part.curnode >= 0) & (model.part.curnode < len(predicted)))
    measurement.update(accuracy_unlabeled=float(np.mean(predicted[unknown] == case['truth'][unknown])),
                       prediction_sha256=fingerprint(predicted),
                       rng_seed_effective=123456789 if impl == 'cython' else case['seed'],
                       propagation_us_per_iteration=1e6*measurement['propagation_s']/count,
                       stopping_audit_passed=True)
    return measurement, predicted


def benchmark(args, baseline, cases):
    if platform.system() != 'Windows' or not fixed.pcc_step.__file__.endswith('.pyd'):
        raise RuntimeError('requires Windows and native Cython')
    protocol = dict(baseline['protocol'], early_stop=True, max_iter=args.iterations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / 'backend_convergence'
    with args.baseline.with_suffix('.csv').open(encoding='utf-8', newline='') as stream:
        baseline_rows = {(row['dataset'], row['configuration'], int(row['seed'])): row
                         for row in csv.DictReader(stream)}
    metadata = dict(protocol=protocol, baseline_json_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    environment=dict(python=sys.version, executable=sys.executable,
                                     conda_prefix=os.environ.get('CONDA_PREFIX'), platform=platform.platform(),
                                     processor=fixed.processor_name(),
                                     versions={m.__name__: m.__version__ for m in
                                               (np, fixed.numba, fixed.Cython, fixed.scipy,
                                                fixed.sklearn, fixed.setuptools)},
                                     cython_binary_sha256=hashlib.sha256(
                                         Path(fixed.pcc_step.__file__).read_bytes()).hexdigest()),
                    warmups=[], cases=[{key: case[key] for key in
                                        ('dataset', 'seed', 'stop_max', 'earliest_possible_stop',
                                         'shared_dataset_setup_s')} for case in cases])
    begin = time.perf_counter()
    fixed.seed_numba(0)
    metadata['numba_seed_helper_first_call_s'] = time.perf_counter() - begin
    rows = []
    for case in cases:
        name, seed = case['dataset'], case['seed']
        if seed == 0:
            for configuration in fixed.CONFIGURATIONS:
                for call in ('first', 'warm'):
                    values, _ = run(configuration, case, protocol, 2)
                    metadata['warmups'].append(dict(dataset=name, configuration=configuration,
                                                   call=call, **values))
        order = list(fixed.CONFIGURATIONS)
        offset = (int(name == 'Digits') + seed) % len(order)
        order = order[offset:] + order[:offset]
        predictions, batch = {}, []
        for index, configuration in enumerate(order):
            signatures = tuple(fixed.pcc_numba._pcc_propagate_numba_sequential.signatures)
            values, predicted = run(configuration, case, protocol, args.iterations)
            if configuration == 'numba_sequential':
                assert signatures and signatures == tuple(
                    fixed.pcc_numba._pcc_propagate_numba_sequential.signatures), 'JIT during measured run'
            old = baseline_rows[name, configuration, seed]
            matches = values['prediction_sha256'] == old['prediction_sha256']
            if args.iterations == 10000 and values['effective_iterations'] == 10000:
                assert matches, 'capped trajectory differs from the preserved fixed-budget run'
            row = dict(dataset=name, configuration=configuration, seed=seed, repeat=seed, order_index=index,
                       data_sha256=baseline['datasets'][name]['data_sha256'],
                       graph_sha256=fingerprint(case['edges'], case['degrees']),
                       labels_sha256=fingerprint(case['labels']), early_stop=True, max_iter=args.iterations,
                       k_nn=protocol['k_nn'], p_grd=protocol['p_grd'], delta_v=protocol['delta_v'],
                       deltap=protocol['deltap'], dexp=protocol['dexp'], es_chk=protocol['es_chk'],
                       shared_dataset_setup_s=case['shared_dataset_setup_s'],
                       earliest_possible_stop=case['earliest_possible_stop'],
                       fixed_budget_prediction_match=matches, **values)
            predictions[configuration] = predicted
            batch.append(row)
            print(json.dumps(row), flush=True)
        for row in batch:
            row['prediction_mismatches_vs_numpy_sequential'] = int(np.count_nonzero(
                predictions[row['configuration']] != predictions['numpy_sequential']))
        rows.extend(batch)
        fixed.write_csv(prefix.with_suffix('.csv'), rows)
        prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    summary = fixed.summarize(rows)
    for item in summary:
        selected = [r for r in rows if (r['dataset'], r['configuration']) ==
                    (item['dataset'], item['configuration'])]
        base = [r for r in rows if r['dataset'] == item['dataset'] and r['configuration'] == 'numpy_sequential']
        for key in ('checks', 'stalled_checks', 'propagation_us_per_iteration'):
            values = [r[key] for r in selected]
            item[key+'_mean'], item[key+'_std'] = float(np.mean(values)), float(np.std(values, ddof=1))
        item['early_stop_runs'] = sum(r['stop_reason'] == 'early_stop' for r in selected)
        item['max_iter_runs'] = sum(r['stop_reason'] == 'max_iter' for r in selected)
        item['per_iteration_speedup_ratio_of_means'] = float(
            np.mean([r['propagation_us_per_iteration'] for r in base]) /
            item['propagation_us_per_iteration_mean'])
    metadata['summary'] = summary
    fixed.write_csv(args.output_dir / 'backend_convergence_summary.csv', summary)
    fixed.write_csv(args.output_dir / 'backend_convergence_warmups.csv', metadata['warmups'])
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path,
                        default=Path('docs/benchmarks/backend_comparison.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('docs/benchmarks'))
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    baseline, cases = prepare_cases(args.baseline)
    if args.prepare_only:
        for case in cases:
            print(json.dumps({key: case[key] for key in
                              ('dataset', 'seed', 'stop_max', 'earliest_possible_stop')}), flush=True)
    else:
        benchmark(args, baseline, cases)
