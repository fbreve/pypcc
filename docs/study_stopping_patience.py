"""Patience-study inventory and feasibility gate; run from repository root.

conda run -n pypcc-dev python docs/study_stopping_patience.py --inventory
conda run -n pypcc-dev python docs/study_stopping_patience.py --feasibility
--full is reserved for the explicitly approved second phase, not this PR delivery.
"""
import argparse
import copy
import json
import os
import platform
import statistics
import subprocess
import sys
import uuid
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import benchmark_backends as fixed
from diagnose_digits_stopping import initial_arguments
from experiments.patience_data import load, sha, retrieve
from experiments.stopping_patience import run, identity, rng_state, STATE_INDICES, original_threshold
from experiments.patience_summary import summaries

ROOT = Path('docs/benchmarks/stopping_patience')
CACHE = Path('.cache/stopping_patience')


def protocol():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def protocol_hash():
    # Git may convert checkout newlines on Windows; provenance concerns contents.
    return sha((ROOT / 'protocol.json').read_bytes().replace(b'\r\n', b'\n'))


def memory_bytes():
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(value.PeakWorkingSetSize)
    import resource
    scale = 1 if platform.system() == 'Darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(), cpu=fixed.processor_name(),
        logical_cpus=os.cpu_count(), numpy=fixed.np.__version__, numba=fixed.numba.__version__,
        cython=fixed.Cython.__version__, scipy=fixed.scipy.__version__, sklearn=fixed.sklearn.__version__,
        setuptools=fixed.setuptools.__version__, native_cython_file=str(Path(fixed.pcc_step.__file__).resolve()),
        native_cython_sha256=sha(Path(fixed.pcc_step.__file__).read_bytes()),
        conda_prefix=os.environ.get('CONDA_PREFIX'),
        source_sha256={str(path): sha(path.read_bytes()) for path in (
            Path(__file__), Path('experiments/stopping_patience.py'), Path('experiments/patience_data.py'))},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())


def require_windows():
    if platform.system() != 'Windows' or not fixed.pcc_step.__file__.endswith('.pyd'):
        raise RuntimeError('Measurements require Windows and native Cython in pypcc-dev')


def inventory():
    p = protocol()
    manifest = ROOT / 'inventory.json'
    if manifest.exists():
        raise ValueError('Inventory is frozen; delete/rebuild only with explicit protocol review')
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in p['datasets']:
        started = time.perf_counter()
        raw_x, x, y, info = load(entry, CACHE / 'sources')
        load_s = time.perf_counter() - started
        started = time.perf_counter()
        graph = fixed.PCC(impl='numpy')
        graph.build_graph(x, k_nn=p['k_nn'], n_jobs=1)
        edges, degrees = graph.neib_list, graph.neib_qt
        graph_s = time.perf_counter() - started
        indices = np.concatenate([edges[i, :int(degrees[i])] for i in range(len(y))])
        indptr = np.concatenate([[0], np.cumsum(degrees)])
        adjacency = csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(y), len(y)))
        components = int(connected_components(adjacency, directed=False, return_labels=False))
        path = CACHE / (entry['name'] + '.npz')
        np.savez_compressed(path, edges=edges, degrees=degrees, truth=y)
        label_fingerprints = {}
        for seed in p['seeds']:
            label_fingerprints[str(seed)] = fixed.fingerprint(labels_for(y, seed, p))
        row = dict(name=entry['name'], **info, raw_data_fingerprint=fixed.fingerprint(raw_x, y),
            processed_data_fingerprint=fixed.fingerprint(x, y), graph_fingerprint=fixed.fingerprint(edges, degrees),
            label_fingerprints=label_fingerprints, labeled_nodes=3 * info['classes'],
            labeled_fraction=3 * info['classes'] / len(y), degree_min=int(degrees.min()),
            degree_mean=float(degrees.mean()), degree_max=int(degrees.max()),
            degree_std=float(degrees.std(ddof=1)), connected_components=components,
            directed_edges=int(degrees.sum()), load_download_s=load_s, graph_s=graph_s,
            graph_array_bytes=edges.nbytes + degrees.nbytes, peak_process_memory_bytes=memory_bytes())
        rows.append(row)
        print(f"Inventory {entry['name']}: n={len(y)}, c={info['classes']}, graph={graph_s:.3f}s", flush=True)
    value = dict(protocol_sha256=protocol_hash(),
                 selection_frozen_before_stopping_outcomes=True, environment=environment(), datasets=rows)
    manifest.write_text(json.dumps(value, indent=2), encoding='utf-8')
    inventory_csv(rows)


def inventory_csv(rows):
    fields = sorted(set().union(*(r.keys() for r in rows)) - {'class_counts', 'class_names', 'label_fingerprints'})
    fixed.write_csv(ROOT / 'inventory.csv', [{key: row.get(key) for key in fields} for row in rows])


def labels_for(truth, seed, p):
    labels = np.full(len(truth), -1, dtype=np.int64)
    random = np.random.default_rng(seed)
    for label in np.unique(truth):
        chosen = random.choice(np.flatnonzero(truth == label), p['labels_per_class'], replace=False)
        labels[chosen] = label
    return labels


def case_for(entry, seed, p, frozen):
    raw_x, x, y, _ = load(entry, CACHE / 'sources')
    if fixed.fingerprint(raw_x, y) != frozen['raw_data_fingerprint'] or fixed.fingerprint(x, y) != frozen['processed_data_fingerprint']:
        raise ValueError('Dataset differs from frozen inventory')
    path = CACHE / (entry['name'] + '.npz')
    if not path.exists():
        graph = fixed.PCC(impl='numpy')
        graph.build_graph(x, k_nn=p['k_nn'], n_jobs=1)
        np.savez_compressed(path, edges=graph.neib_list, degrees=graph.neib_qt, truth=y)
        if fixed.fingerprint(x, y) != frozen['processed_data_fingerprint']:
            raise ValueError('Processed dataset fingerprint differs from frozen inventory')
    with np.load(path) as data:
        case = {key: data[key] for key in ('edges', 'degrees', 'truth')}
    np.testing.assert_array_equal(case['truth'], y)
    case.update(dataset=entry['name'], seed=seed, labels=labels_for(case['truth'], seed, p))
    if fixed.fingerprint(case['edges'], case['degrees']) != frozen['graph_fingerprint']:
        raise ValueError('Graph differs from frozen inventory; graph tie order may depend on software version')
    if fixed.fingerprint(case['labels']) != frozen['label_fingerprints'][str(seed)]:
        raise ValueError('Labels differ from frozen inventory')
    return case


def baseline(initial, seed, cap):
    a = copy.deepcopy(initial)
    fixed.seed_numba(seed)
    start = time.perf_counter()
    fixed.pcc_numba._pcc_propagate_numba_sequential(*a[:22], cap, False, a[24], a[25], False)
    elapsed = time.perf_counter() - start
    return elapsed, a, rng_state()


def feasibility():
    require_windows()
    p = protocol()
    frozen = json.loads((ROOT / 'inventory.json').read_text(encoding='utf-8'))
    if frozen['protocol_sha256'] != protocol_hash():
        raise ValueError('Protocol changed after inventory freeze')
    cap, seed = p['feasibility']['iterations'], p['feasibility']['seed']
    measurements, verifications = [], []
    for entry, info in zip(p['datasets'], frozen['datasets']):
        case = case_for(entry, seed, p, info)
        started = time.perf_counter()
        initial = initial_arguments(case, p, cap)
        assert initial[25] == original_threshold(info['nodes'], info['labeled_nodes'], p['k_nn'], p['es_chk'])
        preparation_s = time.perf_counter() - started
        # Warm-up is discarded, including JIT, and never mutates the measured initial state.
        warm_s, _, _ = baseline(initial, seed, p['feasibility']['warmup_iterations'])
        warm_dir = CACHE / 'pilot' / entry['name'] / 'warmup'
        run(initial, case, 101, warm_dir, 'warmup', fixed.seed_numba, resume=False)
        for repeat in range(p['feasibility']['repetitions']):
            directory = CACHE / 'pilot' / entry['name'] / str(repeat)
            args = dict(initial=initial, case=case, cap=cap, directory=directory,
                        run_id=identity(p, case, info['graph_fingerprint']), seed_function=fixed.seed_numba,
                        resume=False, verify_points=(5001, cap))
            if repeat % 2 == 0:
                base_s, base, base_rng = baseline(initial, seed, cap)
                result, observed, checks = run(**args)
            else:
                result, observed, checks = run(**args)
                base_s, base, base_rng = baseline(initial, seed, cap)
            assert all(np.array_equal(observed[i], base[i]) for i in STATE_INDICES)
            assert rng_state() == base_rng
            for point in checks:
                verifications.append(dict(dataset=case['dataset'], repeat=repeat, **point))
            measurements.append(dict(dataset=case['dataset'], seed=seed, repeat=repeat,
                iterations=cap, preparation_s=preparation_s, discarded_first_kernel_s=warm_s,
                unobserved_propagation_s=base_s, instrumented_s=result['elapsed_s'],
                checkpoint_io_s=result['checkpoint_io_s'], instrumented_iterations_s=cap/result['elapsed_s'],
                observer_cost_ratio=result['elapsed_s']/base_s,
                peak_process_memory_bytes=memory_bytes(),
                checks_bytes=(directory / 'checks.csv').stat().st_size,
                samples_bytes=(directory / 'samples.csv').stat().st_size,
                checkpoint_bytes=(directory / 'recovery.pkl').stat().st_size,
                complete_bytes=(directory / 'complete.json').stat().st_size,
                initial_array_bytes=sum(value.nbytes for value in initial if isinstance(value, np.ndarray))))
            # Incremental pilot measurements survive a later case failure.
            fixed.write_csv(ROOT / 'feasibility_runs.csv', measurements)
            fixed.write_csv(ROOT / 'verification.csv', verifications)
            print(f"Pilot {entry['name']} repeat {repeat}: baseline {base_s:.3f}s; observed {result['elapsed_s']:.3f}s", flush=True)
    summaries = []
    for info in frozen['datasets']:
        rows = [v for v in measurements if v['dataset'] == info['name']]
        durations = [v['instrumented_s'] for v in rows]
        summaries.append(dict(dataset=info['name'], instrumented_mean_s=statistics.mean(durations),
            instrumented_sd_s=statistics.stdev(durations),
            unobserved_mean_s=statistics.mean(v['unobserved_propagation_s'] for v in rows),
            projected_10_trajectories_s=statistics.mean(durations)*p['max_iter']/cap*len(p['seeds']),
            projected_check_csv_bytes=statistics.mean(v['checks_bytes'] for v in rows)*p['max_iter']/cap*len(p['seeds']),
            projected_sample_csv_bytes=statistics.mean(v['samples_bytes'] for v in rows)*p['max_iter']/cap*len(p['seeds']),
            checkpoint_bytes=max(v['checkpoint_bytes'] for v in rows),
            graph_s=info['graph_s'], load_download_s=info['load_download_s']))
    fixed.write_csv(ROOT / 'feasibility_summary.csv', summaries)
    estimates = dict(environment=environment(), pilot_cap=cap, full_trajectories_started=0,
        projected_serial_s=sum(v['projected_10_trajectories_s'] for v in summaries),
        projected_checks_bytes=sum(v['projected_check_csv_bytes'] for v in summaries),
        projected_samples_bytes=sum(v['projected_sample_csv_bytes'] for v in summaries),
        measured_peak_process_memory_bytes=max(v['peak_process_memory_bytes'] for v in measurements),
        measured_peak_checkpoint_bytes=max(v['checkpoint_bytes'] for v in measurements),
        retained_200_checkpoint_bytes=sum(v['checkpoint_bytes']*len(p['seeds']) for v in summaries),
        graph_total_s=sum(v['graph_s'] for v in summaries), download_load_total_s=sum(v['load_download_s'] for v in summaries),
        extrapolation='Linear 20000-to-500000 scaling including CSV and recovery writes; only seed10, two repetitions. Future events may increase observer state and disk; allow headroom. Verification replay excluded from timing; budget separately.',
        recovery='Atomic latest local checkpoint, exact particle/buffer/observer/Numba MT state, CSV offsets truncate trailing partial work. Completed case identity checked before reuse. Process interruption tested; power-loss durability and cross-version recovery not guaranteed.')
    (ROOT / 'feasibility.json').write_text(json.dumps(estimates, indent=2), encoding='utf-8')
    plot_feasibility(summaries)


def plot_feasibility(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    y = np.arange(len(rows))
    axes[0].barh(y, [r['projected_10_trajectories_s']/60 for r in rows])
    axes[0].set_yticks(y, [r['dataset'] for r in rows])
    axes[0].set_xlabel('Projected minutes: 10 x 500000, observed')
    ratios = [r['instrumented_mean_s']/r['unobserved_mean_s'] for r in rows]
    axes[1].barh(y, ratios)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel('Observed / unobserved fixed-budget runtime')
    fig.suptitle('Feasibility only: linear extrapolation of 20000-iteration pilots')
    fig.tight_layout()
    fig.savefig(ROOT / 'feasibility.png', dpi=150)
    plt.close(fig)


def full():
    """Call only after explicit feasibility approval; never invoked by --feasibility."""
    require_windows()
    p = protocol()
    frozen = json.loads((ROOT / 'inventory.json').read_text(encoding='utf-8'))
    if frozen['protocol_sha256'] != protocol_hash():
        raise ValueError('Protocol differs from frozen inventory')
    rows = []
    for entry, info in zip(p['datasets'], frozen['datasets']):
        for seed in p['seeds']:
            case = case_for(entry, seed, p, info)
            initial = initial_arguments(case, p, p['max_iter'])
            baseline(initial, seed, 101)  # discarded warm-up
            directory = ROOT / 'trajectories' / entry['name'] / str(seed)
            result, _, checks = run(initial, case, p['max_iter'], directory,
                identity(p, case, info['graph_fingerprint']), fixed.seed_numba,
                spacing=p['measurement_spacing'], recovery_spacing=p['recovery_spacing'],
                references=tuple(p['intermediate_iterations']), verify_points=(30000, 500000))
            for row in result['rows']:
                row.update(nodes=info['nodes'], classes=info['classes'], labeled_nodes=info['labeled_nodes'],
                    labeled_fraction=info['labeled_fraction'], connected_components=info['connected_components'],
                    degree_mean=info['degree_mean'], degree_max=info['degree_max'],
                    measured_full_trajectory_s=result['elapsed_s'])
            rows.extend(result['rows'])
            fixed.write_csv(ROOT / 'individual.csv', rows)
            if checks:
                (directory / 'verification.json').write_text(json.dumps(checks), encoding='utf-8')
            print(f"Completed {entry['name']} seed {seed}", flush=True)
    dataset_rows, overall_rows = summaries(rows)
    fixed.write_csv(ROOT / 'dataset_summary.csv', dataset_rows)
    fixed.write_csv(ROOT / 'overall_equal_dataset_summary.csv', overall_rows)
    strata = dict(binary=[r for r in rows if r['classes'] == 2],
                  multiclass=[r for r in rows if r['classes'] > 2],
                  nodes_lt1000=[r for r in rows if r['nodes'] < 1000],
                  nodes_1000_4999=[r for r in rows if 1000 <= r['nodes'] < 5000],
                  nodes_ge5000=[r for r in rows if r['nodes'] >= 5000],
                  labeled_lt1pct=[r for r in rows if r['labeled_fraction'] < .01],
                  labeled_ge1pct=[r for r in rows if r['labeled_fraction'] >= .01])
    stratum_rows = []
    for name, values in strata.items():
        if values:
            _, grouped = summaries(values)
            stratum_rows.extend(dict(stratum=name, **row) for row in grouped)
    fixed.write_csv(ROOT / 'stratum_equal_dataset_summary.csv', stratum_rows)
    plot_full(dataset_rows)


def plot_full(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for name in dict.fromkeys(r['dataset'] for r in rows):
        values = sorted((r for r in rows if r['dataset'] == name), key=lambda r: r['multiplier'])
        axes[0].plot([r['observation_iteration_mean'] for r in values],
                     [r['accuracy_difference_mean'] for r in values], '.-', label=name)
        axes[1].plot([r['multiplier'] for r in values],
                     [r['observation_iteration_mean'] for r in values], '.-')
    axes[0].set(xlabel='Mean stop/censor observation iteration', ylabel='Accuracy difference vs 500000')
    axes[1].set(xlabel='Patience multiplier', ylabel='Mean stop/censor observation iteration')
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle('Descriptive paired results: censored cases observed at cap, not estimated stop times')
    fig.tight_layout()
    fig.savefig(ROOT / 'paired_tradeoffs.png', dpi=150)
    plt.close(fig)


def recovery_check():
    """Actual process restart on the largest dataset, only a 20000-step pilot."""
    require_windows()
    code = '''
import sys,json,numpy as np
sys.path.insert(0,'docs')
import study_stopping_patience as s
p=s.protocol()
f=json.loads((s.ROOT/'inventory.json').read_text())
entry=next(e for e in p['datasets'] if e['name']=='PenDigits')
info=next(e for e in f['datasets'] if e['name']=='PenDigits')
case=s.case_for(entry,10,p,info)
initial=s.initial_arguments(case,p,20000)
mode=sys.argv[1]
directory=Path(sys.argv[2])/('baseline' if mode=='baseline' else 'resumed')
try:
 result,a,checks=s.run(initial,case,20000,directory,'recovery-real',s.fixed.seed_numba,
  resume=(mode=='resume'),interrupt_at=10000 if mode=='interrupt' else None,verify_points=(5001,20000))
except InterruptedError:
 print('Interrupted at durable checkpoint')
else:
 value=dict(state_sha256=s.fixed.fingerprint(*(a[i] for i in s.STATE_INDICES)),
            rng_sha256=result['rng_sha256'],rows=result['rows'],verification=checks)
 (directory/'comparison.json').write_text(json.dumps(value))
'''
    code = 'from pathlib import Path\n' + code
    directory = CACHE / 'recovery_process' / uuid.uuid4().hex
    for mode in ('baseline', 'interrupt', 'resume'):
        subprocess.run([sys.executable, '-c', code, mode, str(directory)], check=True)
    base = json.loads((directory / 'baseline/comparison.json').read_text())
    resumed = json.loads((directory / 'resumed/comparison.json').read_text())
    assert base == resumed
    csv_equal = {name: (directory / 'baseline' / name).read_bytes() ==
                 (directory / 'resumed' / name).read_bytes() for name in ('checks.csv', 'samples.csv')}
    assert all(csv_equal.values())
    report = dict(dataset='PenDigits', seed=10, cap=20000, interrupted_at=10000,
                  separate_process_restart=True, state_and_rng_equal=True, event_rows_equal=True,
                  csv_equal=csv_equal, verification=resumed['verification'])
    (ROOT / 'recovery_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Separate-process recovery: states, RNG, events and CSVs identical', flush=True)


def download_check():
    """Measure public network retrieval independently of cached inventory loading."""
    frozen = json.loads((ROOT / 'inventory.json').read_text(encoding='utf-8'))
    rows = []
    directory = CACHE / 'download_trial'
    for info in frozen['datasets']:
        if 'openml_id' not in info:
            continue
        ident = info['openml_id']
        meta_path, data_path = directory / f'{ident}.json', directory / f'{ident}.arff'
        if meta_path.exists() or data_path.exists():
            raise ValueError('Download timing requires fresh cache paths; do not report cached reads as downloads')
        start = time.perf_counter()
        retrieve(f'https://www.openml.org/api/v1/json/data/{ident}', meta_path)
        meta_s = time.perf_counter() - start
        start = time.perf_counter()
        raw = retrieve(info['download_url'], data_path)
        file_s = time.perf_counter() - start
        assert sha(raw) == info['raw_sha256']
        rows.append(dict(dataset=info['name'], openml_id=ident, metadata_s=meta_s,
                         file_s=file_s, total_s=meta_s+file_s, bytes=len(raw), sha256=sha(raw)))
        fixed.write_csv(ROOT / 'download_runs.csv', rows)
        print(f"Fresh-cache download {info['name']}: {meta_s+file_s:.2f}s", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--inventory', action='store_true')
    modes.add_argument('--feasibility', action='store_true')
    modes.add_argument('--recovery-check', action='store_true')
    modes.add_argument('--download-check', action='store_true')
    modes.add_argument('--full', action='store_true', help='ONLY after explicit maintainer feasibility approval')
    args = parser.parse_args()
    if args.inventory:
        inventory()
    elif args.feasibility:
        feasibility()
    elif args.recovery_check:
        recovery_check()
    elif args.download_check:
        download_check()
    else:
        full()
