"""Optional bounded synthetic overhead probe, never an ordinary benchmark result.

Run: python -m experiments.trajectory_overhead --output .cache/trajectory-overhead.json
JIT warm-ups are excluded. No historical data or existing benchmark files used.
"""
import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from numba import njit

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numba import pcc_propagate_numba
from .convergence_trajectories import observe_fit, array_hash, metadata, write_output, read_output
from .numba_trajectories import seed_numba
from .trajectory_io import source_metadata

@njit
def continuation():
    return np.random.random(10)


def run(mode, seed, backend='numba', *, allow_compilation=False):
    nodes, cap = 64, 20000
    edges = np.array([[(i + offset) % nodes for offset in (-1, 1, 3, 11)] for i in range(nodes)])
    labels = np.full(nodes, -1, dtype=np.int64)
    labels[[0, 1, 21, 22, 42, 43]] = [0, 0, 1, 1, 2, 2]
    truth = np.arange(nodes) * 3 // nodes
    model = PCC(impl=backend)
    model.set_graph(edges, np.full(nodes, 4))
    if backend == 'numba':
        seed_numba(seed)
        propagate = pcc_propagate_numba
    else:
        import pcc_step
        propagate = pcc_step.pcc_propagate
    measured = {}
    trace = None
    if mode == 'plain':
        def ordinary(*args):
            entered = time.perf_counter()
            measured['metrics'] = propagate(*args, return_info=True)
            measured['native_s'] = time.perf_counter() - entered
        model._get_backend_fn = lambda: ordinary
    start = time.perf_counter()
    if mode == 'plain':
        predictions = model.fit_predict(labels, max_iter=cap, early_stop=False)
    else:
        milestones = (0, 10000) if mode == 'sparse' else tuple(range(0, cap + 1, 200))
        predictions, trace, measured['metrics'] = observe_fit(model, labels, truth=truth,
            propagation_seed=seed if backend == 'numba' else 123456789,
            milestones=milestones, max_iter=cap, early_stop=False)
        measured['native_s'] = trace.execution['timings']['native_propagation_s']
        measured['postprocessing_s'] = trace.execution['timings']['postprocessing_s']
        measured['phase_timings'] = trace.phase_timings
        if trace.execution['timings']['compilation_occurred'] and not allow_compilation:
            raise AssertionError('JIT compilation occurred inside measured probe')
    measured['fit_s'] = time.perf_counter() - start
    state = [array_hash(a) for a in (predictions, model.node.dominance, model.part.curnode,
                                   model.part.strength, model.part.dist_table, model.owndeg)]
    # Cython has no persistent RNG between fits. Its dedicated parity suite uses
    # the private terminal-state/copy probe; do not add probe draws to timings.
    evidence = (state, measured.pop('metrics'), continuation().tolist() if backend == 'numba' else None)
    if trace is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace'
            provenance = metadata(dataset='synthetic-ring64', label_seed=0, collector=trace)
            entered = time.perf_counter()
            write_output(path, trace, provenance)
            read_output(path)
            measured['export_validate_s'] = time.perf_counter() - entered
            measured['bundle_bytes'] = sum(p.stat().st_size for p in path.iterdir())
        measured.update(observations=len(trace.rows), budget=trace.execution['budget'])
    return measured, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=('numba', 'cython'), default=['numba', 'cython'])
    args = parser.parse_args(argv)
    modes = ('plain', 'sparse', 'dense')
    warm_start = time.perf_counter()
    for backend in args.backends:
        for mode in modes:
            run(mode, 19, backend, allow_compilation=True)
    warmup_s = time.perf_counter() - warm_start
    samples = {backend: [] for backend in args.backends}
    for repeat in range(5):
        for backend in args.backends if repeat % 2 == 0 else args.backends[::-1]:
            results = {}
            order = modes if repeat % 2 == 0 else modes[::-1]
            evidence = []
            for mode in order:
                results[mode], state = run(mode, 19 + repeat, backend)
                evidence.append(state)
            if not all(e == evidence[0] for e in evidence):
                raise AssertionError('overhead probe changed state, stopping or subsequent RNG')
            samples[backend].append(results)
    summary = {}
    for backend, runs in samples.items():
        summary[backend] = {}
        for mode in modes:
            keys = [key for key in runs[0][mode] if key.endswith('_s')]
            summary[backend][mode] = {key: statistics.median(s[mode][key] for s in runs) for key in keys}
            if mode != 'plain':
                for group in ('phases', 'source_components'):
                    summary[backend][mode][group] = {key: statistics.median(
                        s[mode]['phase_timings'][group][key] for s in runs)
                        for key in runs[0][mode]['phase_timings'][group]}
                summary[backend][mode]['native_ratio'] = summary[backend][mode]['native_s'] / summary[backend]['plain']['native_s']
                summary[backend][mode]['fit_ratio'] = summary[backend][mode]['fit_s'] / summary[backend]['plain']['fit_s']
    payload = dict(source_metadata(), purpose='synthetic instrumentation overhead only',
        nodes=64, particles=6, classes=3, degree=4, max_iter=20000, early_stop=False,
        repetitions=5, warmup_excluded_s=warmup_s,
        propagation_seeds={backend: list(range(19, 24)) if backend == 'numba' else [123456789] * 5
                           for backend in args.backends},
        probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        parity='predictions, state and stopping metrics match; Numba next 10 draws match; Cython RNG parity tested separately',
        summary_medians=summary, samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
