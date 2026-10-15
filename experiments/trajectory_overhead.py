"""Optional bounded synthetic overhead probe, never an ordinary benchmark result.

Run: python -m experiments.trajectory_overhead --output .cache/trajectory-overhead.json
JIT warm-ups are excluded. No historical data or existing benchmark files used.
"""
import argparse
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


def run(mode, seed):
    nodes, cap = 64, 20000
    edges = np.array([[(i + offset) % nodes for offset in (-1, 1, 3, 11)] for i in range(nodes)])
    labels = np.full(nodes, -1, dtype=np.int64)
    labels[[0, 1, 21, 22, 42, 43]] = [0, 0, 1, 1, 2, 2]
    truth = np.arange(nodes) * 3 // nodes
    model = PCC(impl='numba')
    model.set_graph(edges, np.full(nodes, 4))
    seed_numba(seed)
    measured = {}
    trace = None
    if mode == 'plain':
        def ordinary(*args):
            entered = time.perf_counter()
            measured['metrics'] = pcc_propagate_numba(*args, return_info=True)
            measured['native_s'] = time.perf_counter() - entered
        model._get_backend_fn = lambda: ordinary
    start = time.perf_counter()
    if mode == 'plain':
        predictions = model.fit_predict(labels, max_iter=cap, early_stop=False)
    else:
        milestones = (0, 10000) if mode == 'sparse' else tuple(range(0, cap + 1, 200))
        predictions, trace, measured['metrics'] = observe_fit(model, labels, truth=truth,
            propagation_seed=seed, milestones=milestones, max_iter=cap, early_stop=False)
        measured['native_s'] = trace.execution['timings']['native_propagation_s']
        measured['postprocessing_s'] = trace.execution['timings']['postprocessing_s']
    measured['fit_s'] = time.perf_counter() - start
    state = [array_hash(a) for a in (predictions, model.node.dominance, model.part.curnode,
                                   model.part.strength, model.part.dist_table, model.owndeg)]
    evidence = (state, measured.pop('metrics'), continuation().tolist())
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
    args = parser.parse_args(argv)
    modes = ('plain', 'sparse', 'dense')
    warm_start = time.perf_counter()
    for mode in modes:
        run(mode, 19)
    warmup_s = time.perf_counter() - warm_start
    samples = []
    for repeat in range(5):
        results = {}
        order = modes if repeat % 2 == 0 else modes[::-1]
        evidence = []
        for mode in order:
            results[mode], state = run(mode, 19 + repeat)
            evidence.append(state)
        if not all(e == evidence[0] for e in evidence):
            raise AssertionError('overhead probe changed state, stopping or subsequent RNG')
        samples.append(results)
    summary = {}
    for mode in modes:
        keys = [key for key in samples[0][mode] if key.endswith('_s')]
        summary[mode] = {key: statistics.median(s[mode][key] for s in samples) for key in keys}
    for mode in ('sparse', 'dense'):
        summary[mode]['native_ratio'] = summary[mode]['native_s'] / summary['plain']['native_s']
        summary[mode]['fit_ratio'] = summary[mode]['fit_s'] / summary['plain']['fit_s']
    payload = dict(source_metadata(), purpose='synthetic instrumentation overhead only',
        nodes=64, particles=6, classes=3, degree=4, max_iter=20000, early_stop=False,
        repetitions=5, warmup_excluded_s=warmup_s, seeds=list(range(19, 24)),
        parity='predictions, state, stopping metrics and next 10 native draws match',
        summary_medians=summary, samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
