"""Stage A: opt-in NumPy trajectory diagnostics; no historical campaign runner.

Use a fresh collector for one propagation call. Ground truth is used only here,
after iteration boundaries, never by the propagation or stopping calculations.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


def observe_fit(model, labels, *, truth=None, milestones=(0, 10000),
                include_checks=True, **fit_parameters):
    """Observe one fit using instance-local dispatch; reject native fallback.

    Truth is in public label space here. Returns predictions, collector, metrics.
    This research helper leaves the public model API and its defaults unchanged.
    """
    from pcc_numpy import pcc_propagate_numpy
    if model.impl != 'numpy' or model.update_mode != 'sequential':
        raise ValueError('Stage A instrumentation supports explicit sequential NumPy only')
    if np.unique(np.asarray(labels)[np.asarray(labels) != -1]).size < 2:
        raise ValueError('trajectory requires at least two labeled classes')
    if model._get_backend_fn() is not pcc_propagate_numpy:
        raise ValueError('unexpected backend dispatch')
    result = {}

    def observed(*args, **kwargs):
        mapped_truth = None
        if truth is not None:
            mapped_truth = np.full(np.asarray(truth).shape, -2, dtype=np.int64)
            for label, index in model.label_to_idx.items():
                mapped_truth[np.asarray(truth) == label] = index
        collector = Trajectory(args[2], truth=mapped_truth, milestones=milestones,
                               include_checks=include_checks)
        result['collector'] = collector
        result['metrics'] = pcc_propagate_numpy(*args, **kwargs, observer=collector, return_info=True)

    original = model.__dict__.get('_get_backend_fn')
    model._get_backend_fn = lambda: observed
    try:
        predictions = model.fit_predict(labels, **fit_parameters)
    finally:
        if original is None:
            del model._get_backend_fn
        else:
            model._get_backend_fn = original
    return predictions, result['collector'], result['metrics']


def array_hash(value):
    value = np.asarray(value)
    canonical = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder('<'))
    digest = hashlib.sha256()
    digest.update(json.dumps([list(canonical.shape), canonical.dtype.str]).encode())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


class Trajectory:
    """Read-only collector for pcc_propagate_numpy(observer=collector).

    Iterations count completed steps. Check boundaries are 1, 11, 21, ...,
    because production checks zero-based loop indices divisible by ten.
    Changes/stability compare observations, not every intervening iteration.
    Labels and truth must use the same internal class indices as dominance.
    """

    def __init__(self, labels, *, truth=None, milestones=(0, 10000),
                 include_checks=True):
        schedule = list(milestones)
        if any(type(i) is not int or i < 0 for i in schedule):
            raise ValueError('milestones must be nonnegative integers')
        if schedule != sorted(set(schedule)):
            raise ValueError('milestones must be strictly increasing')
        self.milestones = schedule
        self.include_checks = bool(include_checks)
        self.labels = np.array(labels, copy=True)
        self.truth = None if truth is None else np.array(truth, copy=True)
        if self.labels.ndim != 1 or (self.truth is not None and self.truth.shape != self.labels.shape):
            raise ValueError('labels/truth must be matching vectors')
        self.rows = []
        self.previous = None
        self.previous_assignments = None
        self.started = time.perf_counter()
        self.finished = False

    def __call__(self, iteration, dominance, positions, strengths, distances, owndeg, info):
        if self.finished:
            raise ValueError('collector is single-use')
        if info is None and iteration not in self.milestones and not (
                self.include_checks and iteration > 0 and (iteration - 1) % 10 == 0):
            return
        if dominance.shape[0] != self.labels.size or dominance.shape[1] < 2:
            raise ValueError('trajectory requires matching nodes and at least two classes')
        if info is not None and self.rows and self.rows[-1]['iteration'] == iteration:
            self.rows[-1].update(final=True, stop_reason=info['stop_reason'])
            self.finished = True
            return
        assigned = np.argmax(dominance, axis=1)
        known = self.labels != -1
        assigned[known] = self.labels[known]
        unknown = ~known
        top = np.sort(dominance, axis=1)[:, -2:]
        gap = top[:, 1] - top[:, 0]
        hashes = {name: array_hash(value) for name, value in dict(
            assignments=assigned, dominance=dominance, positions=positions,
            strengths=strengths, distances=distances, owndeg=owndeg).items()}
        rng = np.random.get_state()
        hashes['numpy_rng'] = hashlib.sha256(json.dumps(
            [rng[0], rng[1].tolist(), *rng[2:]], allow_nan=False).encode()).hexdigest()
        row = dict(iteration=int(iteration), assignments=assigned.tolist(),
                   class_changes=None if self.previous is None else int(np.count_nonzero(
                       assigned != self.previous_assignments)),
                   dominance_mean_abs_change=None if self.previous is None else float(
                       np.abs(dominance - self.previous).mean()),
                   dominance_max_abs_change=None if self.previous is None else float(
                       np.abs(dominance - self.previous).max()),
                   margin_mean=float(gap.mean()), margin_min=float(gap.min()),
                   accuracy_unlabeled_retrospective=None if self.truth is None or not unknown.any()
                   else float(np.mean(assigned[unknown] == self.truth[unknown])),
                   state_hashes=hashes, elapsed_instrumented_s=time.perf_counter() - self.started,
                   final=info is not None, stop_reason=None if info is None else info['stop_reason'])
        self.rows.append(row)
        self.previous = dominance.copy()
        self.previous_assignments = assigned.copy()
        self.finished = info is not None


def metadata(*, dataset, label_seed, propagation_seed, arrays, parameters):
    """Capture provenance without seeding, fitting, downloading or loading data."""
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
    return dict(protocol='convergence-trajectories-stage-a-v1', dataset=dataset,
                protocol_sha256=hashlib.sha256((root / 'docs/convergence_trajectories_protocol.md').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                label_seed=label_seed, backend='numpy', update_mode='sequential',
                source_commit=commit, source_dirty=dirty, python=platform.python_version(),
                platform=platform.platform(),
                dependencies={name: importlib.metadata.version(name) for name in
                              ('numpy', 'scipy', 'scikit-learn')},
                fixture_hashes={name: array_hash(value) for name, value in arrays.items()},
                parameters=dict(parameters), propagation_seed=propagation_seed,
                seed_policy='caller seeds NumPy once before a single propagation call',
                evaluation='retrospective transductive diagnostic only',
                timing='instrumented elapsed time; not a kernel benchmark')


def write_output(directory, collector, provenance):
    """Write a new bundle; manifest written last marks successful completion."""
    if not collector.finished:
        raise ValueError('cannot export incomplete trajectory')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = dict(metadata=dict(provenance, milestones=collector.milestones,
                                include_checks=collector.include_checks), observations=collector.rows)
    (directory / 'trajectory.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    context = {key: provenance[key] for key in ('dataset', 'label_seed', 'backend')}
    with (directory / 'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*context, *collector.rows[0]])
        writer.writeheader()
        for row in collector.rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict))
                             else value for key, value in dict(context, **row).items()})
    manifest = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ('trajectory.json', 'trajectory.csv')}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def read_output(directory):
    """Reject partial or corrupted bundles before returning JSON observations."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest) != {'trajectory.json', 'trajectory.csv'}:
        raise ValueError('invalid trajectory manifest')
    for name, expected in manifest.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('trajectory checksum mismatch: ' + name)
    return json.loads((directory / 'trajectory.json').read_text(encoding='utf-8'))
