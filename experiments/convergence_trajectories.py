"""Bounded single-fit NumPy/Numba diagnostics, separate from benchmark timing."""
import hashlib
import inspect
import json
import time

import numpy as np

DEFAULT_MAX_OBSERVATIONS = 128
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def array_hash(value):
    value = np.asarray(value)
    canonical = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder('<'))
    digest = hashlib.sha256()
    digest.update(json.dumps([list(canonical.shape), canonical.dtype.str]).encode())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def numpy_rng_hash(state=None):
    state = np.random.get_state() if state is None else state
    return hashlib.sha256(json.dumps(
        [state[0], state[1].tolist(), *state[2:]], allow_nan=False).encode()).hexdigest()


def effective_parameters(parameters):
    from pcc import ParticleCompetitionAndCooperation as PCC
    defaults = {k: p.default for k, p in inspect.signature(PCC.fit_predict).parameters.items()
                if k not in ('self', 'labels')}
    if set(parameters) - set(defaults):
        raise ValueError('unknown execution parameters')
    defaults.update(parameters)
    for key in defaults:
        convert = bool if key == 'early_stop' else int if key in ('max_iter', 'es_chk') else float
        defaults[key] = convert(defaults[key])
    if defaults['max_iter'] < 0 or not all(np.isfinite(v) for v in defaults.values()):
        raise ValueError('invalid execution parameters')
    return defaults


def schedule_for(cap, milestones, include_checks, limit):
    if include_checks and (cap + 9) // 10 > limit:
        raise ValueError('stopping-check sampling exceeds max_observations')
    schedule = {i for i in milestones if i <= cap} | {cap}
    if include_checks:
        schedule.update(range(1, cap + 1, 10))
    if len(schedule) > limit:
        raise ValueError('milestones exceed max_observations')
    return sorted(schedule)


class Trajectory:
    """Trusted NumPy observer; defaults retain milestones and final only.

    Check sampling is explicit. Row count and estimated storage are bounded.
    Changes compare adjacent observations, not every intervening iteration.
    """
    def __init__(self, labels, *, truth=None, milestones=(0, 10000), include_checks=False,
                 max_observations=DEFAULT_MAX_OBSERVATIONS, max_bytes=DEFAULT_MAX_BYTES):
        schedule = list(milestones)
        if any(type(i) is not int or i < 0 or i > np.iinfo(np.int64).max for i in schedule):
            raise ValueError('milestones must be nonnegative int64 Python integers')
        if schedule != sorted(set(schedule)):
            raise ValueError('milestones must be strictly increasing')
        if type(max_observations) is not int or max_observations < 1:
            raise ValueError('max_observations must be positive')
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('max_bytes must be positive')
        if type(include_checks) is not bool:
            raise ValueError('include_checks must be boolean')
        self.milestones, self.include_checks = schedule, include_checks
        self.max_observations, self.max_bytes = max_observations, max_bytes
        self.labels = np.array(labels, dtype=np.int64, copy=True)
        self.truth = None if truth is None else np.array(truth, copy=True)
        if self.labels.ndim != 1 or (self.truth is not None and self.truth.shape != self.labels.shape):
            raise ValueError('labels/truth must be matching vectors')
        self.rows, self.previous, self.previous_assignments = [], None, None
        self.started, self.finished = time.perf_counter(), False
        self.execution, self.metrics = None, None
        self.phase_timings = {}
        self.snapshots = None  # strictly opt-in; legacy rows/metadata stay unchanged

    def estimated_bytes(self, rows, classes, native_bytes=0):
        # Conservative Python list/dict + scratch accounting, not a process RSS limit.
        n = self.labels.size
        return int(native_bytes + rows * (128 * n + 16384) + 128 * n * classes)

    def __call__(self, iteration, dominance, positions, strengths, distances, owndeg, info):
        self.capture(iteration, dominance, positions, strengths, distances, owndeg, info)

    def capture(self, iteration, dominance, positions, strengths, distances, owndeg, info,
                *, selected=False, native=False, elapsed=None, cython_rng=None):
        if self.finished:
            raise ValueError('collector is single-use')
        if not selected and info is None and iteration not in self.milestones and not (
                self.include_checks and iteration > 0 and (iteration - 1) % 10 == 0):
            return
        if dominance.shape[0] != self.labels.size or dominance.shape[1] < 2:
            raise ValueError('trajectory requires matching nodes and at least two classes')
        if info is not None and self.rows and self.rows[-1]['iteration'] == iteration:
            self.rows[-1].update(final=True, stop_reason=info['stop_reason'])
            self.finished, self.metrics = True, dict(info)
            return
        if self.rows and iteration <= self.rows[-1]['iteration']:
            raise ValueError('observations must be strictly increasing')
        if len(self.rows) >= self.max_observations or self.estimated_bytes(
                len(self.rows) + 1, dominance.shape[1]) > self.max_bytes:
            raise ValueError('observation storage budget exceeded')
        assigned = np.argmax(dominance, axis=1)
        known = self.labels != -1
        assigned[known] = self.labels[known]
        unknown = ~known
        top = np.sort(dominance, axis=1)[:, -2:]
        gap = top[:, 1] - top[:, 0]
        hashes = {name: array_hash(value) for name, value in dict(
            assignments=assigned, dominance=dominance, positions=positions,
            strengths=strengths, distances=distances, owndeg=owndeg).items()}
        if not native:
            hashes['numpy_rng'] = numpy_rng_hash()
        if cython_rng is not None:
            hashes['cython_rng'] = array_hash(cython_rng)
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
                   state_hashes=hashes, elapsed_instrumented_s=(elapsed if native else time.perf_counter() - self.started),
                   final=info is not None, stop_reason=None if info is None else info['stop_reason'])
        self.rows.append(row)
        if self.snapshots is not None:
            self.snapshots.capture(iteration, dominance, positions, strengths, native=native)
        self.previous, self.previous_assignments = dominance.copy(), assigned.copy()
        self.finished = info is not None
        if info is not None:
            self.metrics = dict(info)


def observe_fit(model, labels, *, truth=None, milestones=(0, 10000), include_checks=False,
                propagation_seed=None, max_observations=DEFAULT_MAX_OBSERVATIONS,
                max_bytes=DEFAULT_MAX_BYTES, snapshot_config=None, **fit_parameters):
    """Observe one fit. An explicit seed seeds only the selected backend, once.

    With no seed, preserve caller RNG state and record an unknown seed.
    The original NumPy return tuple and instance-local dispatch are preserved.
    """
    overall_start = time.perf_counter()
    from pcc_numpy import pcc_propagate_numpy
    if model.impl not in ('numpy', 'numba', 'cython') or model.update_mode != 'sequential':
        raise ValueError('diagnostics require an explicit sequential backend')
    backend = pcc_propagate_numpy
    if model.impl == 'numba':
        from pcc_numba import pcc_propagate_numba
        backend = pcc_propagate_numba
    elif model.impl == 'cython':
        from pcc_step import pcc_propagate
        backend = pcc_propagate
    if model._get_backend_fn() is not backend:
        raise ValueError('unexpected backend dispatch')
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels[labels != -1]).size < 2:
        raise ValueError('trajectory requires at least two labeled classes')
    if model.neib_list is None or labels.shape != (model.neib_list.shape[0],):
        raise ValueError('set a matching graph before observing')
    truth = None if truth is None else np.asarray(truth, dtype=np.int64)
    if truth is not None and truth.shape != labels.shape:
        raise ValueError('truth must match labels')
    if propagation_seed is not None and (type(propagation_seed) is not int or not 0 <= propagation_seed < 2**32):
        raise ValueError('propagation_seed must be an unsigned 32-bit integer or None')
    if model.impl == 'cython' and propagation_seed not in (None, 123456789):
        raise ValueError('Cython always starts from its fixed seed 123456789')
    if model.impl == 'cython':
        propagation_seed = 123456789
    parameters = effective_parameters(fit_parameters)
    probe = Trajectory(labels, truth=truth, milestones=milestones, include_checks=include_checks,
                       max_observations=max_observations, max_bytes=max_bytes)
    schedule = schedule_for(parameters['max_iter'], probe.milestones, include_checks, max_observations)
    if snapshot_config is not None:
        from .mechanistic_snapshots import SnapshotConfig, estimates, check_budget
        if not isinstance(snapshot_config, SnapshotConfig):
            raise TypeError('snapshot_config must be SnapshotConfig or None')
        check_budget(snapshot_config, estimates(labels.size, np.unique(labels[labels != -1]).size,
                     int(np.count_nonzero(labels != -1)), len(schedule), model.impl))
    from .trajectory_io import source_metadata
    source_start = time.perf_counter()
    profile = dict(preflight_s=source_start - overall_start)
    source_profile = {}
    source = source_metadata(profile=source_profile)
    profile['source_metadata_s'] = time.perf_counter() - source_start
    result = {}

    def observed(*args, **kwargs):
        setup_start = time.perf_counter()
        profile['model_initialization_s'] = setup_start - fit_start
        actual_parameters = effective_parameters(dict(p_grd=args[3], delta_v=args[4],
            deltap=args[13], dexp=args[14], max_iter=args[-4], early_stop=args[-3], es_chk=args[-2]))
        if actual_parameters != parameters:
            raise ValueError('effective backend parameters differ from requested fit')
        mapped_truth = None
        if truth is not None:
            mapped_truth = np.full(truth.shape, -2, dtype=np.int64)
            for label, index in model.label_to_idx.items():
                mapped_truth[truth == label] = index
        collector = Trajectory(args[2], truth=mapped_truth, milestones=probe.milestones,
                               include_checks=include_checks, max_observations=max_observations,
                               max_bytes=max_bytes)
        if snapshot_config is not None:
            from .mechanistic_snapshots import Snapshots
            collector.snapshots = Snapshots(snapshot_config, args[2], model.part.homenode,
                                           model.part.label, schedule, model.impl)
        native_bytes, native_buffer = 0, None
        if model.impl == 'numba':
            from .numba_trajectories import allocate_buffer, propagate_observed, seed_numba
            native_buffer, native_bytes = allocate_buffer(args, schedule, collector)
        elif model.impl == 'cython':
            from .native_trajectories import allocate_buffer, propagate_cython
            native_buffer, native_bytes = allocate_buffer(args, schedule, collector, cython=True)
        elif collector.estimated_bytes(len(schedule), model.c) > max_bytes:
            raise ValueError('observation storage budget exceeded')
        if collector.snapshots is not None and native_buffer is not None:
            collector.snapshots.bind_native(native_buffer)
        if propagation_seed is not None:
            if model.impl == 'numba':
                seed_numba(propagation_seed)
            elif model.impl == 'numpy':
                np.random.seed(propagation_seed)
        arrays = dict(neighbors=model.neib_list, degrees=model.neib_qt, labels=model.labels)
        if truth is not None:
            arrays['truth'] = truth
        collector.execution = dict(source, backend=model.impl, parameters=actual_parameters,
            fixture_hashes={k: array_hash(v) for k, v in arrays.items()},
            nodes=model.node.amount, classes=model.c, particles=model.part.amount,
            graph_has_edges=bool(np.any(args[1])), class_index_to_label=model.unique_labels.tolist(),
            k_nn=int(model.k_nn), stop_max=int(args[-1]), truth_present=truth is not None,
            propagation_seed=propagation_seed,
            seed_policy=('cython-fixed-seed' if model.impl == 'cython' else
                         model.impl + ('-seed-once' if propagation_seed is not None else '-caller-state')),
            initial_numpy_rng=numpy_rng_hash() if model.impl == 'numpy' else None,
            rng_fingerprint={'numpy': 'numpy-state', 'numba': 'not-captured', 'cython': 'xorshift32-state'}[model.impl],
            budget=dict(max_observations=max_observations, max_bytes=max_bytes,
                        scheduled_observations=len(schedule), native_buffer_bytes=native_bytes,
                        estimated_peak_bytes=collector.estimated_bytes(len(schedule), model.c, native_bytes)))
        collector.started = time.perf_counter()
        profile['diagnostic_setup_s'] = collector.started - setup_start
        if model.impl == 'numba':
            metrics = propagate_observed(args, native_buffer, collector)
        elif model.impl == 'cython':
            metrics = propagate_cython(args, native_buffer, collector)
        else:
            metrics = backend(*args, **kwargs, observer=collector, return_info=True)
        result.update(collector=collector, metrics=metrics, propagation_end=time.perf_counter())

    original = model.__dict__.get('_get_backend_fn')
    model._get_backend_fn = lambda: observed
    try:
        fit_start = time.perf_counter()
        predictions = model.fit_predict(labels, **parameters)
    finally:
        if original is None:
            del model._get_backend_fn
        else:
            model._get_backend_fn = original
    ended = time.perf_counter()
    collector = result['collector']
    profile.update(collector.phase_timings)
    profile['model_finalization_s'] = ended - result['propagation_end']
    if model.impl in ('numba', 'cython'):
        timings = collector.execution['timings']
        profile['native_propagation_s'] = timings['native_propagation_s']
        profile['snapshot_postprocessing_s'] = timings['postprocessing_s']
    profile['unattributed_s'] = ended - overall_start - sum(profile.values())
    profile['complete_fit_s'] = ended - overall_start
    collector.phase_timings = dict(phases=profile, source_components=source_profile)
    return predictions, collector, result['metrics']


from .trajectory_io import metadata, write_output, read_output  # noqa: E402
