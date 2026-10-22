"""Shared bounded native buffers; no backend or RNG imports at module load."""
import time

import numpy as np


def validate_live_shapes(args):
    n, c, p = args[0].shape[0], args[5], args[7].size
    if args[11].shape != (n, c) or args[12].shape != (n, c) or args[10].shape != (n, p):
        raise ValueError('native diagnostics require exact live state dimensions')
    return n, c, p


def validate_schedule(schedule, cap, limit):
    values = list(schedule)
    if (not values or len(values) > limit or any(type(i) is not int or i < 0 or i > cap for i in values)
            or values != sorted(set(values)) or values[-1] != cap):
        raise ValueError('native schedule must be increasing, within cap, and reserve final capacity')
    return values


def allocate_buffer(args, schedule, collector, *, cython=False):
    schedule = validate_schedule(schedule, args[-4], collector.max_observations)
    n, c, p = validate_live_shapes(args)
    slots = len(schedule)
    required = slots * (16 * n * c + 16 * p + n * p + 16 + (4 if cython else 0)) + 16
    if collector.estimated_bytes(slots, c, required) > collector.max_bytes:
        raise ValueError('native observation storage budget exceeded')
    state = (np.asarray(schedule, dtype=np.int64), np.empty(slots, dtype=np.int64),
             np.empty((slots, n, c)), np.empty((slots, p), dtype=np.int64),
             np.empty((slots, p)), np.empty((slots, n, p), dtype=np.uint8),
             np.empty((slots, n, c)), np.zeros(2, dtype=np.int64))
    if cython:
        state += (np.empty(slots, dtype=np.uint32),)
    assert sum(a.nbytes for a in state) == required
    return state, required


def validate_buffer(args, state, collector, *, cython=False):
    if not isinstance(state, tuple) or len(state) != (9 if cython else 8):
        raise ValueError('invalid native buffer tuple')
    schedule = state[0]
    if not isinstance(schedule, np.ndarray) or schedule.ndim != 1 or schedule.dtype != np.int64:
        raise ValueError('invalid native schedule array')
    validate_schedule(schedule.tolist(), args[-4], collector.max_observations)
    n, c, p = validate_live_shapes(args)
    slots = len(schedule)
    shapes = [(slots,), (slots,), (slots, n, c), (slots, p), (slots, p),
              (slots, n, p), (slots, n, c), (2,)]
    dtypes = [np.int64, np.int64, np.float64, np.int64, np.float64, np.uint8, np.float64, np.int64]
    if cython:
        shapes.append((slots,))
        dtypes.append(np.uint32)
    for value, shape, dtype in zip(state, shapes, dtypes):
        if (not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype
                or not value.flags.writeable or not value.flags.c_contiguous):
            raise ValueError('native buffer shape/dtype/layout mismatch')
    if np.any(state[7] != 0) or collector.rows or collector.finished:
        raise ValueError('native buffers and collector must be fresh')
    required = sum(a.nbytes for a in state)
    if collector.estimated_bytes(slots, c, required) > collector.max_bytes:
        raise ValueError('native observation storage budget exceeded')


def finish_trace(state, collector, info, native_elapsed, *, cython=False):
    started = time.perf_counter()
    _, iterations, dom, pos, strength, dist, own, cursor = state[:8]
    for index in range(cursor[0]):
        final = index == cursor[0] - 1
        collector.capture(int(iterations[index]), dom[index], pos[index], strength[index],
                          dist[index], own[index], info if final else None, selected=True,
                          native=True, elapsed=native_elapsed if final else None,
                          cython_rng=state[8][index:index + 1] if cython else None)
    return time.perf_counter() - started


def propagate_cython(args, state, collector):
    validation_started = time.perf_counter()
    import pcc_step
    validate_buffer(args, state, collector, cython=True)
    native_buffer = pcc_step._TrajectoryBuffer(state)
    started = time.perf_counter()
    collector.phase_timings['native_validation_s'] = started - validation_started
    info = pcc_step._pcc_propagate_impl(*args, return_info=True, _observations=native_buffer)
    native_elapsed = time.perf_counter() - started
    postprocessing = finish_trace(state, collector, info, native_elapsed, cython=True)
    collector.execution['timings'] = dict(native_propagation_s=native_elapsed,
        postprocessing_s=postprocessing, compilation_occurred=False)
    return info
