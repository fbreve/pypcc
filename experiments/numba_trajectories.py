"""Fixed-size native snapshot buffers. No object mode, RNG calls or loop replay."""
import time

import numpy as np
from numba import njit

from pcc_numba import _pcc_propagate_numba_sequential
from pcc_metrics import propagation_info
from pcc_validation import validate_step_inputs, validate_buffers


@njit
def seed_numba(seed):
    np.random.seed(seed)


@njit
def record_native(iteration, dominance, positions, strengths, distances, owndeg, final, state):
    schedule, iterations, dom, pos, strength, dist, own, cursor = state
    used, next_milestone = cursor[0], cursor[1]
    if used > 0 and iterations[used - 1] == iteration:
        return
    if not final and (next_milestone >= schedule.size or schedule[next_milestone] != iteration):
        return
    if used >= iterations.size:
        raise ValueError('native observation buffer exhausted')
    iterations[used] = iteration
    dom[used] = dominance
    pos[used] = positions
    strength[used] = strengths
    dist[used] = distances
    own[used] = owndeg
    cursor[0] += 1
    if next_milestone < schedule.size and schedule[next_milestone] == iteration:
        cursor[1] += 1


def allocate_buffer(args, schedule, collector):
    slots, n, c, p = len(schedule), args[11].shape[0], args[5], args[7].size
    required = slots * (16 * n * c + 16 * p + n * p + 16) + 16
    if collector.estimated_bytes(slots, c, required) > collector.max_bytes:
        raise ValueError('native observation storage budget exceeded')
    state = (np.asarray(schedule, dtype=np.int64), np.empty(slots, dtype=np.int64),
             np.empty((slots, n, c)), np.empty((slots, p), dtype=np.int64),
             np.empty((slots, p)), np.empty((slots, n, p), dtype=np.uint8),
             np.empty((slots, n, c)), np.zeros(2, dtype=np.int64))
    assert sum(a.nbytes for a in state) == required
    return state, required


def propagate_observed(args, state, collector):
    max_degree = validate_step_inputs(*args[:3], args[5], args[6], *args[7:13])
    validate_buffers(args[5], max_degree, *args[15:22])
    signatures_before = len(_pcc_propagate_numba_sequential.signatures)
    start = time.perf_counter()
    iterations, stalled = _pcc_propagate_numba_sequential(*args, True, record_native, state)
    native_elapsed = time.perf_counter() - start
    info = propagation_info(iterations, args[-3], args[-4], args[-1], stalled)
    _, observed_iterations, dom, pos, strength, dist, own, cursor = state
    for index in range(cursor[0]):
        final = index == cursor[0] - 1
        collector.capture(int(observed_iterations[index]), dom[index], pos[index], strength[index],
                          dist[index], own[index], info if final else None, selected=True,
                          native=True, elapsed=native_elapsed if final else None)
    collector.execution['timings'] = dict(native_propagation_s=native_elapsed,
        postprocessing_s=time.perf_counter() - start - native_elapsed,
        compilation_occurred=len(_pcc_propagate_numba_sequential.signatures) != signatures_before)
    return info
