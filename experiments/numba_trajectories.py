"""Fixed-size native snapshot buffers. No object mode, RNG calls or loop replay."""
import time

import numpy as np
from numba import njit

from pcc_numba import _pcc_propagate_numba_sequential
from pcc_metrics import propagation_info
from pcc_validation import validate_step_inputs, validate_buffers
from .native_trajectories import allocate_buffer, validate_buffer, finish_trace


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


def propagate_observed(args, state, collector):
    validation_started = time.perf_counter()
    validate_buffer(args, state, collector)
    max_degree = validate_step_inputs(*args[:3], args[5], args[6], *args[7:13])
    validate_buffers(args[5], max_degree, *args[15:22])
    signatures_before = len(_pcc_propagate_numba_sequential.signatures)
    start = time.perf_counter()
    collector.phase_timings['native_validation_s'] = start - validation_started
    iterations, stalled = _pcc_propagate_numba_sequential(*args, True, record_native, state)
    native_elapsed = time.perf_counter() - start
    info = propagation_info(iterations, args[-3], args[-4], args[-1], stalled)
    postprocessing = finish_trace(state, collector, info, native_elapsed)
    collector.execution['timings'] = dict(native_propagation_s=native_elapsed,
        postprocessing_s=postprocessing,
        compilation_occurred=len(_pcc_propagate_numba_sequential.signatures) != signatures_before)
    return info
