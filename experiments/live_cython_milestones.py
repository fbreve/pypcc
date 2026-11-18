"""Research-only bounded live observations from an isolated native Cython clone.

Requires first building `pcc_step_live_research` via
`python -m experiments.build_live_cython`. The original pcc_step module,
project packaging and default backends are never modified.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from pcc_validation import validate_seed, validate_propagation_controls


@dataclass(frozen=True)
class Milestone:
    iteration: int
    arrays: object
    rng_state_uint32: int


@dataclass(frozen=True)
class ObservedResult:
    info: object
    milestones: tuple
    observed_iterations: tuple
    bytes_copied: int
    final_rng_state_uint32: int
    next_rng_uint32: int
    cached_rowmax_enabled: bool


def _plan(milestones, max_iter, max_observations, max_bytes, arrays):
    if type(max_iter) is not int or max_iter < 0:
        raise ValueError("max_iter must be nonnegative int")
    if type(max_observations) is not int or not 1 <= max_observations <= 32:
        raise ValueError("max_observations must be integer in 1..32")
    if type(max_bytes) is not int or not 0 < max_bytes <= 64 * 1024 * 1024:
        raise ValueError("max_bytes must be integer in 1..67108864")
    if type(milestones) not in (tuple, list) or not milestones:
        raise ValueError("milestones must be a nonempty tuple/list")
    if len(milestones) > max_observations:
        raise ValueError("too many observation milestones")
    if any(type(x) is not int or x < 0 or x > max_iter for x in milestones):
        raise ValueError("milestones must be integers in 0..max_iter")
    if tuple(milestones) != tuple(sorted(set(milestones))):
        raise ValueError("milestones must be strictly increasing and unique")
    size = sum(a.nbytes for a in arrays.values()) * len(milestones)
    size += 4 * len(milestones)  # Native RNG uint32 stored per milestone.
    if size > max_bytes:
        raise ValueError("planned snapshots exceed max_bytes")
    return tuple(milestones), size


def xorshift32_next(state):
    """Compute the NEXT native xorshift32 unsigned output without changing state."""
    x = int(state) & 0xffffffff
    x ^= (x << 13) & 0xffffffff
    x ^= x >> 17
    x ^= (x << 5) & 0xffffffff
    return x & 0xffffffff


def _allocate(states, schedule):
    n = len(schedule)
    return {
        name: np.empty((n,) + arr.shape, dtype=arr.dtype)
        for name, arr in states.items()
    }


def _record_buffers(states, schedule):
    out = _allocate(states, schedule)
    out["schedule"] = np.asarray(schedule, dtype=np.int64)
    out["rng"] = np.empty(len(schedule), dtype=np.uint32)
    out["final_rng"] = np.zeros(1, dtype=np.uint32)
    out["recorded"] = np.zeros(1, dtype=np.int64)
    return out


def observe_cython_sequential(
    neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
    part_curnode, part_label, part_strength, dist_table, dominance, owndeg,
    deltap, dexp, dom_row, reduc, dom_list, dist_list, prob, slices,
    dist_weights, max_iter, early_stop, es_chk, stop_max,
    *, seed=None, experimental_rowmax_cache=True,
    experimental_verify_rowmax_cache=False,
    experimental_canonical_rowmax_init=False,
    milestones=(0, 1, 10), max_observations=16,
    max_bytes=64 * 1024 * 1024,
):
    """Capture exact Cython iteration boundaries under its unmodified hotloops.

    In particular, this is a SINGLE compiled propagation: calling the normal
    function for consecutive iteration prefixes would RESET xorshift32.
    Two native kernels (cached and legacy) are preserved byte-for-byte.
    """
    validate_propagation_controls(max_iter, es_chk, stop_max)
    seed = validate_seed(seed)
    active_cache = bool(experimental_rowmax_cache and early_stop != 0)
    # For research fixture inputs these are exact allocated, active shapes.
    source_arrays = {
        "positions": part_curnode,
        "strengths": part_strength,
        "distances": dist_table,
        "dominance": dominance,
        "owndeg": owndeg,
        "rowmax": np.empty(dominance.shape[0], dtype=np.float64),
    }
    schedule, planned_bytes = _plan(
        milestones, max_iter, max_observations, max_bytes, source_arrays,
    )
    out = _record_buffers(source_arrays, schedule)

    # Build is an explicit preflight step; never compile or alter packaging
    # while a propagation with an already seeded RNG is in progress.
    from pcc_step_live_research import pcc_propagate
    info = pcc_propagate(
        neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
        part_curnode, part_label, part_strength, dist_table, dominance, owndeg,
        deltap, dexp, dom_row, reduc, dom_list, dist_list, prob, slices,
        dist_weights, max_iter, early_stop, es_chk, stop_max,
        return_info=True, seed=seed,
        experimental_rowmax_cache=experimental_rowmax_cache,
        experimental_verify_rowmax_cache=experimental_verify_rowmax_cache,
        experimental_canonical_rowmax_init=experimental_canonical_rowmax_init,
        research_observer=out,
    )
    count = int(out["recorded"][0])
    if count > len(schedule):
        raise AssertionError("compiled observer wrote more milestones than allocated")
    snapshots = []
    bytes_per = sum(a.nbytes for a in source_arrays.values()) + 4
    for index in range(count):
        arrays = {}
        for name in source_arrays:
            array = out[name][index]
            array.setflags(write=False)
            arrays[name] = array
        snapshots.append(Milestone(
            schedule[index], MappingProxyType(arrays), int(out["rng"][index]),
        ))
    final_state = int(out["final_rng"][0])
    return ObservedResult(
        MappingProxyType(info), tuple(snapshots), tuple(schedule[:count]),
        bytes_per * count, final_state, xorshift32_next(final_state),
        active_cache,
    )
