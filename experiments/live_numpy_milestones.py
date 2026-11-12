"""Experimental bounded *live* NumPy iteration-boundary observations.

Research only: uses the current production NumPy step function unchanged and
duplicates ONLY its tiny outer stop/control loop so the actual particle/RNG
operations remain the canonical implementation. No hooks in native kernels.
NOT the PR #18 protocol; old C2 source-specific preflights are incompatible.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from pcc_numpy import _pcc_step_numpy_sequential_unchecked
from pcc_metrics import propagation_info
from pcc_validation import (
    validate_step_inputs, validate_kernel_scalars,
    validate_fractional_buffers, validate_mutable_arrays,
    validate_seed, validate_propagation_controls,
)


@dataclass(frozen=True)
class Milestone:
    iteration: int
    arrays: object


@dataclass(frozen=True)
class ObservedResult:
    info: object
    milestones: tuple
    bytes_copied: int
    observed_iterations: tuple


def _plan(values, max_iter, max_observations, max_bytes, data):
    # Plan and budget every possible copy before RNG seed/state mutation.
    if type(max_iter) is not int or max_iter < 0:
        raise ValueError("max_iter must be nonnegative int")
    if type(max_observations) is not int or not 1 <= max_observations <= 32:
        raise ValueError("max_observations must be an integer in 1..32")
    if type(max_bytes) is not int or not 0 < max_bytes <= 64 * 1024 * 1024:
        raise ValueError("max_bytes must be an integer in 1..67108864")
    if type(values) not in (tuple, list) or not values:
        raise ValueError("milestones must be a nonempty tuple/list")
    if len(values) > max_observations:
        raise ValueError("too many observation milestones")
    if any(type(v) is not int or not 0 <= v <= max_iter for v in values):
        raise ValueError("milestones must be integers in 0..max_iter")
    if tuple(values) != tuple(sorted(set(values))):
        raise ValueError("milestones must be strictly increasing and unique")
    bytes_per = sum(x.nbytes for x in data.values())
    requested = bytes_per * len(values)
    if requested > max_bytes:
        raise ValueError("planned observation arrays exceed max_bytes")
    return tuple(values), bytes_per


def observe_numpy_sequential(
    neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
    part_curnode, part_label, part_strength, dist_table,
    dominance, owndeg, deltap, dexp,
    dom_row, reduc, dom_list, dist_list, prob, slices,
    dist_weights, max_iter, early_stop, es_chk, stop_max,
    *, seed=None, milestones=(0, 1, 10), max_observations=16,
    max_bytes=64 * 1024 * 1024,
):
    """Observe selected NumPy iteration boundaries with original step/RNG.

    Uses the exact production sequential **particle step** callable. The
    outer stopping logic repeats production's exact check cadence (iterations
    1, 11, 21, ...) and strict increase criterion; no early-stop default or
    algorithmic policy is changed. Observations happen *after* each completed
    iteration's stopping check and before return. At iteration zero the
    initial state is copied immediately before the first particle iteration.

    Arrays are copied into separate read-only C-contiguous allocations.
    No ground truth, label-derived oracle or live Python callback is used.
    Every copy is planned/budgeted *before* consuming the propagation seed.
    """

    # Exactly the validations and order of production's sequential wrapper.
    validate_step_inputs(neib_list, neib_qt, labels, c, zerovec,
                         part_curnode, part_label, part_strength,
                         dist_table, dominance, owndeg)
    validate_kernel_scalars(p_grd, delta_v, deltap, dexp)
    validate_fractional_buffers(part_strength, dominance, owndeg)
    validate_mutable_arrays(("part_curnode", part_curnode),
                            ("part_strength", part_strength),
                            ("dist_table", dist_table),
                            ("dominance", dominance), ("owndeg", owndeg))
    validate_propagation_controls(max_iter, es_chk, stop_max)
    seed = validate_seed(seed)

    states = {
        "dominance": dominance,
        "owndeg": owndeg,
        "positions": part_curnode,
        "strengths": part_strength,
        "distances": dist_table,
    }
    schedule, bytes_per = _plan(
        milestones, max_iter, max_observations, max_bytes, states
    )
    result = []
    bytes_copied = 0

    def capture(iteration):
        nonlocal bytes_copied
        # No RNG reads/draws and no mutations of the original arrays.
        values = {}
        for name, source in states.items():
            copy = np.array(source, copy=True, order="C")
            copy.setflags(write=False)
            values[name] = copy
        result.append(Milestone(iteration, MappingProxyType(values)))
        bytes_copied += bytes_per

    if seed is not None:
        np.random.seed(seed)
    desired = set(schedule)
    if 0 in desired:
        capture(0)

    it = -1
    max_mmpot = 0.0
    stop_cnt = 0
    for it in range(max_iter):
        _pcc_step_numpy_sequential_unchecked(
            neib_list, neib_qt, labels, p_grd, delta_v, c,
            part_curnode, part_label, part_strength, dist_table,
            dominance, owndeg, deltap, dexp, dist_weights,
        )

        stopped = False
        if early_stop and it % 10 == 0:
            mmpot = np.mean(np.max(dominance, axis=1))
            if mmpot > max_mmpot:
                max_mmpot = mmpot
                stop_cnt = 0
            else:
                stop_cnt += 1
                if stop_cnt > stop_max:
                    stopped = True

        if it + 1 in desired:
            capture(it + 1)
        if stopped:
            break

    return ObservedResult(
        MappingProxyType(propagation_info(
            it + 1, early_stop, max_iter, stop_max, stop_cnt
        )),
        tuple(result), bytes_copied, tuple(x.iteration for x in result),
    )
