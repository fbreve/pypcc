"""Research-only live milestone observation of the current compiled Numba kernel.

No changes to production modules. The loop and particle step remain nopython
compiled; snapshots are copied into a bounded preallocated output buffer.
This is deliberately independent of historical PR #18 and NumPy PR #253.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numba import njit

from pcc_numba import (
    _pcc_step_numba_sequential, _pcc_calc_mmpot_sequential, _seed_numba_rng,
)
from pcc_metrics import propagation_info
from pcc_validation import (
    validate_kernel_scalars, validate_step_inputs, validate_buffers,
    validate_fractional_buffers, validate_mutable_arrays,
    validate_propagation_controls, validate_seed,
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


def _plan(milestones, max_iter, max_observations, max_bytes, states):
    if type(max_iter) is not int or max_iter < 0:
        raise ValueError("max_iter must be nonnegative int")
    if type(max_observations) is not int or not 1 <= max_observations <= 32:
        raise ValueError("max_observations must be an integer in 1..32")
    if type(max_bytes) is not int or not 0 < max_bytes <= 64 * 1024 * 1024:
        raise ValueError("max_bytes must be an integer in 1..67108864")
    if type(milestones) not in (tuple, list) or not milestones:
        raise ValueError("milestones must be a nonempty tuple/list")
    if len(milestones) > max_observations:
        raise ValueError("too many observations")
    if any(type(x) is not int or not 0 <= x <= max_iter for x in milestones):
        raise ValueError("milestones must contain integers in 0..max_iter")
    if tuple(milestones) != tuple(sorted(set(milestones))):
        raise ValueError("milestones must be strictly increasing and unique")
    required = sum(x.nbytes for x in states.values()) * len(milestones)
    if required > max_bytes:
        raise ValueError("planned snapshot arrays exceed max_bytes")
    return tuple(milestones), required


@njit(cache=True)
def _observe_compiled(
    neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
    part_curnode, part_label, part_strength, dist_table, dominance, owndeg,
    deltap, dexp, dom_row, reduc, dom_list, dist_list, prob, slices,
    dist_weights, max_iter, early_stop, stop_max, schedule,
    positions_out, strengths_out, distances_out, dominance_out, owndeg_out,
):
    n_nodes = neib_list.shape[0]
    max_mmpot = 0.0
    stop_cnt = 0
    count = 0
    it = -1

    if schedule[0] == 0:
        positions_out[count, :] = part_curnode
        strengths_out[count, :] = part_strength
        distances_out[count, :, :] = dist_table
        dominance_out[count, :, :] = dominance
        owndeg_out[count, :, :] = owndeg
        count += 1

    for it in range(max_iter):
        _pcc_step_numba_sequential(
            neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
            part_curnode, part_label, part_strength, dist_table,
            dominance, owndeg, deltap, dexp, dom_row, reduc,
            dom_list, dist_list, prob, slices, dist_weights,
        )

        stopped = False
        if early_stop and it % 10 == 0:
            mmpot = _pcc_calc_mmpot_sequential(dominance, n_nodes, c)
            if mmpot > max_mmpot:
                max_mmpot = mmpot
                stop_cnt = 0
            else:
                stop_cnt += 1
                if stop_cnt > stop_max:
                    stopped = True

        if count < schedule.shape[0] and schedule[count] == it + 1:
            positions_out[count, :] = part_curnode
            strengths_out[count, :] = part_strength
            distances_out[count, :, :] = dist_table
            dominance_out[count, :, :] = dominance
            owndeg_out[count, :, :] = owndeg
            count += 1
        if stopped:
            break

    return it + 1, stop_cnt, count


def observe_numba_sequential(
    neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
    part_curnode, part_label, part_strength, dist_table,
    dominance, owndeg, deltap, dexp, dom_row, reduc, dom_list,
    dist_list, prob, slices, dist_weights,
    max_iter, early_stop, es_chk, stop_max, *,
    seed=None, milestones=(0, 1, 10),
    max_observations=16, max_bytes=64 * 1024 * 1024,
):
    """Capture exact within-Numba state at selected completed iterations."""
    validate_kernel_scalars(p_grd, delta_v, deltap, dexp)
    max_degree = validate_step_inputs(
        neib_list, neib_qt, labels, c, zerovec, part_curnode,
        part_label, part_strength, dist_table, dominance, owndeg,
    )
    validate_buffers(c, max_degree, dom_row, reduc, dom_list,
                     dist_list, prob, slices, dist_weights)
    validate_fractional_buffers(part_strength, dominance, owndeg,
                                dom_row, dom_list, dist_list, prob, slices)
    validate_mutable_arrays(
        ("part_curnode", part_curnode), ("part_strength", part_strength),
        ("dist_table", dist_table), ("dominance", dominance),
        ("owndeg", owndeg), ("dom_row", dom_row), ("reduc", reduc),
        ("dom_list", dom_list), ("dist_list", dist_list),
        ("prob", prob), ("slices", slices),
    )
    validate_propagation_controls(max_iter, es_chk, stop_max)
    seed = validate_seed(seed)

    states = {
        "positions": part_curnode, "strengths": part_strength,
        "distances": dist_table, "dominance": dominance, "owndeg": owndeg,
    }
    schedule, planned_bytes = _plan(
        milestones, max_iter, max_observations, max_bytes, states,
    )

    # All allocations and schedule checks precede any Numba RNG seeding.
    output = {
        name: np.empty((len(schedule),) + value.shape, dtype=value.dtype)
        for name, value in states.items()
    }
    if seed is not None:
        _seed_numba_rng(seed)

    iterations, stop_cnt, count = _observe_compiled(
        neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
        part_curnode, part_label, part_strength, dist_table,
        dominance, owndeg, deltap, dexp, dom_row, reduc,
        dom_list, dist_list, prob, slices, dist_weights,
        max_iter, early_stop, stop_max, np.asarray(schedule, dtype=np.int64),
        output["positions"], output["strengths"], output["distances"],
        output["dominance"], output["owndeg"],
    )
    snapshots = []
    for index in range(count):
        arrays = {}
        for name, values in output.items():
            snapshot = values[index]
            snapshot.setflags(write=False)
            arrays[name] = snapshot
        snapshots.append(Milestone(schedule[index], MappingProxyType(arrays)))

    # Hide the owning backing arrays from callers; each captured slot is
    # disjoint. Read-only views are not a security boundary against .base.
    info = propagation_info(
        iterations, early_stop, max_iter, stop_max, stop_cnt,
    )
    return ObservedResult(
        MappingProxyType(info), tuple(snapshots),
        sum(x.nbytes for x in states.values()) * count,
        tuple(schedule[:count]),
    )
