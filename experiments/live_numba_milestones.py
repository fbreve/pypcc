"""Experimental Numba-native PCC live milestones on current master.

Only research use: calls the exact existing @njit sequential particle step
and metric, without patching production modules or using Python callbacks
inside the compiled propagation loop. Not a migration of frozen PR #18 C2.
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
    validate_step_inputs, validate_buffers, validate_kernel_scalars,
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
    observed_iterations: tuple
    reserved_bytes: int
    bytes_copied: int


def _validate_plan(milestones, max_iter, max_snapshots, max_bytes, states):
    # Full worst-case allocation is decided before seeding the Numba RNG.
    if type(max_iter) is not int or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative Python integer")
    if type(max_snapshots) is not int or not 1 <= max_snapshots <= 32:
        raise ValueError("max_snapshots must be an integer in 1..32")
    if type(max_bytes) is not int or not 0 < max_bytes <= 64 * 1024 * 1024:
        raise ValueError("max_bytes must be an integer in 1..67108864")
    if type(milestones) not in (list, tuple) or not milestones:
        raise ValueError("milestones must be a nonempty list or tuple")
    if len(milestones) > max_snapshots:
        raise ValueError("too many milestones")
    if any(type(x) is not int or not 0 <= x <= max_iter
           for x in milestones):
        raise ValueError("milestones must be integers in 0..max_iter")
    if tuple(milestones) != tuple(sorted(set(milestones))):
        raise ValueError("milestones must be strictly increasing and unique")
    per = sum(x.nbytes for x in states.values())
    requested = len(milestones) * per
    if requested > max_bytes:
        raise ValueError("reserved snapshot arrays exceed max_bytes")
    return tuple(milestones), requested, per


@njit(cache=True)
def _copy_at(index, dominance, owndeg, part_curnode, part_strength,
             dist_table, out_dom, out_own, out_pos, out_strength, out_dist):
    # Do not read, seed, or draw from any random generator here.
    for i in range(dominance.shape[0]):
        for j in range(dominance.shape[1]):
            out_dom[index, i, j] = dominance[i, j]
            out_own[index, i, j] = owndeg[i, j]
    for i in range(part_curnode.shape[0]):
        out_pos[index, i] = part_curnode[i]
        out_strength[index, i] = part_strength[i]
    for i in range(dist_table.shape[0]):
        for j in range(dist_table.shape[1]):
            out_dist[index, i, j] = dist_table[i, j]


@njit(cache=True)
def _run_observed_numba(neib_list, neib_qt, labels, p_grd, delta_v, c,
                        zerovec, part_curnode, part_label, part_strength,
                        dist_table, dominance, owndeg, deltap, dexp,
                        dom_row, reduc, dom_list, dist_list, prob, slices,
                        dist_weights, max_iter, early_stop, es_chk, stop_max,
                        requested, out_dom, out_own, out_pos, out_strength,
                        out_dist):
    # This *outer* loop preserves the production Numba check/break order.
    # It does not reimplement _pcc_step_numba_sequential or change RNG draws.
    it = -1
    n_nodes = neib_list.shape[0]
    max_mmpot = 0.0
    stop_cnt = 0
    next_slot = 0

    if requested[0] == 0:
        _copy_at(next_slot, dominance, owndeg, part_curnode,
                 part_strength, dist_table,
                 out_dom, out_own, out_pos, out_strength, out_dist)
        next_slot += 1

    for it in range(max_iter):
        _pcc_step_numba_sequential(
            neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
            part_curnode, part_label, part_strength, dist_table,
            dominance, owndeg, deltap, dexp, dom_row, reduc, dom_list,
            dist_list, prob, slices, dist_weights,
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

        if next_slot < requested.shape[0] and it + 1 == requested[next_slot]:
            _copy_at(next_slot, dominance, owndeg, part_curnode,
                     part_strength, dist_table,
                     out_dom, out_own, out_pos, out_strength, out_dist)
            next_slot += 1
        if stopped:
            break

    return it + 1, stop_cnt, next_slot


def observe_numba_sequential(
    neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
    part_curnode, part_label, part_strength, dist_table, dominance, owndeg,
    deltap, dexp, dom_row, reduc, dom_list, dist_list, prob, slices,
    dist_weights, max_iter, early_stop, es_chk, stop_max,
    *, seed=None, milestones=(0, 1, 10), max_snapshots=16,
    max_bytes=64 * 1024 * 1024,
):
    """Capture genuine compiled Numba iteration-boundary states, opt-in.

    Input validation order reproduces the current Numba propagation wrapper.
    Every potential snapshot is allocated and budget-checked *before* seeding.
    Numba records directly in preallocated independent arrays; no Python or
    callbacks are entered from its compiled particle propagation loop.
    """
    validate_kernel_scalars(p_grd, delta_v, deltap, dexp)
    max_degree = validate_step_inputs(
        neib_list, neib_qt, labels, c, zerovec, part_curnode, part_label,
        part_strength, dist_table, dominance, owndeg,
    )
    validate_buffers(c, max_degree, dom_row, reduc, dom_list, dist_list,
                     prob, slices, dist_weights)
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

    arrays = {
        "dominance": dominance, "owndeg": owndeg,
        "positions": part_curnode, "strengths": part_strength,
        "distances": dist_table,
    }
    schedule, reserved, per = _validate_plan(
        milestones, max_iter, max_snapshots, max_bytes, arrays,
    )
    length = len(schedule)
    out_dom = np.empty((length,) + dominance.shape, dtype=dominance.dtype)
    out_own = np.empty((length,) + owndeg.shape, dtype=owndeg.dtype)
    out_pos = np.empty((length,) + part_curnode.shape, dtype=part_curnode.dtype)
    out_strength = np.empty((length,) + part_strength.shape,
                            dtype=part_strength.dtype)
    out_dist = np.empty((length,) + dist_table.shape, dtype=dist_table.dtype)

    if seed is not None:
        _seed_numba_rng(seed)

    iterations, stop_cnt, captured = _run_observed_numba(
        neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
        part_curnode, part_label, part_strength, dist_table,
        dominance, owndeg, deltap, dexp, dom_row, reduc, dom_list,
        dist_list, prob, slices, dist_weights,
        max_iter, early_stop, es_chk, stop_max,
        np.asarray(schedule, dtype=np.int64),
        out_dom, out_own, out_pos, out_strength, out_dist,
    )

    observations = []
    for i in range(captured):
        saved = {}
        for name, value in (
            ("dominance", out_dom[i]), ("owndeg", out_own[i]),
            ("positions", out_pos[i]), ("strengths", out_strength[i]),
            ("distances", out_dist[i]),
        ):
            # View is independent of model arrays, but backed by the private
            # preallocated snapshot slab; read-only is an API convention,
            # not a cryptographic archival integrity guarantee.
            view = value.view()
            view.setflags(write=False)
            saved[name] = view
        observations.append(Milestone(schedule[i], MappingProxyType(saved)))

    return ObservedResult(
        MappingProxyType(
            propagation_info(iterations, early_stop, max_iter, stop_max,
                             stop_cnt)
        ),
        tuple(observations),
        tuple(schedule[:captured]),
        reserved,
        captured * per,
    )
