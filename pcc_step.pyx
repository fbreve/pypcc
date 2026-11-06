# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pcc_metrics import propagation_info
from pcc_validation import (validate_step_inputs, validate_buffers, validate_vector,
                            validate_kernel_scalars, validate_seed, validate_propagation_controls,
                            validate_finite)
from libc.stdlib cimport malloc, free
from libc.math cimport isinf

# Fast XORShift RNG (Sequential version)
@cython.cdivision(True)
cdef inline unsigned int xorshift32(unsigned int *state) noexcept nogil:
    cdef unsigned int x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x

@cython.cdivision(True)
cdef inline double rand_double(unsigned int *state) noexcept nogil:
    cdef unsigned int x = xorshift32(state)
    return <double>x / 4294967296.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _pcc_calc_mmpot(double[:, :] dominance, Py_ssize_t n_nodes, Py_ssize_t c) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double row_max, total_max = 0.0
    for i in range(n_nodes):
        row_max = dominance[i, 0]
        for j in range(1, c):
            if dominance[i, j] > row_max:
                row_max = dominance[i, j]
        total_max += row_max
    return total_max / n_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _pcc_initialize_rowmax_cache(
    double[:, :] dominance, Py_ssize_t n_nodes, Py_ssize_t c,
    double[:] row_max_cache
) noexcept nogil:
    """Compute the same per-row maximum as legacy global potential."""
    cdef Py_ssize_t i, j
    cdef double row_max
    for i in range(n_nodes):
        row_max = dominance[i, 0]
        for j in range(1, c):
            if dominance[i, j] > row_max:
                row_max = dominance[i, j]
        row_max_cache[i] = row_max


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _pcc_calc_mmpot_from_rowmax(
    double[:] row_max_cache, Py_ssize_t n_nodes
) noexcept nogil:
    """Exact original node-index sum order; no running float accumulator."""
    cdef Py_ssize_t i
    cdef double total_max = 0.0
    for i in range(n_nodes):
        total_max += row_max_cache[i]
    return total_max / n_nodes


cpdef pcc_propagate(
    np.int64_t[:, :] neib_list,
    np.int64_t[:]    neib_qt,
    np.int64_t[:]    labels,
    double           p_grd,
    double           delta_v,
    np.int64_t       c,
    double[:]        zerovec,
    np.int64_t[:]    part_curnode,
    np.int64_t[:]    part_label,
    double[:]        part_strength,
    unsigned char[:, :] dist_table,
    double[:, :]     dominance,
    double[:, :]     owndeg,
    double           deltap,
    double           dexp,
    double[:]        dom_row,
    double[:]        reduc,
    double[:]        dom_list,
    double[:]        dist_list,
    double[:]        prob,
    double[:]        slices,
    double[:]        dist_weights,
    int              max_iter,
    int              early_stop,
    int              es_chk,
    int              stop_max,
    bint             return_info = False,
    object           seed = None,
    bint             experimental_rowmax_cache = True,
    bint             experimental_verify_rowmax_cache = False
):
    cdef Py_ssize_t it = -1, i, n_nodes = neib_list.shape[0]
    cdef double mmpot, max_mmpot = 0.0
    cdef int stop_cnt = 0
    cdef unsigned int rng_state = 123456789
    cdef double[:] row_max_cache = None
    cdef bint cache_enabled = bool(experimental_rowmax_cache and early_stop != 0)
    cdef int metric_mismatch_iteration = 0
    if experimental_verify_rowmax_cache and not cache_enabled:
        raise ValueError("experimental metric verification requires active rowmax cache")
    validate_propagation_controls(max_iter, es_chk, stop_max)
    seed = validate_seed(seed)
    if seed is not None:
        rng_state = <unsigned int>seed
        if rng_state == 0:
            rng_state = 123456789  # xorshift32 cannot evolve from zero

    validate_kernel_scalars(p_grd, delta_v, deltap, dexp)
    max_degree = validate_step_inputs(
        neib_list, neib_qt, labels, c, zerovec, part_curnode, part_label,
        part_strength, dist_table, dominance, owndeg)
    validate_buffers(c, max_degree, dom_row, reduc, dom_list, dist_list,
                     prob, slices, dist_weights)
    # Reject nonfinite probability inputs before RNG/state mutation or cache
    # initialization. Scan only elements actually read by the Cython kernel:
    # legal overallocated padding is not part of the active simulation.
    validate_finite("part_strength", np.asarray(part_strength)[:part_curnode.shape[0]])
    validate_finite("dominance", np.asarray(dominance)[:n_nodes, :c])
    validate_finite("dist_weights", np.asarray(dist_weights)[:256])

    if cache_enabled:
        row_max_cache = np.empty(n_nodes, dtype=np.float64)
        with nogil:
            _pcc_initialize_rowmax_cache(dominance, n_nodes, c, row_max_cache)

    with nogil:
        for it in range(max_iter):
            # Two specialized compiled kernels are intentional: the V1
            # per-particle runtime cache conditional regressed default-off
            # Digits runtime by ~30% on Linux. The original hot loop remains
            # identical to master; the cached copy only adds the row-max write.
            # tests/test_cython_cache_hotloop_maintenance.py enforces that
            # invariant whenever either particle movement rule changes.
            if cache_enabled:
                pcc_step_sequential_cached(
                    neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
                    part_curnode, part_label, part_strength, dist_table,
                    dominance, owndeg, deltap, dexp,
                    dist_weights, prob, slices, &rng_state, row_max_cache)
            else:
                pcc_step_sequential(
                    neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
                    part_curnode, part_label, part_strength, dist_table,
                    dominance, owndeg, deltap, dexp,
                    dist_weights, prob, slices, &rng_state)

            if early_stop != 0 and it % 10 == 0:
                if cache_enabled:
                    mmpot = _pcc_calc_mmpot_from_rowmax(row_max_cache, n_nodes)
                    if experimental_verify_rowmax_cache and (
                            mmpot != _pcc_calc_mmpot(dominance, n_nodes, c)):
                        metric_mismatch_iteration = it + 1
                        break
                else:
                    mmpot = _pcc_calc_mmpot(dominance, n_nodes, c)
                if mmpot > max_mmpot:
                    max_mmpot = mmpot
                    stop_cnt = 0
                else:
                    stop_cnt += 1
                    if stop_cnt > stop_max:
                        break

    if metric_mismatch_iteration:
        raise AssertionError(
            f"Incremental potential cache differs from legacy at iteration "
            f"{metric_mismatch_iteration}"
        )
    if return_info:
        return propagation_info(it + 1, early_stop, max_iter, stop_max, stop_cnt)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void pcc_step_sequential(
    np.int64_t[:, :] neib_list,
    np.int64_t[:]    neib_qt,
    np.int64_t[:]    labels,
    double           p_grd,
    double           delta_v,
    np.int64_t       c,
    double[:]        zerovec,
    np.int64_t[:]    part_curnode,
    np.int64_t[:]    part_label,
    double[:]        part_strength,
    unsigned char[:, :] dist_table,
    double[:, :]     dominance,
    double[:, :]     owndeg,
    double           deltap,
    double           dexp,
    double[:]        dist_weights,
    double[:]        prob_buf,
    double[:]        slices_buf,
    unsigned int    *rng_state
) noexcept nogil:
    cdef Py_ssize_t n_particles = part_curnode.shape[0]
    cdef Py_ssize_t n_nodes = neib_list.shape[0]
    cdef Py_ssize_t p_i, curnode, k, i, choice, next_node
    cdef Py_ssize_t label
    cdef double randval, step, reduc_val, sum_reduc, dom_value
    cdef double prob_sum, max_dom, max_weight
    cdef unsigned char cur_d, next_d
    cdef int greedy

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        if curnode < 0 or curnode >= n_nodes: continue

        k = neib_qt[curnode]
        if k <= 0: continue

        label = part_label[p_i]
        if label < 0 or label >= c:
            continue

        randval = rand_double(rng_state)

        if randval < p_grd:
            # greedy
            prob_sum = 0.0
            for i in range(k):
                next_node = neib_list[curnode, i]
                # lookup with bounds check
                if next_node < 0 or next_node >= n_nodes:
                    prob_buf[i] = 0.0
                else:
                    prob_buf[i] = dist_weights[dist_table[next_node, p_i]] * dominance[next_node, label]
                prob_sum += prob_buf[i]

            if isinf(prob_sum):
                max_weight = 0.0
                for i in range(k):
                    if prob_buf[i] > max_weight:
                        max_weight = prob_buf[i]
                prob_sum = 0.0
                for i in range(k):
                    prob_buf[i] /= max_weight
                    prob_sum += prob_buf[i]

            if prob_sum > 0.0:
                randval = rand_double(rng_state) * prob_sum
                choice = 0
                prob_sum = 0.0
                for i in range(k):
                    prob_sum += prob_buf[i]
                    if randval < prob_sum:
                        choice = i
                        break
                next_node = neib_list[curnode, choice]
                greedy = 1
            else:
                next_node = neib_list[curnode, xorshift32(rng_state) % k]
                greedy = 0
        else:
            # random
            next_node = neib_list[curnode, xorshift32(rng_state) % k]
            greedy = 0

        # update
        if next_node >= 0 and next_node < n_nodes:
            if labels[next_node] == -1:
                step = part_strength[p_i] * (delta_v / (c - 1))
                sum_reduc = 0.0
                for i in range(c):
                    dom_value = dominance[next_node, i]
                    if dom_value < step:
                        reduc_val = dom_value
                    else:
                        reduc_val = step
                    dominance[next_node, i] = dom_value - reduc_val
                    sum_reduc += reduc_val
                dominance[next_node, label] += sum_reduc

            if deltap == 1.0:
                part_strength[p_i] = dominance[next_node, label]
            else:
                part_strength[p_i] += (dominance[next_node, label] - part_strength[p_i]) * deltap

            cur_d = dist_table[curnode, p_i]
            next_d = dist_table[next_node, p_i]
            if cur_d < 255:
                if next_d > <unsigned char>(cur_d + 1):
                    dist_table[next_node, p_i] = cur_d + 1

            if greedy == 0:
                owndeg[next_node, label] += part_strength[p_i]

            max_dom = dominance[next_node, 0]
            for i in range(1, c):
                if dominance[next_node, i] > max_dom:
                    max_dom = dominance[next_node, i]

            if dominance[next_node, label] == max_dom:
                part_curnode[p_i] = next_node


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void pcc_step_sequential_cached(
    np.int64_t[:, :] neib_list,
    np.int64_t[:]    neib_qt,
    np.int64_t[:]    labels,
    double           p_grd,
    double           delta_v,
    np.int64_t       c,
    double[:]        zerovec,
    np.int64_t[:]    part_curnode,
    np.int64_t[:]    part_label,
    double[:]        part_strength,
    unsigned char[:, :] dist_table,
    double[:, :]     dominance,
    double[:, :]     owndeg,
    double           deltap,
    double           dexp,
    double[:]        dist_weights,
    double[:]        prob_buf,
    double[:]        slices_buf,
    unsigned int    *rng_state,
    double[:]        row_max_cache
) noexcept nogil:
    cdef Py_ssize_t n_particles = part_curnode.shape[0]
    cdef Py_ssize_t n_nodes = neib_list.shape[0]
    cdef Py_ssize_t p_i, curnode, k, i, choice, next_node
    cdef Py_ssize_t label
    cdef double randval, step, reduc_val, sum_reduc, dom_value
    cdef double prob_sum, max_dom, max_weight
    cdef unsigned char cur_d, next_d
    cdef int greedy

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        if curnode < 0 or curnode >= n_nodes: continue

        k = neib_qt[curnode]
        if k <= 0: continue

        label = part_label[p_i]
        if label < 0 or label >= c:
            continue

        randval = rand_double(rng_state)

        if randval < p_grd:
            # greedy
            prob_sum = 0.0
            for i in range(k):
                next_node = neib_list[curnode, i]
                # lookup with bounds check
                if next_node < 0 or next_node >= n_nodes:
                    prob_buf[i] = 0.0
                else:
                    prob_buf[i] = dist_weights[dist_table[next_node, p_i]] * dominance[next_node, label]
                prob_sum += prob_buf[i]

            if isinf(prob_sum):
                max_weight = 0.0
                for i in range(k):
                    if prob_buf[i] > max_weight:
                        max_weight = prob_buf[i]
                prob_sum = 0.0
                for i in range(k):
                    prob_buf[i] /= max_weight
                    prob_sum += prob_buf[i]

            if prob_sum > 0.0:
                randval = rand_double(rng_state) * prob_sum
                choice = 0
                prob_sum = 0.0
                for i in range(k):
                    prob_sum += prob_buf[i]
                    if randval < prob_sum:
                        choice = i
                        break
                next_node = neib_list[curnode, choice]
                greedy = 1
            else:
                next_node = neib_list[curnode, xorshift32(rng_state) % k]
                greedy = 0
        else:
            # random
            next_node = neib_list[curnode, xorshift32(rng_state) % k]
            greedy = 0

        # update
        if next_node >= 0 and next_node < n_nodes:
            if labels[next_node] == -1:
                step = part_strength[p_i] * (delta_v / (c - 1))
                sum_reduc = 0.0
                for i in range(c):
                    dom_value = dominance[next_node, i]
                    if dom_value < step:
                        reduc_val = dom_value
                    else:
                        reduc_val = step
                    dominance[next_node, i] = dom_value - reduc_val
                    sum_reduc += reduc_val
                dominance[next_node, label] += sum_reduc

            if deltap == 1.0:
                part_strength[p_i] = dominance[next_node, label]
            else:
                part_strength[p_i] += (dominance[next_node, label] - part_strength[p_i]) * deltap

            cur_d = dist_table[curnode, p_i]
            next_d = dist_table[next_node, p_i]
            if cur_d < 255:
                if next_d > <unsigned char>(cur_d + 1):
                    dist_table[next_node, p_i] = cur_d + 1

            if greedy == 0:
                owndeg[next_node, label] += part_strength[p_i]

            max_dom = dominance[next_node, 0]
            for i in range(1, c):
                if dominance[next_node, i] > max_dom:
                    max_dom = dominance[next_node, i]

            if labels[next_node] == -1:
                # Existing movement decision already calculated the final row
                # maximum in the legacy class-order. No second scan needed.
                row_max_cache[next_node] = max_dom

            if dominance[next_node, label] == max_dom:
                part_curnode[p_i] = next_node

cpdef pcc_step(
    np.int64_t[:, :] neib_list,
    np.int64_t[:]    neib_qt,
    np.int64_t[:]    labels,
    double           p_grd,
    double           delta_v,
    np.int64_t       c,
    double[:]        zerovec,
    np.int64_t[:]    part_curnode,
    np.int64_t[:]    part_label,
    double[:]        part_strength,
    unsigned char[:, :] dist_table,
    double[:, :]     dominance,
    double[:, :]     owndeg,
    double           deltap = 1.0,
    double           dexp   = 2.0,
    double[:]        prob = None,
    double[:]        slices = None,
    double[:]        dist_weights = None,
):
    cdef unsigned int rng_state = 123456789
    validate_kernel_scalars(p_grd, delta_v, deltap, dexp)
    max_degree = validate_step_inputs(
        neib_list, neib_qt, labels, c, zerovec, part_curnode, part_label,
        part_strength, dist_table, dominance, owndeg)
    if prob is None:
        prob = np.empty(max_degree, dtype=np.float64)
    if slices is None:
        slices = np.empty(max_degree, dtype=np.float64)
    validate_vector("prob", prob, max_degree)
    validate_vector("slices", slices, max_degree)
    if dist_weights is None:
        dist_weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** dexp
    
    validate_vector("dist_weights", dist_weights, 256)
    # Identical active-input preflight for the public single-step entry point.
    validate_finite("part_strength", np.asarray(part_strength)[:part_curnode.shape[0]])
    validate_finite("dominance", np.asarray(dominance)[:neib_list.shape[0], :c])
    validate_finite("dist_weights", np.asarray(dist_weights)[:256])

    with nogil:
        pcc_step_sequential(neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
                           part_curnode, part_label, part_strength, dist_table,
                           dominance, owndeg, deltap, dexp,
                           dist_weights, prob, slices, &rng_state)
