# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

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
    return <double>x / 4294967295.0

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
    int              stop_max
):
    cdef Py_ssize_t it, i, n_nodes = neib_list.shape[0]
    cdef double mmpot, max_mmpot = 0.0
    cdef int stop_cnt = 0
    cdef unsigned int rng_state = 123456789
    
    with nogil:
        for it in range(max_iter):
            pcc_step_sequential(neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
                               part_curnode, part_label, part_strength, dist_table,
                               dominance, owndeg, deltap, dexp,
                               dist_weights, prob, slices, &rng_state)

            if early_stop != 0 and it % 10 == 0:
                mmpot = _pcc_calc_mmpot(dominance, n_nodes, c)
                if mmpot > max_mmpot:
                    max_mmpot = mmpot
                    stop_cnt = 0
                else:
                    stop_cnt += 1
                    if stop_cnt > stop_max:
                        break

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
    cdef double randval, step, reduc_val, sum_reduc
    cdef double prob_sum, max_dom
    cdef unsigned char cur_d, next_d
    cdef int greedy
    
    # Pre-allocated buffers (must be at least max_degree)
    cdef double* p_prob = &prob_buf[0]
    cdef double* p_slices = &slices_buf[0]
    cdef Py_ssize_t max_buf_size = prob_buf.shape[0]

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        if curnode < 0 or curnode >= n_nodes: continue

        k = neib_qt[curnode]
        if k <= 0: continue
        
        # Safety: cap degree at buffer size
        if k > max_buf_size:
            k = max_buf_size

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
                    p_prob[i] = 0.0
                else:
                    p_prob[i] = dist_weights[dist_table[next_node, p_i]] * dominance[next_node, label]
                prob_sum += p_prob[i]
                p_slices[i] = prob_sum
            
            if prob_sum > 0.0:
                randval = rand_double(rng_state) * prob_sum
                choice = 0
                for i in range(k):
                    if randval <= p_slices[i]:
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
                    if dominance[next_node, i] < step:
                        reduc_val = dominance[next_node, i]
                    else:
                        reduc_val = step
                    dominance[next_node, i] -= reduc_val
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
    if dist_weights is None:
        dist_weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** dexp
    
    with nogil:
        pcc_step_sequential(neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
                           part_curnode, part_label, part_strength, dist_table,
                           dominance, owndeg, deltap, dexp,
                           dist_weights, prob, slices, &rng_state)
