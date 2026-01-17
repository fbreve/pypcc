# -*- coding: utf-8 -*-
"""
Versão Cython serial otimizada do _pcc_step, com RNG em C (rand).
"""

# distutils: language = c
# distutils: extra_compile_args = -O3 -march=native
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX   # RNG em C

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pcc_step(
    np.int64_t[:, :] neib_list,
    np.int64_t[:]    neib_qt,
    np.int64_t[:]    labels,
    double           p_grd,
    double           delta_v,
    np.int64_t       c,
    double[:]        zerovec,      # mantida pela assinatura
    np.int64_t[:]    part_curnode,
    np.int64_t[:]    part_label,
    double[:]        part_strength,
    np.uint8_t[:, :] dist_table,
    double[:, :]     dominance,
    double[:, :]     owndeg,
):
    """
    Atualiza uma iteração do PCC sobre todas as partículas.
    Arrays são atualizados in-place.
    """

    cdef Py_ssize_t n_particles = part_curnode.shape[0]
    cdef Py_ssize_t p_i, curnode, k, i, choice, next_node, k_rand
    cdef Py_ssize_t label
    cdef double randval, step
    cdef double prob_sum, acc
    cdef double dom_val, d_val
    cdef double tmp, max_dom
    cdef int greedy

    # determinar grau máximo para buffers
    cdef Py_ssize_t max_k = 0
    cdef Py_ssize_t n_nodes = neib_list.shape[0]
    for i in range(neib_qt.shape[0]):
        if neib_qt[i] > max_k:
            max_k = neib_qt[i]
    if max_k <= 0:
        return

    # buffers temporários para dominância
    cdef np.ndarray dom_row_arr = np.empty(c, dtype=np.float64)
    cdef np.ndarray reduc_arr   = np.empty(c, dtype=np.float64)
    cdef double[:] dom_row  = dom_row_arr
    cdef double[:] reduc    = reduc_arr

    # buffers temporários para vizinhos
    cdef np.ndarray dom_list_arr  = np.empty(max_k, dtype=np.float64)
    cdef np.ndarray dist_list_arr = np.empty(max_k, dtype=np.float64)
    cdef np.ndarray prob_arr      = np.empty(max_k, dtype=np.float64)
    cdef np.ndarray slices_arr    = np.empty(max_k, dtype=np.float64)
    cdef double[:] dom_list   = dom_list_arr
    cdef double[:] dist_list  = dist_list_arr
    cdef double[:] prob       = prob_arr
    cdef double[:] slices     = slices_arr

    cdef np.int64_t[:] neighbors_view

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        if curnode < 0 or curnode >= n_nodes:
            continue

        k = neib_qt[curnode]
        if k <= 0:
            continue

        neighbors_view = neib_list[curnode, :k]

        # ===== greedy x random com rand() da libc =====
        randval = rand() / <double>RAND_MAX
        if randval < p_grd:
            # --- greedy walk ---
            label = part_label[p_i]

            # dominância da classe da partícula em cada vizinho
            for i in range(k):
                dom_list[i] = dominance[neighbors_view[i], label]

            # dist_list: 1 / (1 + d)^2
            for i in range(k):
                d_val = <double>dist_table[neighbors_view[i], p_i]
                tmp = 1.0 + d_val
                dist_list[i] = 1.0 / (tmp * tmp)

            for i in range(k):
                prob[i] = dom_list[i] * dist_list[i]

            # cumsum
            prob_sum = 0.0
            for i in range(k):
                prob_sum += prob[i]
                slices[i] = prob_sum

            if slices[k - 1] <= 0.0:
                # fallback: se prob_sum ~ 0, escolhe vizinho aleatório
                choice = rand() % k
            else:
                # roleta: rand em [0, slices[-1]]
                randval = (rand() / <double>RAND_MAX) * slices[k - 1]
                acc = 0.0
                choice = 0
                for i in range(k):
                    acc = slices[i]
                    if randval <= acc:
                        choice = i
                        break

            next_node = neighbors_view[choice]
            greedy = 1
        else:
            # --- random walk ---
            k_rand = k
            choice = rand() % k_rand
            next_node = neighbors_view[choice]
            greedy = 0

        # ===== update dominance / força / dist_table / posição =====

        if labels[next_node] == -1:
            # copia linha de dominância
            for i in range(c):
                dom_row[i] = dominance[next_node, i]

            step = part_strength[p_i] * (delta_v / (c - 1))

            # reduc = dom - max(dom - step, 0)
            for i in range(c):
                tmp = dom_row[i] - step
                if tmp < 0.0:
                    tmp = 0.0
                reduc[i] = dom_row[i] - tmp

            # dom -= reduc
            for i in range(c):
                dom_row[i] -= reduc[i]

            # dom[label] += sum(reduc)
            label = part_label[p_i]
            tmp = 0.0
            for i in range(c):
                tmp += reduc[i]
            dom_row[label] += tmp

            # grava de volta em dominance
            for i in range(c):
                dominance[next_node, i] = dom_row[i]

        label = part_label[p_i]
        part_strength[p_i] = dominance[next_node, label]

        # dist_table
        if dist_table[next_node, p_i] > dist_table[curnode, p_i] + 1:
            dist_table[next_node, p_i] = dist_table[curnode, p_i] + 1

        # owndeg se random
        if greedy == 0:
            owndeg[next_node, label] += part_strength[p_i]

        # choque: move só se classe for dominante
        max_dom = dominance[next_node, 0]
        for i in range(1, c):
            dom_val = dominance[next_node, i]
            if dom_val > max_dom:
                max_dom = dom_val

        if dominance[next_node, label] == max_dom:
            part_curnode[p_i] = next_node
