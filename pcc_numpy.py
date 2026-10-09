# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:01:28 2026

@author: fbrev
"""

# pcc_numpy.py
import numpy as np
import random 

def pcc_step_numpy(neib_list, neib_qt,
                   labels, p_grd, delta_v, c, zerovec,
                   part_curnode, part_label, part_strength, dist_table,
                   dominance, owndeg, deltap=1.0, dexp=2.0,
                   dom_row=None, reduc=None, dom_list=None, dist_list=None, prob=None, slices=None,
                   dist_weights=None):
    """
    Versão NumPy/Python do _pcc_step (Fase 5: Layout Fortran e Loops Nativos).
    """
    """One sequential PCC iteration, matching the Cython/Numba update order.

    NumPy is used for storage and neighbor-weight calculations, but each
    particle must complete its visit before the next one selects a node.
    """
    n_particles = part_curnode.shape[0]
    n_nodes = neib_list.shape[0]
    if dist_weights is None:
        dist_weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** dexp

    for p_i in range(n_particles):
        cur = int(part_curnode[p_i])
        if cur < 0 or cur >= n_nodes:
            continue
        degree = int(neib_qt[cur])
        if degree <= 0:
            continue

        neighbors = neib_list[cur, :degree]
        cls = int(part_label[p_i])
        if cls < 0 or cls >= c:
            continue

        greedy = np.random.random() < p_grd
        if greedy:
            weights = dominance[neighbors, cls] * dist_weights[dist_table[neighbors, p_i]]
            total = float(np.sum(weights))
            if total > 0.0:
                threshold = np.random.random() * total
                idx = int(np.searchsorted(np.cumsum(weights), threshold, side="left"))
                idx = min(idx, degree - 1)
            else:
                idx = int(np.random.randint(degree))
                greedy = False
        else:
            idx = int(np.random.randint(degree))

        nxt = int(neighbors[idx])
        if nxt < 0 or nxt >= n_nodes:
            continue

        if labels[nxt] == -1:
            step = part_strength[p_i] * (delta_v / (c - 1))
            reduction = np.minimum(dominance[nxt, :], step)
            dominance[nxt, :] -= reduction
            dominance[nxt, cls] += reduction.sum()

        if deltap == 1.0:
            part_strength[p_i] = dominance[nxt, cls]
        else:
            part_strength[p_i] += (dominance[nxt, cls] - part_strength[p_i]) * deltap

        cur_dist = int(dist_table[cur, p_i])
        next_dist = int(dist_table[nxt, p_i])
        if cur_dist < 255 and next_dist > cur_dist + 1:
            dist_table[nxt, p_i] = cur_dist + 1

        if not greedy:
            owndeg[nxt, cls] += part_strength[p_i]

        if dominance[nxt, cls] >= np.max(dominance[nxt, :]):
            part_curnode[p_i] = nxt

def pcc_propagate_numpy(neib_list, neib_qt,
                        labels, p_grd, delta_v, c, zerovec,
                        part_curnode, part_label, part_strength, dist_table,
                        dominance, owndeg, deltap, dexp,
                        dom_row, reduc, dom_list, dist_list, prob, slices,
                        dist_weights,
                        max_iter, early_stop, es_chk, stop_max):
    
    max_mmpot = 0.0
    stop_cnt = 0
    
    for it in range(max_iter):
        pcc_step_numpy(neib_list, neib_qt,
                       labels, p_grd, delta_v, c, zerovec,
                       part_curnode, part_label, part_strength, dist_table,
                       dominance, owndeg, deltap, dexp,
                       dom_row, reduc, dom_list, dist_list, prob, slices,
                       dist_weights)
        
        if early_stop and it % 10 == 0:
            mmpot = np.mean(np.max(dominance, axis=1))
            if mmpot > max_mmpot:
                max_mmpot = mmpot
                stop_cnt = 0
            else:
                stop_cnt += 1
                if stop_cnt > stop_max:
                    break