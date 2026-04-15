# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:02:02 2026

@author: fbrev
"""

# pcc_numba.py
import numpy as np
from numba import njit

from numba import njit

@njit
def _pcc_step_numba_sequential(neib_list, neib_qt,
                               labels, p_grd, delta_v, c, zerovec,
                               part_curnode, part_label, part_strength, dist_table,
                               dominance, owndeg, deltap, dexp,
                               dom_row, reduc, dom_list, dist_list, prob, slices,
                               dist_weights):
    n_particles = part_curnode.shape[0]
    n_nodes = neib_list.shape[0]

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        if curnode < 0 or curnode >= n_nodes:
            continue
            
        k = neib_qt[curnode]
        if k <= 0:
            continue

        neighbors = neib_list[curnode, :k]

        use_greedy = np.random.random() < p_grd
        label = part_label[p_i]
        
        if use_greedy:
            # greedy
            for i in range(k):
                next_node = neighbors[i]
                if next_node >= 0 and next_node < n_nodes and label >= 0 and label < c:
                    dom_list[i] = dominance[next_node, label]
                else:
                    dom_list[i] = 0.0
                
                if next_node >= 0 and next_node < n_nodes:
                    d_idx = dist_table[next_node, p_i]
                    dist_list[i] = dist_weights[d_idx]
                else:
                    dist_list[i] = dist_weights[255]
            
            for i in range(k):
                prob[i] = dom_list[i] * dist_list[i]
            
            total = 0.0
            for i in range(k):
                total += prob[i]
                slices[i] = total

            if total > 0.0:
                rand_val = np.random.uniform(0.0, total)
                choice = 0
                for i in range(k):
                    if rand_val <= slices[i]:
                        choice = i
                        break
                next_node = neighbors[choice]
                greedy = True
            else:
                idx = np.random.randint(k)
                next_node = neighbors[idx]
                greedy = False
        else:
            # random
            idx = np.random.randint(k)
            next_node = neighbors[idx]
            greedy = False

        # update dominance / força / dist_table / posição
        if next_node >= 0 and next_node < n_nodes and label >= 0 and label < c:
            if labels[next_node] == -1:
                for i in range(c):
                    dom_row[i] = dominance[next_node, i]
                    
                step = part_strength[p_i] * (delta_v / (c - 1))
                sum_reduc = 0.0
                for i in range(c):
                    if dom_row[i] < step:
                        reduc_val = dom_row[i]
                    else:
                        reduc_val = step
                    dom_row[i] -= reduc_val
                    sum_reduc += reduc_val

                dom_row[label] += sum_reduc
                for i in range(c):
                    dominance[next_node, i] = dom_row[i]
                
            if deltap == 1.0:
                part_strength[p_i] = dominance[next_node, label]
            else:
                part_strength[p_i] += (dominance[next_node, label] - part_strength[p_i]) * deltap

            cur_d = dist_table[curnode, p_i]
            next_d = dist_table[next_node, p_i]
            if cur_d < 255:
                if next_d > cur_d + 1:
                    dist_table[next_node, p_i] = cur_d + 1

            if not greedy:
                owndeg[next_node, label] += part_strength[p_i]

            max_dom = dominance[next_node, 0]
            for i in range(1, c):
                if dominance[next_node, i] > max_dom:
                    max_dom = dominance[next_node, i]

            if dominance[next_node, label] == max_dom:
                part_curnode[p_i] = next_node

@njit
def _pcc_calc_mmpot_sequential(dominance, n_nodes, c):
    """
    Sequential version of mean amax.
    """
    total_max = 0.0
    for i in range(n_nodes):
        row_max = dominance[i, 0]
        for j in range(1, c):
            if dominance[i, j] > row_max:
                row_max = dominance[i, j]
        total_max += row_max
    return total_max / n_nodes

@njit
def pcc_propagate_numba(neib_list, neib_qt,
                        labels, p_grd, delta_v, c, zerovec,
                        part_curnode, part_label, part_strength, dist_table,
                        dominance, owndeg, deltap, dexp,
                        dom_row, reduc, dom_list, dist_list, prob, slices,
                        dist_weights,
                        max_iter, early_stop, es_chk, stop_max):
    
    n_nodes = neib_list.shape[0]
    max_mmpot = 0.0
    stop_cnt = 0
    
    for it in range(max_iter):
        _pcc_step_numba_sequential(neib_list, neib_qt,
                                   labels, p_grd, delta_v, c, zerovec,
                                   part_curnode, part_label, part_strength, dist_table,
                                   dominance, owndeg, deltap, dexp,
                                   dom_row, reduc, dom_list, dist_list, prob, slices,
                                   dist_weights)
        
        if early_stop and it % 10 == 0:
            mmpot = _pcc_calc_mmpot_sequential(dominance, n_nodes, c)
            if mmpot > max_mmpot:
                max_mmpot = mmpot
                stop_cnt = 0
            else:
                stop_cnt += 1
                if stop_cnt > stop_max:
                    break

def pcc_step_numba(neib_list, neib_qt,
                   labels, p_grd, delta_v, c, zerovec,
                   part_curnode, part_label, part_strength, dist_table,
                   dominance, owndeg, deltap=1.0, dexp=2.0,
                   dom_row=None, reduc=None, dom_list=None, dist_list=None, prob=None, slices=None,
                   dist_weights=None):
    # Backward compatibility wrapper for single step
    if dom_row is None:
        dom_row = np.empty(c, dtype=np.float64)
        reduc = np.empty(c, dtype=np.float64)
        max_k = 32
        dom_list = np.empty(max_k, dtype=np.float64)
        dist_list = np.empty(max_k, dtype=np.float64)
        prob = np.empty(max_k, dtype=np.float64)
        slices = np.empty(max_k, dtype=np.float64)
    
    if dist_weights is None:
        dist_weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** dexp
        
    return _pcc_step_numba(neib_list, neib_qt,
                           labels, p_grd, delta_v, c, zerovec,
                           part_curnode, part_label, part_strength, dist_table,
                           dominance, owndeg, deltap, dexp,
                           dom_row, reduc, dom_list, dist_list, prob, slices,
                           dist_weights)