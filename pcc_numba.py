# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:02:02 2026

@author: fbrev
"""

# pcc_numba.py
import numpy as np
from numba import njit

@njit
def _pcc_step_numba(neib_list, neib_qt,
                    labels, p_grd, delta_v, c, zerovec,
                    part_curnode, part_label, part_strength, dist_table,
                    dominance, owndeg):
    n_particles = part_curnode.shape[0]

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        k = neib_qt[curnode]

        neighbors = neib_list[curnode, :k]

        if np.random.random() < p_grd:
            # greedy
            label = part_label[p_i]

            dom_list = dominance[neighbors, label]

            d = dist_table[neighbors, p_i].astype(np.float64)
            dist_list = 1.0 / ((1.0 + d) * (1.0 + d))

            prob = dom_list * dist_list
            slices = np.cumsum(prob)

            rand = np.random.uniform(0.0, slices[-1])
            choice = np.searchsorted(slices, rand)
            next_node = neighbors[choice]
            greedy = True
        else:
            # random
            k_rand = neighbors.shape[0]
            idx = np.random.randint(k_rand)
            next_node = neighbors[idx]
            greedy = False

        if labels[next_node] == -1:
            dom = dominance[next_node, :]
            step = part_strength[p_i] * (delta_v / (c - 1))

            reduc = dom - np.maximum(dom - step, zerovec)

            dom -= reduc
            dom[part_label[p_i]] += np.sum(reduc)

        part_strength[p_i] = dominance[next_node, part_label[p_i]]

        if dist_table[next_node, p_i] > dist_table[curnode, p_i] + 1:
            dist_table[next_node, p_i] = dist_table[curnode, p_i] + 1

        if not greedy:
            owndeg[next_node, part_label[p_i]] += part_strength[p_i]

        if dominance[next_node, part_label[p_i]] == np.max(dominance[next_node, :]):
            part_curnode[p_i] = next_node

def pcc_step_numba(*args):
    # wrapper apenas para manter assinatura semelhante às outras
    return _pcc_step_numba(*args)
