# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:01:28 2026

@author: fbrev
"""

# pcc_numpy.py
import numpy as np

def pcc_step_numpy(neib_list, neib_qt,
                   labels, p_grd, delta_v, c, zerovec,
                   part_curnode, part_label, part_strength, dist_table,
                   dominance, owndeg):
    """
    Versão NumPy/Python do _pcc_step (lógica espelhada do Numba/Cython).
    Tudo in-place.
    """
    n_particles = part_curnode.shape[0]

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        k = neib_qt[curnode]

        neighbors = neib_list[curnode, :k]

        use_greedy = np.random.random() < p_grd
        if use_greedy:
            # greedy
            label = part_label[p_i]

            dom_list = dominance[neighbors, label]
            d = dist_table[neighbors, p_i].astype(np.float64)
            dist_list = 1.0 / ((1.0 + d) * (1.0 + d))

            prob = dom_list * dist_list
            total = prob.sum()

            if total > 0.0:
                slices = np.cumsum(prob)
                rand = np.random.uniform(0.0, total)
                choice = np.searchsorted(slices, rand)
                next_node = neighbors[choice]
                greedy = True
            else:
                # all neighbors have zero dominance for this label — fall back to random
                idx = np.random.randint(neighbors.shape[0])
                next_node = neighbors[idx]
                greedy = False
        else:
            # random
            idx = np.random.randint(neighbors.shape[0])
            next_node = neighbors[idx]
            greedy = False

        # dominance
        if labels[next_node] == -1:
            dom = dominance[next_node, :]
            step = part_strength[p_i] * (delta_v / (c - 1))

            reduc = dom - np.maximum(dom - step, zerovec)

            dom -= reduc
            dom[part_label[p_i]] += np.sum(reduc)

        # strength
        part_strength[p_i] = dominance[next_node, part_label[p_i]]

        # dist_table — cast to int to avoid uint8 overflow when curnode value is 255
        if int(dist_table[next_node, p_i]) > int(dist_table[curnode, p_i]) + 1:
            dist_table[next_node, p_i] = dist_table[curnode, p_i] + 1

        # owndeg
        if not greedy:
            owndeg[next_node, part_label[p_i]] += part_strength[p_i]

        # choque
        if dominance[next_node, part_label[p_i]] == np.max(dominance[next_node, :]):
            part_curnode[p_i] = next_node