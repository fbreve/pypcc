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
                   dist_weights=None, update_mode="parallel", trace=None):
    """
    Versão NumPy/Python do _pcc_step (Fase 5: Layout Fortran e Loops Nativos).
    """
    """One sequential PCC iteration, matching the Cython/Numba update order.

    NumPy is used for storage and neighbor-weight calculations, but each
    particle must complete its visit before the next one selects a node.
    """
    if update_mode == "parallel":
        return _pcc_step_numpy_parallel(
            neib_list, neib_qt, labels, p_grd, delta_v, c, zerovec,
            part_curnode, part_label, part_strength, dist_table,
            dominance, owndeg, deltap, dexp, dom_row, reduc, dom_list,
            dist_list, prob, slices, dist_weights)
    if update_mode != "sequential":
        raise ValueError("update_mode must be 'parallel' or 'sequential'")
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
        if trace is not None:
            trace[p_i, 0] = nxt
            trace[p_i, 1] = int(greedy)
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

        accepted = dominance[nxt, cls] >= np.max(dominance[nxt, :])
        if trace is not None:
            trace[p_i, 2] = int(accepted)
        if accepted:
            part_curnode[p_i] = nxt

def _pcc_step_numpy_parallel(neib_list, neib_qt,
                   labels, p_grd, delta_v, c, zerovec,
                   part_curnode, part_label, part_strength, dist_table,
                   dominance, owndeg, deltap=1.0, dexp=2.0,
                   dom_row=None, reduc=None, dom_list=None, dist_list=None, prob=None, slices=None,
                   dist_weights=None):
    n_particles = part_curnode.shape[0]
    n_nodes = neib_list.shape[0]
    is_deltap_one = (deltap == 1.0)
    if dist_weights is None:
        dist_weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** dexp
    
    # 1. Validation and Setup
    valid_mask = (part_curnode >= 0) & (part_curnode < n_nodes)
    valid_mask &= neib_qt[np.clip(part_curnode, 0, n_nodes - 1)] > 0
    if not np.any(valid_mask):
        return

    # 2. Movement Decision (Greedy vs Random)
    random_vals = np.random.random(n_particles)
    greedy_mask = valid_mask & (random_vals < p_grd)
    random_mask = valid_mask & (random_vals >= p_grd)
    
    next_nodes = part_curnode.copy()
    is_greedy = np.zeros(n_particles, dtype=bool)

    # 3. Process Greedy Walks
    if np.any(greedy_mask):
        g_indices = np.where(greedy_mask)[0]
        cur_nodes_g = part_curnode[g_indices]
        p_labels_g = part_label[g_indices]
        
        # Max neighbors in this batch
        k_vals_g = neib_qt[cur_nodes_g]
        max_k_g = np.max(k_vals_g)
        
        neighbors_g = neib_list[cur_nodes_g, :max_k_g]
        k_mask_g = np.arange(max_k_g) < k_vals_g[:, None]
        safe_neighbors_g = np.where(k_mask_g, neighbors_g, 0)
        
        # Dominance of particle class at neighbors: dominance[neighbors, p_labels]
        # neighbors shape: (n_g, max_k_g). p_labels_g shape: (n_g,)
        # We use fancy indexing. dominance is (n_nodes, c).
        dom_vals_g = dominance[safe_neighbors_g, p_labels_g[:, None]]
        
        # Distance weights: dist_weights[dist_table[neighbors, particle_idx]]
        # dist_table is (n_nodes, n_particles)
        # We need dist_table[neighbors[i, j], g_indices[i]]
        d_indices_g = dist_table[safe_neighbors_g, g_indices[:, None]]
        dist_vals_g = dist_weights[d_indices_g]
        
        prob_vec_g = dom_vals_g * dist_vals_g
        
        # Mask out-of-bounds neighbors (k < max_k_g)
        k_mask_g = np.arange(max_k_g) < k_vals_g[:, None]
        prob_vec_g[~k_mask_g] = 0.0
        
        totals_g = np.sum(prob_vec_g, axis=1)
        has_prob = totals_g > 0
        
        # Case: total > 0 (probabilistic walk)
        if np.any(has_prob):
            hp_idx = np.where(has_prob)[0]
            # Probabilistic selection for those with total > 0
            # rand_vals * total
            r_vals_hp = np.random.random(len(hp_idx)) * totals_g[hp_idx]
            # cumsum to find interval
            slices_hp = np.cumsum(prob_vec_g[hp_idx], axis=1)
            # Find first index where slices >= rand_val
            # (slices < rand_val).sum(axis=1) gives the index
            choices_hp = np.sum(slices_hp < r_vals_hp[:, None], axis=1)
            # Clip choices to handle floating point edge cases (though sum should be safe)
            choices_hp = np.minimum(choices_hp, k_vals_g[hp_idx] - 1)
            
            chosen_nodes = neighbors_g[hp_idx, choices_hp]
            next_nodes[g_indices[hp_idx]] = chosen_nodes
            is_greedy[g_indices[hp_idx]] = True
        
        # Case: total == 0 (greedy failed -> fall back to random for these specific particles)
        no_prob = ~has_prob
        if np.any(no_prob):
            np_idx = np.where(no_prob)[0]
            rand_choices = (np.random.random(len(np_idx)) * k_vals_g[np_idx]).astype(np.int64)
            next_nodes[g_indices[np_idx]] = neighbors_g[np_idx, rand_choices]

    # 4. Process Random Walks
    if np.any(random_mask):
        r_indices = np.where(random_mask)[0]
        cur_nodes_r = part_curnode[r_indices]
        k_vals_r = neib_qt[cur_nodes_r]
        
        # Pick random neighbor index
        rand_choices = (np.random.random(len(r_indices)) * k_vals_r).astype(np.int64)
        # Efficiently extract selected neighbors
        # We can't easily 2D index if rows have different k, but here next_nodes 
        # is just neib_list[cur_node, choice]
        next_nodes[r_indices] = neib_list[cur_nodes_r, rand_choices]

    # 5. Dominance Update
    # Only update for particles on unlabeled nodes
    safe_next_nodes = np.where(valid_mask, next_nodes, 0)
    update_mask = (labels[safe_next_nodes] == -1) & valid_mask
    if np.any(update_mask):
        upd_idx = np.where(update_mask)[0]
        nodes_upd = next_nodes[upd_idx]
        p_labels_upd = part_label[upd_idx]
        p_strengths_upd = part_strength[upd_idx]
        
        # Aggregate simultaneous influence at each node. The winning class
        # receives the mass removed from competing classes; own-class
        # influence does not subtract from itself.
        steps = p_strengths_upd * (delta_v / (c - 1))
        influence = np.zeros((n_nodes, c), dtype=np.float64)
        np.add.at(influence, (nodes_upd, p_labels_upd), steps)
        total = influence.sum(axis=1, keepdims=True)
        # Each class can lose at most its existing dominance. Competing
        # influences act simultaneously, with conservation of total mass.
        loss = np.minimum(dominance, np.maximum(total - influence, 0.0))
        gain = loss.sum(axis=1, keepdims=True)
        fractions = np.divide(influence, total, out=np.zeros_like(influence),
                              where=total > 0)
        dominance -= loss
        dominance += gain * fractions

    # 6. Strength Update
    # Update strength based on the (potentially updated) dominance at next_node
    # part_strength[i] = dom[next_nodes[i], p_label[i]]
    new_dom_vals = dominance[safe_next_nodes, part_label]
    if is_deltap_one:
        part_strength[valid_mask] = new_dom_vals[valid_mask]
    else:
        part_strength[valid_mask] += (new_dom_vals[valid_mask] - part_strength[valid_mask]) * deltap

    # 7. Distance Table Update
    # next_d = min(next_d, cur_d + 1)
    cur_dist = dist_table[np.where(valid_mask, part_curnode, 0), np.arange(n_particles)]
    next_dist = dist_table[safe_next_nodes, np.arange(n_particles)]
    
    mask_dist = valid_mask & (cur_dist < 255) & (next_dist > cur_dist + 1)
    if np.any(mask_dist):
        dist_table[next_nodes[mask_dist], np.arange(n_particles)[mask_dist]] = cur_dist[mask_dist] + 1

    # 8. Own Degree Update (for non-greedy moves)
    # only for random walks OR greedy fallbacks that became random
    owndeg_mask = valid_mask & (~is_greedy)
    if np.any(owndeg_mask):
        od_idx = np.where(owndeg_mask)[0]
        np.add.at(owndeg, (next_nodes[od_idx], part_label[od_idx]), part_strength[od_idx])

    # 9. Movement (Shock Check)
    # Move particle only if its class is (now) the maximal one at next_node
    # We do a tie-break or just compare with max.
    max_dom_at_next = np.max(dominance[safe_next_nodes, :], axis=1)
    is_max = (new_dom_vals == max_dom_at_next)
    
    # Update positions
    part_curnode[valid_mask & is_max] = next_nodes[valid_mask & is_max]

def pcc_propagate_numpy(neib_list, neib_qt,
                        labels, p_grd, delta_v, c, zerovec,
                        part_curnode, part_label, part_strength, dist_table,
                        dominance, owndeg, deltap, dexp,
                        dom_row, reduc, dom_list, dist_list, prob, slices,
                        dist_weights,
                        max_iter, early_stop, es_chk, stop_max, update_mode="parallel"):
    
    max_mmpot = 0.0
    stop_cnt = 0
    
    for it in range(max_iter):
        pcc_step_numpy(neib_list, neib_qt,
                       labels, p_grd, delta_v, c, zerovec,
                       part_curnode, part_label, part_strength, dist_table,
                       dominance, owndeg, deltap, dexp,
                       dom_row, reduc, dom_list, dist_list, prob, slices,
                       dist_weights, update_mode=update_mode)
        
        if early_stop and it % 10 == 0:
            mmpot = np.mean(np.max(dominance, axis=1))
            if mmpot > max_mmpot:
                max_mmpot = mmpot
                stop_cnt = 0
            else:
                stop_cnt += 1
                if stop_cnt > stop_max:
                    break