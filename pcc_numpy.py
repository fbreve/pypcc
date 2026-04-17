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
    n_particles = part_curnode.shape[0]
    n_nodes = neib_list.shape[0]
    is_deltap_one = (deltap == 1.0)
    
    # 1. Validation and Setup
    valid_mask = (part_curnode >= 0) & (part_curnode < n_nodes)
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
        
        # Dominance of particle class at neighbors: dominance[neighbors, p_labels]
        # neighbors shape: (n_g, max_k_g). p_labels_g shape: (n_g,)
        # We use fancy indexing. dominance is (n_nodes, c).
        dom_vals_g = dominance[neighbors_g, p_labels_g[:, None]]
        
        # Distance weights: dist_weights[dist_table[neighbors, particle_idx]]
        # dist_table is (n_nodes, n_particles)
        # We need dist_table[neighbors[i, j], g_indices[i]]
        d_indices_g = dist_table[neighbors_g, g_indices[:, None]]
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
    update_mask = (labels[next_nodes] == -1) & valid_mask
    if np.any(update_mask):
        upd_idx = np.where(update_mask)[0]
        nodes_upd = next_nodes[upd_idx]
        p_labels_upd = part_label[upd_idx]
        p_strengths_upd = part_strength[upd_idx]
        
        # Calculate reductions: min(dominance[node, :], step)
        step = p_strengths_upd * (delta_v / (c - 1))
        dom_rows = dominance[nodes_upd, :] # (n_upd, c)
        
        reduc_vals = np.minimum(dom_rows, step[:, None])
        total_reduc = np.sum(reduc_vals, axis=1)
        
        # Apply updates using np.add.at/subtract.at to handle node collisions
        # subtract reduction from all classes
        np.subtract.at(dominance, (nodes_upd[:, None], np.arange(c)), reduc_vals)
        # add total reduction to the particle's class
        np.add.at(dominance, (nodes_upd, p_labels_upd), total_reduc)

    # 6. Strength Update
    # Update strength based on the (potentially updated) dominance at next_node
    # part_strength[i] = dom[next_nodes[i], p_label[i]]
    new_dom_vals = dominance[next_nodes, part_label]
    if is_deltap_one:
        part_strength[valid_mask] = new_dom_vals[valid_mask]
    else:
        part_strength[valid_mask] += (new_dom_vals[valid_mask] - part_strength[valid_mask]) * deltap

    # 7. Distance Table Update
    # next_d = min(next_d, cur_d + 1)
    cur_dist = dist_table[part_curnode, np.arange(n_particles)]
    next_dist = dist_table[next_nodes, np.arange(n_particles)]
    
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
    max_dom_at_next = np.max(dominance[next_nodes, :], axis=1)
    is_max = (new_dom_vals == max_dom_at_next)
    
    # Update positions
    part_curnode[valid_mask & is_max] = next_nodes[valid_mask & is_max]

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