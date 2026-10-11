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
                   dominance, owndeg, deltap=1.0, dexp=2.0,
                   dom_row=None, reduc=None, dom_list=None, dist_list=None, prob=None, slices=None,
                   dist_weights=None, update_mode="sequential"):
    """Run sequential PCC (default) or the experimental synchronous variant."""
    if update_mode == "synchronous":
        return _pcc_step_numpy_synchronous(
            neib_list, neib_qt, labels, p_grd, delta_v, c,
            part_curnode, part_label, part_strength, dist_table,
            dominance, owndeg, deltap, dexp, dist_weights)
    if update_mode != "sequential":
        raise ValueError("update_mode must be 'sequential' or 'synchronous'")
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
            valid = (neighbors >= 0) & (neighbors < n_nodes)
            weights = np.zeros(degree, dtype=np.float64)
            targets = neighbors[valid]
            weights[valid] = dominance[targets, cls] * dist_weights[dist_table[targets, p_i]]
            total = float(np.sum(weights))
            if total > 0.0:
                threshold = np.random.random() * total
                idx = int(np.searchsorted(np.cumsum(weights), threshold, side="left"))
                idx = min(idx, degree - 1)
            else:
                idx = int(np.random.random() * degree)
                greedy = False
        else:
            idx = int(np.random.random() * degree)

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

        class_dom = dominance[nxt, cls]
        max_dom = np.max(dominance[nxt, :])
        accepted = class_dom == max_dom
        if accepted:
            part_curnode[p_i] = nxt

def _pcc_step_numpy_synchronous(neib_list, neib_qt, labels, p_grd, delta_v, c,
                                part_curnode, part_label, part_strength, dist_table,
                                dominance, owndeg, deltap, dexp, dist_weights):
    """Select from the initial state, aggregate, then commit all particle updates.

    Each donor class loses at most its initial dominance; its loss is shared
    only by competing visiting classes, in proportion to their influence.
    See docs/numpy_synchronous.md for the equations and experimental semantics.
    """
    n_nodes = neib_list.shape[0]
    if n_nodes == 0 or part_curnode.size == 0:
        return
    valid = (part_curnode >= 0) & (part_curnode < n_nodes)
    valid &= (part_label >= 0) & (part_label < c)
    valid &= neib_qt[np.clip(part_curnode, 0, n_nodes - 1)] > 0
    particles = np.flatnonzero(valid)
    if particles.size == 0:
        return
    if dist_weights is None:
        dist_weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** dexp

    # No shared state is modified until every destination has been selected.
    current = part_curnode[particles]
    classes = part_label[particles]
    degrees = neib_qt[current]
    greedy = np.random.random(particles.size) < p_grd
    choices = (np.random.random(particles.size) * degrees).astype(np.int64)
    g = np.flatnonzero(greedy)
    if g.size:
        neighbors = neib_list[current[g], :int(degrees[g].max())]
        mask = np.arange(neighbors.shape[1]) < degrees[g, None]
        mask &= (neighbors >= 0) & (neighbors < n_nodes)
        safe = np.where(mask, neighbors, 0)
        weights = dominance[safe, classes[g, None]] * dist_weights[
            dist_table[safe, particles[g, None]]]
        weights[~mask] = 0.0
        cumulative = np.cumsum(weights, axis=1)
        totals = cumulative[:, -1]
        positive = totals > 0.0
        selected = g[positive]
        thresholds = np.random.random(selected.size) * totals[positive]
        indices = np.sum(cumulative[positive] <= thresholds[:, None], axis=1)
        choices[selected] = np.minimum(indices, degrees[selected] - 1)
        greedy[g[~positive]] = False

    targets = neib_list[current, choices]
    valid_target = (targets >= 0) & (targets < n_nodes)
    particles = particles[valid_target]
    current = current[valid_target]
    classes = classes[valid_target]
    targets = targets[valid_target]
    greedy = greedy[valid_target]
    if particles.size == 0:
        return

    unlabeled = labels[targets] == -1
    if np.any(unlabeled):
        nodes, inverse = np.unique(targets[unlabeled], return_inverse=True)
        influence = np.zeros((nodes.size, c), dtype=np.float64)
        steps = part_strength[particles[unlabeled]] * (delta_v / (c - 1))
        np.add.at(influence, (inverse, classes[unlabeled]), steps)
        competitors = np.ones((c, c), dtype=np.float64) - np.eye(c)
        pressure = influence @ competitors
        initial = dominance[nodes]
        loss = np.minimum(initial, pressure)
        ratios = np.divide(loss, pressure, out=np.zeros_like(loss), where=pressure > 0)
        gain = influence * (ratios @ competitors)
        updated = initial - loss + gain
        # Roundoff cleanup only; the transfer rule itself conserves the simplex.
        np.clip(updated, 0.0, 1.0, out=updated)
        updated /= updated.sum(axis=1, keepdims=True)
        dominance[nodes] = updated

    # Everyone observes the same committed dominance, without per-particle writes.
    own = dominance[targets, classes]
    if deltap == 1.0:
        part_strength[particles] = own
    else:
        part_strength[particles] += (own - part_strength[particles]) * deltap

    # Cast before adding 1 so uint8 distances cannot wrap at 255.
    before = dist_table[current, particles].astype(np.int64)
    after = dist_table[targets, particles].astype(np.int64)
    shorten = (before < 255) & (after > before + 1)
    dist_table[targets[shorten], particles[shorten]] = before[shorten] + 1
    random = ~greedy
    np.add.at(owndeg, (targets[random], classes[random]), part_strength[particles[random]])
    accepted = own == np.max(dominance[targets], axis=1)
    part_curnode[particles[accepted]] = targets[accepted]


def pcc_propagate_numpy(neib_list, neib_qt,
                        labels, p_grd, delta_v, c, zerovec,
                        part_curnode, part_label, part_strength, dist_table,
                        dominance, owndeg, deltap, dexp,
                        dom_row, reduc, dom_list, dist_list, prob, slices,
                        dist_weights,
                        max_iter, early_stop, es_chk, stop_max, update_mode="sequential"):
    if update_mode not in ("sequential", "synchronous"):
        raise ValueError("update_mode must be 'sequential' or 'synchronous'")
    
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
