"""Covariance-qdtree k-NN search utilities.

This module provides a pure-Python/NumPy implementation of the
covariance-qdtree strategy used in the original C reference implementation
(`covariancekNN.c`, `kNN.Mahalanobis.c`, `eigenvals.c` in KNN-based_cluster_analysis).
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
from typing import List, Optional, Sequence

import numpy as np


class Polytope:
    """Represents a hyperplane boundary for a node."""
    def __init__(self, vertex: np.ndarray, normal: np.ndarray):
        self.vertex = vertex
        self.normal = normal


@dataclass
class _QDTNode:
    parent: int
    vertex: np.ndarray
    children: List[int]
    points: np.ndarray
    is_leaf: bool
    eigenvec: np.ndarray
    eigenval: np.ndarray
    intrinsic_dim: int
    hyperfaces: List[Polytope]


def _build_hyperqdtree(
    data: np.ndarray,
    max_depth: Optional[int] = None,
    min_points_split: int = 2,
) -> List[_QDTNode]:
    """Build a covariance-subspace quadtree hierarchy.

    Each internal node splits points by their sign patterns when projected onto
    the principal components (eigenvectors of the node's covariance matrix).
    """
    n_nodes, dim = data.shape
    if n_nodes == 0:
        return []

    if max_depth is None:
        # Keeps tree depth bounded for high-dimensional cases.
        max_depth = int(np.ceil(np.log2(max(n_nodes, 2)))) + 1

    nodes: List[_QDTNode] = []

    # Prepare root node
    root_pts = np.arange(n_nodes, dtype=np.int64)
    root_mean = data.mean(axis=0)
    if n_nodes > 1:
        Sigma = np.cov(data, rowvar=False, ddof=0)
        eigenvals, eigenvecs = np.linalg.eigh(Sigma)
        idx_sorted = np.argsort(eigenvals)[::-1]
        eigenvals = np.maximum(eigenvals[idx_sorted], 0.0)
        eigenvecs = eigenvecs[:, idx_sorted].T
        intrinsic_dim = int(np.sum(eigenvals > 1e-15))
        if intrinsic_dim == 0:
            intrinsic_dim = 1
    else:
        eigenvals = np.zeros(dim, dtype=np.float64)
        eigenvecs = np.eye(dim, dtype=np.float64)
        intrinsic_dim = dim

    root = _QDTNode(
        parent=-1,
        vertex=root_mean,
        children=[],
        points=root_pts,
        is_leaf=False,
        eigenvec=eigenvecs,
        eigenval=eigenvals,
        intrinsic_dim=intrinsic_dim,
        hyperfaces=[],
    )
    nodes.append(root)

    stack = [(0, 0)]  # (node_id, depth)
    while stack:
        node_id, depth = stack.pop()
        node = nodes[node_id]
        pts = node.points

        if pts.size <= 1 or pts.size < min_points_split or depth >= max_depth:
            node.is_leaf = True
            continue

        center = node.vertex
        rel = data[pts] - center

        # Project points onto node's principal components
        projections = rel @ node.eigenvec[:node.intrinsic_dim].T
        bits = (projections > 0).astype(np.int64)
        weights = 1 << np.arange(node.intrinsic_dim - 1, -1, -1, dtype=np.int64)
        codes = (bits * weights).sum(axis=1)

        unique_codes = np.unique(codes)

        # Unsplittable/singular partition: keep as leaf bucket.
        if unique_codes.size <= 1:
            node.is_leaf = True
            continue

        node.children = []
        node.is_leaf = False
        for code in unique_codes:
            child_pts = pts[codes == code]
            child_data = data[child_pts]
            child_mean = child_data.mean(axis=0)

            if child_pts.size > 1:
                child_Sigma = np.cov(child_data, rowvar=False, ddof=0)
                c_vals, c_vecs = np.linalg.eigh(child_Sigma)
                idx_sorted = np.argsort(c_vals)[::-1]
                c_vals = np.maximum(c_vals[idx_sorted], 0.0)
                c_vecs = c_vecs[:, idx_sorted].T
                child_intrinsic_dim = int(np.sum(c_vals > 1e-15))
                if child_intrinsic_dim == 0:
                    child_intrinsic_dim = 1
            else:
                c_vals = np.zeros(dim, dtype=np.float64)
                c_vecs = np.eye(dim, dtype=np.float64)
                child_intrinsic_dim = dim

            child = _QDTNode(
                parent=node_id,
                vertex=child_mean,
                children=[],
                points=child_pts,
                is_leaf=False,
                eigenvec=c_vecs,
                eigenval=c_vals,
                intrinsic_dim=child_intrinsic_dim,
                hyperfaces=[],
            )
            child_id = len(nodes)

            # Setup child boundary (analogous to child_boundary in C)
            # 1. division walls from parent
            for l in range(node.intrinsic_dim):
                normdir = -1.0 if ((code >> (node.intrinsic_dim - 1 - l)) % 2) else 1.0
                normal = normdir * node.eigenvec[l]
                child.hyperfaces.append(Polytope(vertex=node.vertex.copy(), normal=normal))

            # 2. inherit ancestor walls
            if len(node.hyperfaces) > 0:
                n_child_faces = len(child.hyperfaces)
                registered = np.zeros(len(node.hyperfaces), dtype=np.int32)
                minimal = np.full(n_child_faces, -1, dtype=np.int32)
                to_append = []
                for l in range(n_child_faces):
                    mintilt = 0.0
                    mindist = 0.0
                    minimal[l] = -1
                    for k in range(len(node.hyperfaces)):
                        r_lk = node.hyperfaces[k].vertex - child.hyperfaces[l].vertex
                        d_lk_l_directed = np.dot(r_lk, child.hyperfaces[l].normal)
                        tilt = np.dot(child.hyperfaces[l].normal, node.hyperfaces[k].normal)
                        d_lk_l = d_lk_l_directed * tilt

                        if k == 0:
                            mindist = d_lk_l
                            minimal[l] = k
                            mintilt = min(tilt, mintilt)
                        elif tilt < mintilt and registered[k] == 0:
                            mintilt = tilt
                            if d_lk_l > mindist:
                                mindist = d_lk_l
                                minimal[l] = k

                    if mintilt < 0.0 and minimal[l] > -1 and registered[minimal[l]] == 0:
                        registered[minimal[l]] = 1
                        to_append.append(node.hyperfaces[minimal[l]])
                child.hyperfaces.extend(to_append)

            nodes.append(child)
            node.children.append(child_id)
            stack.append((child_id, depth + 1))

    return nodes


def Q2NDistance(
    query: np.ndarray,
    node: _QDTNode,
    nodes: Sequence[_QDTNode],
    distmax: float,
    eigenvec: np.ndarray,
    eigenval: np.ndarray,
    knn_intrinsic_dim: int,
) -> float:
    """Compute query-to-node boundary distance."""
    if node.parent == -1:
        return 0.0

    parent = nodes[node.parent]
    nfaces = parent.intrinsic_dim
    l_max = -1

    for l in range(nfaces):
        normal_projection = np.dot(node.hyperfaces[l].normal, query - node.hyperfaces[l].vertex)
        if normal_projection > 0:
            np2 = normal_projection * normal_projection
            if np2 > distmax:
                distmax = np2
                l_max = l

    if l_max > -1:
        dist = 0.0
        normal_lmax = node.hyperfaces[l_max].normal
        eigenval_safe = np.maximum(eigenval, 1e-15)
        for l in range(knn_intrinsic_dim):
            proj = np.dot(normal_lmax, eigenvec[l])
            dist += distmax * distmax * (proj * proj) / eigenval_safe[l]
        return dist
    return distmax


def Q2PDistance(
    query: np.ndarray,
    point: np.ndarray,
    eigenvec: np.ndarray,
    eigenval: np.ndarray,
    knn_intrinsic_dim: int,
) -> float:
    """Compute query-to-point distance in Mahalanobis space."""
    diff = query - point
    proj = eigenvec[:knn_intrinsic_dim] @ diff
    eigenval_safe = np.maximum(eigenval[:knn_intrinsic_dim], 1e-15)
    d = np.sum((proj ** 2) / eigenval_safe)
    return d


class KNNSearchState:
    """Maintains the K-nearest neighbors candidates in sorted order."""
    def __init__(self, k: int, q: int):
        self.k = k
        self.q = q
        self.candidates: List[tuple[float, int]] = []
        self.top_dist = 0.0

    def insert(self, point_idx: int, dist: float):
        if dist == 0.0:
            return
        if point_idx == self.q:
            return

        # Insert in sorted order
        dists = [c[0] for c in self.candidates]
        idx = bisect.bisect_left(dists, dist)
        self.candidates.insert(idx, (dist, point_idx))
        if len(self.candidates) > self.k:
            self.candidates.pop()

        if len(self.candidates) == self.k:
            self.top_dist = self.candidates[-1][0]


def _search_knn(
    node_id: int,
    nodes: Sequence[_QDTNode],
    query: np.ndarray,
    q2nd: float,
    eigenvec: np.ndarray,
    eigenval: np.ndarray,
    knn_intrinsic_dim: int,
    state: KNNSearchState,
    data: np.ndarray,
):
    """Recursive k-NN tree traversal with branch pruning."""
    node = nodes[node_id]

    # Update query-to-node distance
    q2nd = Q2NDistance(query, node, nodes, q2nd, eigenvec, eigenval, knn_intrinsic_dim)

    # Prune branch
    if len(state.candidates) == state.k and q2nd > state.top_dist:
        return

    if node.is_leaf:
        for p in node.points:
            q2pd = Q2PDistance(query, data[p], eigenvec, eigenval, knn_intrinsic_dim)
            if len(state.candidates) == state.k and q2pd > state.top_dist:
                continue
            state.insert(p, q2pd)
    else:
        for child_id in node.children:
            _search_knn(child_id, nodes, query, q2nd, eigenvec, eigenval, knn_intrinsic_dim, state, data)


def _knn_one_query(
    data: np.ndarray,
    nodes: Sequence[_QDTNode],
    q: int,
    k: int,
    tolerance: float = 1e-12,
    max_iterations: int = 16,
) -> np.ndarray:
    """Anisotropic (direction-adaptive) k-NN search refinement loop."""
    dim = data.shape[1]
    query = data[q]

    # Initialize metric (Euclidean metric: identity eigenvectors, unit eigenvalues)
    eigenvec = np.eye(dim, dtype=np.float64)
    eigenval = np.ones(dim, dtype=np.float64)
    knn_intrinsic_dim = dim

    state = KNNSearchState(k, q)

    # Refinement loop
    for itime in range(max_iterations):
        state.candidates = []
        state.top_dist = 0.0

        # Perform kNN search on current metric
        _search_knn(0, nodes, query, 0.0, eigenvec, eigenval, knn_intrinsic_dim, state, data)

        if len(state.candidates) < k:
            break

        if tolerance < 1.0:
            # Check convergence and update metric using covariance of neighbors
            neighbors = [c[1] for c in state.candidates]
            X_neighbors = data[neighbors]
            Sigma = np.cov(X_neighbors, rowvar=False, ddof=0)

            c_vals, c_vecs = np.linalg.eigh(Sigma)
            idx_sorted = np.argsort(c_vals)[::-1]
            c_vals = np.maximum(c_vals[idx_sorted], 0.0)
            c_vecs = c_vecs[:, idx_sorted].T

            # Condition to check convergence
            notconverged = (eigenval[0] < 2.0 * eigenval[1]) if dim > 1 else False
            if notconverged:
                principal = eigenval[0]
                eigenval = c_vals
                eigenvec = c_vecs
                knn_intrinsic_dim = int(np.sum(eigenval > 1e-15))
                if knn_intrinsic_dim == 0:
                    knn_intrinsic_dim = 1

                notconverged = abs(principal - eigenval[0]) > tolerance * (principal + eigenval[0])
                if not notconverged:
                    # Converged: do a final query with new metric
                    state.candidates = []
                    state.top_dist = 0.0
                    _search_knn(0, nodes, query, 0.0, eigenvec, eigenval, knn_intrinsic_dim, state, data)
                    break
            else:
                break
        else:
            break

    if len(state.candidates) < k:
        # Fallback safety in degenerate partitions.
        d2_all = np.sum((data - query) ** 2, axis=1)
        d2_all[q] = np.inf
        idx = np.argpartition(d2_all, kth=k - 1)[:k]
        d2 = d2_all[idx]
        return idx[np.argsort(d2)].astype(np.int64)

    return np.array([idx for _, idx in state.candidates], dtype=np.int64)


def covariance_qdtree_knn_indices(
    data: np.ndarray,
    k_nn: int,
    max_depth: Optional[int] = None,
    min_points_split: int = 2,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return k-NN indices matrix using covariance-qdtree-inspired search.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
    k_nn : int
        Number of nearest neighbors per sample (excluding self).
    max_depth : int, optional
        Maximum tree depth.
    min_points_split : int, optional
        Minimum points required for a node to be split.
    tolerance : float, optional (default=1e-12)
        Convergence threshold for direction-adaptive refinement.

    Returns
    -------
    indices : ndarray, shape (n_samples, k_nn), dtype=int64
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array of shape (n_samples, n_features).")

    n_samples = data.shape[0]
    if n_samples < 2:
        raise ValueError("At least 2 samples are required to build a k-NN graph.")
    if not isinstance(k_nn, int) or k_nn < 1:
        raise ValueError("k_nn must be a positive integer.")
    if k_nn >= n_samples:
        raise ValueError(
            f"k_nn ({k_nn}) must be smaller than the number of samples ({n_samples})."
        )

    nodes = _build_hyperqdtree(
        data,
        max_depth=max_depth,
        min_points_split=max(2, int(min_points_split)),
    )

    indices = np.empty((n_samples, k_nn), dtype=np.int64)
    for q in range(n_samples):
        indices[q] = _knn_one_query(data, nodes, q, k_nn, tolerance=tolerance)

    return indices
