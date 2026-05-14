# -*- coding: utf-8 -*-
"""
Graph construction utilities for Particle Competition and Cooperation (PCC)
===========================================================================

This module separates graph construction from the PCC algorithm itself,
similarly to the MATLAB functions pcc_buildgraph / pcc.

Provides:
    - build_knn_graph: k-NN graph from a feature matrix (sklearn.neighbors).
    - build_graph_from_edge_index: graph from edge_index (e.g. PyTorch Geometric).

Both functions return:
    neib_list: int64 [n_nodes, max_deg]
        Row i contains the neighbors of node i.
    neib_qt: int64 [n_nodes]
        Degree (number of neighbors) of each node.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pcc_covqdtree import covariance_qdtree_knn_indices


def _symmetrize_indices_to_neighbor_lists(indices, n_nodes):
    """Convert directed k-NN indices into symmetric PCC neighbor arrays."""
    if n_nodes == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)

    k_nn = indices.shape[1]
    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), k_nn)
    cols = indices.ravel().astype(np.int64)

    ones = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((ones, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.int8)
    adj = adj + adj.T
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1

    neib_qt = np.diff(adj.indptr).astype(np.int64)
    max_deg = int(neib_qt.max()) if n_nodes > 0 else 0

    neib_list = np.full((n_nodes, max(max_deg, 1)), -1, dtype=np.int64)
    degrees = (adj.indptr[1:] - adj.indptr[:-1]).astype(np.int64)
    row_idx = np.repeat(np.arange(n_nodes, dtype=np.int64), degrees)
    col_idx = np.concatenate(
        [np.arange(d, dtype=np.int64) for d in degrees]
    ) if n_nodes > 0 else np.empty(0, dtype=np.int64)
    if len(row_idx) > 0:
        neib_list[row_idx, col_idx] = adj.indices.astype(np.int64)

    return neib_list, neib_qt


def build_knn_graph(
    data,
    k_nn=10,
    metric="minkowski",
    p=2,
    n_jobs=None,
    nn_method="sklearn",
    qdtree_max_depth=None,
    qdtree_min_points_split=2,
):
    """
    Build a symmetric k-NN graph from a feature matrix.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input feature matrix. Each row is a data item, each column an attribute.
    k_nn : int, optional (default=10)
        Each node is connected to its k nearest neighbors.
    metric : str or callable, optional (default="minkowski")
        Distance metric passed to sklearn.neighbors.NearestNeighbors.
        Common choices:
            - "minkowski" with p=2: Euclidean distance.
            - "minkowski" with p=1: Manhattan distance.
            - Any metric supported by NearestNeighbors.
    p : int, optional (default=2)
        Power parameter for the Minkowski metric. Ignored if metric does not
        use this parameter.
    n_jobs : int, optional (default=None)
        The number of parallel jobs to run for neighbors search. None means 1.
    nn_method : str, optional (default="sklearn")
        Neighbor search method. Supported values:
            - "sklearn": uses sklearn.neighbors.NearestNeighbors
            - "covariance_qdtree" / "covariance-qdtree": uses a
              covariance-qdtree-inspired top-down search.
    qdtree_max_depth : int, optional
        Maximum tree depth for covariance-qdtree method.
    qdtree_min_points_split : int, optional (default=2)
        Minimum number of points required to split a node for covariance-qdtree.

    Returns
    -------
    neib_list : ndarray, shape (n_nodes, max_deg), dtype=int64
        Matrix of neighbors indices for each node. Row i contains the neighbors
        of node i. Columns beyond neib_qt[i] are undefined.
    neib_qt : ndarray, shape (n_nodes,), dtype=int64
        Degree (number of neighbors) of each node.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array of shape (n_samples, n_features).")

    n_nodes = data.shape[0]

    if not isinstance(k_nn, int) or k_nn < 1:
        raise ValueError("k_nn must be a positive integer.")

    if k_nn >= n_nodes:
        raise ValueError(
            f"k_nn ({k_nn}) must be smaller than the number of samples ({n_nodes})."
        )

    method = str(nn_method).lower()
    if method == "sklearn":
        nbrs = NearestNeighbors(
            n_neighbors=k_nn + 1,
            algorithm="auto",
            metric=metric,
            p=p,
            n_jobs=n_jobs,
        ).fit(data)
        indices = nbrs.kneighbors(data, return_distance=False)[:, 1:]
    elif method in ("covariance_qdtree", "covariance-qdtree"):
        # The qdtree method currently uses squared Euclidean distance.
        if metric != "minkowski" or p != 2:
            raise ValueError(
                "covariance-qdtree currently supports only Euclidean metric "
                "(metric='minkowski', p=2)."
            )
        indices = covariance_qdtree_knn_indices(
            data,
            k_nn=k_nn,
            max_depth=qdtree_max_depth,
            min_points_split=qdtree_min_points_split,
        )
    else:
        raise ValueError(
            "nn_method must be one of {'sklearn', 'covariance_qdtree', 'covariance-qdtree'}."
        )

    return _symmetrize_indices_to_neighbor_lists(indices, n_nodes)


def build_graph_from_edge_index(num_nodes, edge_index):
    """
    Build a graph from a pre-defined edge list (edge_index format).

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.
    edge_index : array-like, shape (2, num_edges)
        Edge list as used by PyTorch Geometric:
        edge_index[0, e] = source node of edge e
        edge_index[1, e] = target node of edge e

    Returns
    -------
    neib_list : ndarray, shape (n_nodes, max_deg), dtype=int64
        Matrix of neighbors indices for each node.
    neib_qt : ndarray, shape (n_nodes,), dtype=int64
        Degree (number of neighbors) of each node.

    Notes
    -----
    - Assumes an undirected graph and adds reciprocal connections
      by construction (i.e., both (u,v) and (v,u)).
    - Symmetrization and deduplication are done via a sparse adjacency
      matrix, avoiding Python-level loops over edges.
    - Self-loops present in edge_index are silently removed.
    - All neighbor edges are preserved; no degree cap is applied.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    if num_nodes == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)

    src = edge_index[0]
    dst = edge_index[1]

    # Concatenate forward and backward edges for undirected symmetrization
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])

    ones = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((ones, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.int8)
    adj.setdiag(0)      # remove self-loops
    adj.eliminate_zeros()

    # Deduplicate: keep only binary (0/1) entries
    adj.data[:] = 1

    neib_qt = np.diff(adj.indptr).astype(np.int64)
    max_deg = int(neib_qt.max()) if num_nodes > 0 else 0

    neib_list = np.full((num_nodes, max(max_deg, 1)), -1, dtype=np.int64)
    # Vectorized fill: same CSR-based scatter as build_knn_graph.
    degrees2 = (adj.indptr[1:] - adj.indptr[:-1]).astype(np.int64)
    row_idx2 = np.repeat(np.arange(num_nodes, dtype=np.int64), degrees2)
    col_idx2 = np.concatenate(
        [np.arange(d, dtype=np.int64) for d in degrees2]
    ) if num_nodes > 0 else np.empty(0, dtype=np.int64)
    if len(row_idx2) > 0:
        neib_list[row_idx2, col_idx2] = adj.indices.astype(np.int64)

    return neib_list, neib_qt