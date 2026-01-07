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
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(data, k_nn=10):
    """
    Build a symmetric k-NN graph from a feature matrix.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input feature matrix. Each row is a data item, each column an attribute.
    k_nn : int, optional (default=10)
        Each node is connected to its k nearest neighbors.

    Returns
    -------
    neib_list : ndarray, shape (n_nodes, max_deg), dtype=int64
        Matrix of neighbors indices for each node. Row i contains the neighbors
        of node i. Columns beyond neib_qt[i] are undefined.
    neib_qt : ndarray, shape (n_nodes,), dtype=int64
        Degree (number of neighbors) of each node.

    Notes
    -----
    - Uses sklearn.neighbors.NearestNeighbors, which chooses the most efficient
      method to find the neighbors (usually not brute force).
    - The graph is symmetrized by adding reciprocal connections.
    """
    nbrs = NearestNeighbors(
        n_neighbors=k_nn + 1,
        algorithm="auto",
        n_jobs=-1,
    ).fit(data)

    # indices of neighbors (including self as first neighbor)
    neib_list = nbrs.kneighbors(data, return_distance=False)
    # discard self
    neib_list = neib_list[:, 1:]
    neib_list = neib_list.astype(np.int64)

    qt_node = neib_list.shape[0]

    # initial amount of neighbors (k) per node
    neib_qt = np.full(qt_node, k_nn, dtype=np.int64)

    # current number of allocated columns
    ind_cols = k_nn

    # add reciprocal connections
    for i in range(qt_node):
        for j in range(k_nn):
            target = neib_list[i, j]
            # if there is no space, increase number of columns
            if neib_qt[target] == ind_cols:
                new_cols_qt = round(ind_cols * 0.2) + 1
                extra = np.empty((qt_node, new_cols_qt), dtype=np.int64)
                neib_list = np.append(neib_list, extra, axis=1)
                ind_cols += new_cols_qt

            neib_list[target, neib_qt[target]] = i
            neib_qt[target] += 1

    # remove duplicate neighbors for each node
    for i in range(qt_node):
        k_i = neib_qt[i]
        unique = np.unique(neib_list[i, :k_i])
        neib_qt[i] = unique.shape[0]
        neib_list[i, :neib_qt[i]] = unique

    # discard unused last columns
    ind_cols = int(np.max(neib_qt))
    neib_list = neib_list[:, :ind_cols]

    return neib_list, neib_qt


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
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    neighbors = [[] for _ in range(num_nodes)]

    # add edges as undirected
    for src, dst in edge_index.T:
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    degrees = np.array([len(nei) for nei in neighbors], dtype=np.int64)
    max_deg = int(degrees.max()) if num_nodes > 0 else 0

    neib_list = np.full((num_nodes, max_deg), -1, dtype=np.int64)
    for i, nei in enumerate(neighbors):
        if len(nei) > 0:
            unique = np.unique(np.array(nei, dtype=np.int64))
            neib_list[i, :len(unique)] = unique
            degrees[i] = len(unique)
        else:
            degrees[i] = 0

    neib_qt = degrees
    return neib_list, neib_qt
