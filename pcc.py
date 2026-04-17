# -*- coding: utf-8 -*-
"""
Semi-Supervised Learning with Particle Competition and Cooperation
==================================================================

If you use this algorithm, please cite:
Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming,
"Particle Competition and Cooperation in Networks for Semi-Supervised Learning,"
Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012
doi: 10.1109/TKDE.2011.119

Fixed:
 1. Graph was generating dictonaries with int and int64 numbers.
 2. dict/list access was too slow, changed to matrix.
 3. Bug in GreedyWalk, probability vector was being normalized but
    the vector sum wasn't adjusted to 1.
 4. Particles' strength were kept in a int matrix.
 5. Node labels were kept in a float matrix.

Changes:
 1. Changed __genGraph() to use sklearn.neighbors, which chooses the most
    efficient method to find the neighbors (usually not brute force).
 2. Nearest Neighbors lists now use int32 instead of the mix of int/int64/uint32.
 3. Nearest Neighbors list is now a matrix instead of dict/list
    (the matrix access is much more efficient).
 4. GreedyWalk now avoid loops (vectorization) and has some other tweaks
    to drammatically improve speed.
 5. Changed int type to uint8 in the distance table (saves memory space at
    a small computational cost, check comments)
 6. Changed the particle matrix into a particle class to fix the particle
    strength being held in a int type matrix.
 7. Changed the nodes matrix into a nodes class to keep labels and dominance
    levels separated and with proper data types.
 8. Removed class_map to avoid unnecessary overhead (though it is small).
    This kind of treatment could be reimplemented to be active only twice:
    on input and on output.
 9. Eliminated the loop in the 'labeling the unlabeled nodes' step.
10. Added the early stop criteria.
11. Moved execution time evaluation to the example.
12. Vectorized the update() method and did some other tweaks to improve
    speed.
13. Moved the distance table to the particle attributes.
14. Changed fit/predict to a single fit_predict method, as it makes more sense
    for a transductive SSL method. graph_gen() is available as a separate
    function since one may want to run multiples executions with the same graph.

Note:
 1. Node dominances and particle strenght are using float64 because it is a
    little faster than float32, though we don't really need 64 bits precision
"""

# pcc.py
import numpy as np
import warnings

from dataclasses import dataclass
from pcc_numpy import pcc_step_numpy
from pcc_graph import build_knn_graph

try:
    from pcc_numba import pcc_step_numba
    _HAS_NUMBA = True
except ImportError:
    pcc_step_numba = None
    _HAS_NUMBA = False

try:
    from pcc_step import pcc_step as pcc_step_cython
    _HAS_CYTHON = True
except ImportError:
    pcc_step_cython = None
    _HAS_CYTHON = False


@dataclass
class Particles:
    homenode: np.ndarray
    curnode: np.ndarray
    label: np.ndarray
    strength: np.ndarray
    amount: int
    dist_table: np.ndarray


@dataclass
class Nodes:
    amount: int
    dominance: np.ndarray
    label: np.ndarray


class ParticleCompetitionAndCooperation:

    def __init__(self, impl="auto", n_jobs=None):
        """
        impl: 'auto', 'cython', 'numba' ou 'numpy'
        """
        self.impl = impl
        self.n_jobs = n_jobs
        self.data = None
        self.k_nn = None
        self.neib_list = None
        self.neib_qt = None
        self.labels = None
        self.unique_labels = None
        self.c = None
        self.p_grd = None
        self.delta_v = None
        self.deltap = None
        self.dexp = None
        self.max_iter = None
        self.early_stop = None
        self.es_chk = None
        self.part = None
        self.node = None
        self.zerovec = None
        self.owndeg = None
        self.label_to_idx = None
        self.idx_to_label = None

    def _get_backend_fn(self):
        """
        Determines the appropriate propagation function based on the requested implementation and availability.
        """
        impl = self.impl.lower() if isinstance(self.impl, str) else "auto"
        if impl not in ("cython", "numba", "numpy", "auto"):
            warnings.warn(f"Backend '{self.impl}' is invalid; using 'auto' (cython→numba→numpy).")
            impl = "auto"

        backends = []
        if impl == "cython":
            backends = ["cython", "numba", "numpy"]
        elif impl == "numba":
            backends = ["numba", "cython", "numpy"]
        elif impl == "numpy":
            backends = ["numpy"]
        else:  # auto
            backends = ["cython", "numba", "numpy"]

        for be in backends:
            if be == "cython" and _HAS_CYTHON:
                try:
                    from pcc_step import pcc_propagate as propagate_cython
                    return propagate_cython
                except ImportError:
                    pass
            if be == "numba" and _HAS_NUMBA:
                try:
                    from pcc_numba import pcc_propagate_numba
                    return pcc_propagate_numba
                except ImportError:
                    pass
            if be == "numpy":
                try:
                    from pcc_numpy import pcc_propagate_numpy
                    return pcc_propagate_numpy
                except ImportError:
                    pass
        
        raise RuntimeError("No backend available: cython, numba, or numpy.")

    def build_graph(self, data, k_nn=10, n_jobs=None):
        self.data = data
        self.k_nn = k_nn
        self.neib_list, self.neib_qt = build_knn_graph(
            data, k_nn, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
        )

    def set_graph(self, neib_list, neib_qt, k_nn=None):
        self.neib_list = neib_list.astype(np.int64)
        self.neib_qt = neib_qt.astype(np.int64)
        self.k_nn = int(k_nn) if k_nn is not None else int(neib_qt.max())
        self.data = None

    def fit_predict(self, labels, p_grd=0.5, delta_v=0.1, deltap=1.0, dexp=2.0,
                    max_iter=500000, early_stop=True, es_chk=2000):

        if self.neib_list is None or self.neib_qt is None:
            print("Error: build or set the graph first.")
            return -1

        self.labels = np.ascontiguousarray(labels, dtype=np.int64)
        
        # Internal label mapping: maps original labels to 0..c-1
        self.unique_labels = np.unique(self.labels)
        self.unique_labels = self.unique_labels[self.unique_labels != -1]
        self.c = len(self.unique_labels)
        
        if self.c < 2:
            warnings.warn("PCC requires at least 2 labeled classes to run. Returning original labels.")
            return labels
        
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.unique_labels)}
        self.idx_to_label = {i: lbl for i, lbl in enumerate(self.unique_labels)}
        
        # Create a mapped labels array for internal use
        # Optimization: Use Fortran order for tables often indexed by [neighbors, column]
        # This makes the columns contiguous in memory (better cache locality)
        self.mapped_labels = np.full(self.labels.shape, -1, dtype=np.int64)
        for lbl, idx in self.label_to_idx.items():
            self.mapped_labels[self.labels == lbl] = idx

        self.p_grd = float(p_grd)
        self.delta_v = float(delta_v)
        self.deltap = float(deltap)
        self.dexp = float(dexp)
        self.max_iter = int(max_iter)
        self.early_stop = bool(early_stop)
        self.es_chk = int(es_chk)

        self.part = self.__genParticles()
        self.node = self.__genNodes()

        # Pre-calculate distance weights lookup table for 0..255 range
        # This drastically speeds up the NumPy backend by avoiding power/inversion in loop
        self.dist_weights = 1.0 / (np.arange(257, dtype=np.float64) + 1.0) ** self.dexp
        # Padding to 257 to handle potential 255+1 edge cases safely
        
        self.__labelPropagation()

        # Unmap labels before returning
        final_labels = np.full(self.node.label.shape, -1, dtype=np.int64)
        for idx, lbl in self.idx_to_label.items():
            final_labels[self.node.label == idx] = lbl
        return final_labels

    def __labelPropagation(self):
        self.zerovec = np.zeros(self.c, dtype=np.float64)
        
        # Pre-allocate buffers for backends that use them (Cython/Legacy Numba)
        max_deg = max(int(self.neib_qt.max()), 1)
        self.buf_dom_row = np.empty(self.c, dtype=np.float64)
        self.buf_reduc = np.empty(self.c, dtype=np.float64)
        self.buf_dom_list = np.empty(max_deg, dtype=np.float64)
        self.buf_dist_list = np.empty(max_deg, dtype=np.float64)
        self.buf_prob = np.empty(max_deg, dtype=np.float64)
        self.buf_slices = np.empty(max_deg, dtype=np.float64)

        early_stop = self.early_stop
        node = self.node
        part = self.part

        k_nn = self.k_nn if self.k_nn is not None else int(self.neib_qt.max())
        es_chk = self.es_chk
        
        # Calculate stop_max for early stop logic. Clamp to at least 1 to prevent 
        # premature stopping on large graphs where node/particle ratio might be small.
        stop_max = max(1, round((node.amount / (part.amount * k_nn)) * round(es_chk * 0.1)))

        # DIRECT call to the backend propagation function
        propagate_fn = self._get_backend_fn()
        
        propagate_fn(
            self.neib_list, self.neib_qt,
            self.mapped_labels, self.p_grd, self.delta_v, self.c, self.zerovec,
            part.curnode, part.label, part.strength, part.dist_table,
            node.dominance, self.owndeg, self.deltap, self.dexp,
            self.buf_dom_row, self.buf_reduc, self.buf_dom_list,
            self.buf_dist_list, self.buf_prob, self.buf_slices,
            self.dist_weights,
            self.max_iter, self.early_stop, self.es_chk, stop_max
        )

        unlabeled = node.label == -1
        node.label[unlabeled] = np.argmax(node.dominance[unlabeled, :], axis=1)

        row_sums = np.sum(self.owndeg, axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        self.owndeg = self.owndeg / row_sums

    def __genParticles(self) -> Particles:
        homenode = np.where(self.mapped_labels != -1)[0].astype(np.int64)
        curnode = homenode.copy()
        label = self.mapped_labels[self.mapped_labels != -1].astype(np.int64)
        amount = int(homenode.shape[0])
        strength = np.full(amount, 1.0, dtype=np.float64)

        n_nodes = self.neib_list.shape[0]
        max_dist = min(n_nodes - 1, 255)
        dist_table = np.full(
            shape=(n_nodes, amount),
            fill_value=max_dist,
            dtype=np.uint8,
            order='F',
        )
        dist_table[homenode, np.arange(amount)] = 0

        return Particles(
            homenode=homenode,
            curnode=curnode,
            label=label,
            strength=strength,
            amount=amount,
            dist_table=dist_table,
        )

    def __genNodes(self) -> Nodes:
        n_nodes = self.neib_list.shape[0]
        labeled_idx = self.mapped_labels != -1

        # order='F' ensures columns (classes) are contiguous
        dominance = np.full(
            shape=(n_nodes, self.c),
            fill_value=1.0 / self.c,
            dtype=np.float64,
            order='F',
        )
        # mapped_labels are used here
        dominance[labeled_idx] = 0.0
        for i in range(self.c):
            dominance[self.mapped_labels == i, i] = 1.0

        # owndeg also benefits from F order
        self.owndeg = np.zeros(
            shape=(n_nodes, self.c),
            dtype=np.float64,
            order='F',
        )

        return Nodes(
            amount=n_nodes,
            dominance=dominance,
            label=self.mapped_labels.copy(),
        )