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

    def __init__(self, impl="auto"):
        """
        impl: 'auto', 'cython', 'numba' ou 'numpy'
        """
        self.impl = impl
        self.data = None
        self.k_nn = None
        self.neib_list = None
        self.neib_qt = None
        self.labels = None
        self.unique_labels = None
        self.c = None
        self.p_grd = None
        self.delta_v = None
        self.max_iter = None
        self.early_stop = None
        self.es_chk = None
        self.part = None
        self.node = None
        self.zerovec = None
        self.owndeg = None

    def _step_backend(self,
                      neib_list, neib_qt,
                      labels, p_grd, delta_v, c, zerovec,
                      part_curnode, part_label, part_strength, dist_table,
                      dominance, owndeg):

        # Normaliza impl: se vier algo estranho, trata como "auto"
        impl = self.impl.lower() if isinstance(self.impl, str) else "auto"
        if impl not in ("cython", "numba", "numpy", "auto"):
            warnings.warn(f"Backend '{self.impl}' is invalid; using 'auto' (cython→numba→numpy).")
            impl = "auto"

        # Ordem de fallback
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
                return pcc_step_cython(neib_list, neib_qt,
                                       labels, p_grd, delta_v, c, zerovec,
                                       part_curnode, part_label, part_strength, dist_table,
                                       dominance, owndeg)

            if be == "numba" and _HAS_NUMBA:
                return pcc_step_numba(neib_list, neib_qt,
                                      labels, p_grd, delta_v, c, zerovec,
                                      part_curnode, part_label, part_strength, dist_table,
                                      dominance, owndeg)

            if be == "numpy":
                if impl != "numpy":
                    unavailable = ", ".join(b for b in backends if b != "numpy")
                    warnings.warn(
                        f"Backends {unavailable} are not available; using NumPy implementation."
                    )

                return pcc_step_numpy(neib_list, neib_qt,
                                      labels, p_grd, delta_v, c, zerovec,
                                      part_curnode, part_label, part_strength, dist_table,
                                      dominance, owndeg)

        # Se chegou aqui, algo deu muito errado (nenhum backend disponível)
        raise RuntimeError("No backend available: cython, numba, or numpy.")

    # daqui pra baixo, reutiliza tua implementação original, trocando só o _pcc_step pelo _step_backend:

    from pcc_graph import build_knn_graph  # ajuste import conforme seu projeto

    def build_graph(self, data, k_nn=10):
        self.data = data
        self.k_nn = k_nn
        self.neib_list, self.neib_qt = build_knn_graph(data, k_nn)

    def set_graph(self, neib_list, neib_qt):
        self.neib_list = neib_list.astype(np.int64)
        self.neib_qt = neib_qt.astype(np.int64)
        self.data = None

    def fit_predict(self, labels, p_grd=0.5, delta_v=0.1,
                    max_iter=500000, early_stop=True, es_chk=2000):

        if self.neib_list is None or self.neib_qt is None:
            print("Error: build or set the graph first.")
            return -1

        self.labels = labels.astype(np.int64)
        self.p_grd = float(p_grd)
        self.delta_v = float(delta_v)
        self.max_iter = int(max_iter)
        self.early_stop = bool(early_stop)
        self.es_chk = int(es_chk)

        self.unique_labels = np.unique(self.labels)
        self.unique_labels = self.unique_labels[self.unique_labels != -1]
        self.c = len(self.unique_labels)

        self.part = self.__genParticles()
        self.node = self.__genNodes()

        self.__labelPropagation()

        return self.node.label

    def __labelPropagation(self):
        self.zerovec = np.zeros(self.c, dtype=np.float64)

        early_stop = self.early_stop
        node = self.node
        part = self.part

        k_nn = self.k_nn
        if k_nn is None:
            k_nn = int(np.mean(self.neib_qt)) if self.neib_qt is not None else 10

        es_chk = self.es_chk
        max_iter = self.max_iter
        neib_list = self.neib_list
        neib_qt = self.neib_qt

        if early_stop:
            stop_max = round((node.amount / (part.amount * k_nn)) * round(es_chk * 0.1))
            max_mmpot = 0.0
            stop_cnt = 0

        for it in range(max_iter):
            self._step_backend(neib_list, neib_qt,
                               self.labels, self.p_grd, self.delta_v, self.c, self.zerovec,
                               part.curnode, part.label, part.strength, part.dist_table,
                               node.dominance, self.owndeg)

            if early_stop and it % 10 == 0:
                mmpot = np.mean(np.amax(node.dominance, 1))
                if mmpot > max_mmpot:
                    max_mmpot = mmpot
                    stop_cnt = 0
                else:
                    stop_cnt += 1
                    if stop_cnt > stop_max:
                        break

        unlabeled = node.label == -1
        node.label[unlabeled] = self.unique_labels[
            np.argmax(node.dominance[unlabeled, :], axis=1)
        ]

        row_sums = np.sum(self.owndeg, axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        self.owndeg = self.owndeg / row_sums

    def __genParticles(self) -> Particles:
        homenode = np.where(self.labels != -1)[0].astype(np.int64)
        curnode = homenode.copy()
        label = self.labels[self.labels != -1].astype(np.int64)
        amount = int(homenode.shape[0])
        strength = np.full(amount, 1.0, dtype=np.float64)

        n_nodes = self.neib_list.shape[0]
        max_dist = min(n_nodes - 1, 255)
        dist_table = np.full(
            shape=(n_nodes, amount),
            fill_value=max_dist,
            dtype=np.uint8,
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
        amount = self.neib_list.shape[0]
        n_classes = self.c

        dominance = np.full(
            shape=(amount, n_classes),
            fill_value=float(1.0 / n_classes),
            dtype=np.float64,
        )

        label = self.labels.copy().astype(np.int64)
        dominance[label != -1, :] = 0.0

        for l in self.unique_labels:
            dominance[label == l, l] = 1.0

        self.owndeg = np.full(
            shape=(amount, n_classes),
            fill_value=np.finfo(float).tiny,
            dtype=np.float64,
        )

        return Nodes(
            amount=amount,
            dominance=dominance,
            label=label,
        )
