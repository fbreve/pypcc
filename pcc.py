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

import numpy as np
from dataclasses import dataclass
from numba import njit  # opcional; pode ser comentado para depuração pura em Python
from pcc_graph import build_knn_graph

@dataclass
class Particles:
    homenode: np.ndarray        # int64 [n_particles]
    curnode: np.ndarray         # int64 [n_particles]
    label: np.ndarray           # int64 [n_particles]
    strength: np.ndarray        # float64 [n_particles]
    amount: int                 # número de partículas
    dist_table: np.ndarray      # uint8 [n_nodes, n_particles]


@dataclass
class Nodes:
    amount: int                 # número de nós
    dominance: np.ndarray       # float64 [n_nodes, n_classes]
    label: np.ndarray           # int64 [n_nodes]

@njit  # depois de validar a versão pura em Python, você pode ligar o Numba
def _pcc_step(neib_list, neib_qt,
              labels, p_grd, delta_v, c, zerovec,
              part_curnode, part_label, part_strength, dist_table,
              dominance, owndeg):    
    """
    Executa UMA iteração do PCC sobre todas as partículas.

    Parâmetros (todos NumPy arrays / escalares):
      neib_list: int64 [n_nodes, max_deg]
      neib_qt:   int64 [n_nodes]
      labels:    int64 [n_nodes]
      p_grd:     float64
      delta_v:   float64
      c:         int64 (n_classes)
      zerovec:   float64 [c]
      part_curnode: int64 [n_particles]
      part_label:   int64 [n_particles]
      part_strength: float64 [n_particles]
      dist_table:    uint8  [n_nodes, n_particles]
      dominance:     float64 [n_nodes, c]
      owndeg:       float64 [n_nodes, c]
    """

    n_particles = part_curnode.shape[0]

    for p_i in range(n_particles):
        curnode = part_curnode[p_i]
        k = neib_qt[curnode]

        # vizinhos do nó atual
        neighbors = neib_list[curnode, :k]

        # greedy x random
        if np.random.random() < p_grd:                      
            # --- greedy walk ---
            greedy = True
            
            label = part_label[p_i]

            # dominância da classe da partícula em cada vizinho
            dom_list = dominance[neighbors, label]

            # dist_list: 1 / (1 + d)^2
            d = dist_table[neighbors, p_i].astype(np.float64)
            dist_list = 1.0 / ((1.0 + d) * (1.0 + d))

            prob = dom_list * dist_list
            slices = np.cumsum(prob)

            # roleta
            rand = np.random.uniform(0.0, slices[-1])
            choice = np.searchsorted(slices, rand)
            next_node = neighbors[choice]
        else:
            # --- random walk ---
            greedy = False
            
            k_rand = neighbors.shape[0]
            idx = np.random.randint(k_rand)
            next_node = neighbors[idx]

        # --- update dominance / força / dist_table / posição da partícula ---

        # apenas nós não rotulados sofrem atualização de dominância
        if labels[next_node] == -1:
            dom = dominance[next_node, :]
            step = part_strength[p_i] * (delta_v / (c - 1))

            # redução: dom - max(dom - step, 0)
            reduc = dom - np.maximum(dom - step, zerovec)

            dom -= reduc
            dom[part_label[p_i]] += np.sum(reduc)

        # atualiza força
        part_strength[p_i] = dominance[next_node, part_label[p_i]]

        # dist_table
        current_node = curnode
        if dist_table[next_node, p_i] > dist_table[current_node, p_i] + 1:
            dist_table[next_node, p_i] = dist_table[current_node, p_i] + 1

        # se foi movimento aleatório, acumula owndeg
        if not greedy:
            owndeg[next_node, part_label[p_i]] += part_strength[p_i]

        # move partícula se não houve choque
        if dominance[next_node, part_label[p_i]] == np.max(dominance[next_node, :]):
            part_curnode[p_i] = next_node

class ParticleCompetitionAndCooperation:

    def __init__(self):
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
        """
        Executa o PCC de forma transdutiva e retorna um vetor de rótulos
        (incluindo os nós originalmente não rotulados).
        """
    
        # precisa ter neib_list e neib_qt definidos, independente de data
        if self.neib_list is None or self.neib_qt is None:
            print("Error: You must build or set the graph first "
                  "using build_graph(data) or set_graph(neib_list, neib_qt).")
            return -1

        self.labels = labels.astype(np.int64)
        self.p_grd = float(p_grd)
        self.delta_v = float(delta_v)
        self.max_iter = int(max_iter)
        self.early_stop = bool(early_stop)
        self.es_chk = int(es_chk)

        # classes reais (sem -1)
        self.unique_labels = np.unique(self.labels)
        self.unique_labels = self.unique_labels[self.unique_labels != -1]
        self.c = len(self.unique_labels)

        # gera partículas e nós
        self.part = self.__genParticles()
        self.node = self.__genNodes()

        # propaga rótulos
        self.__labelPropagation()

        return self.node.label

    def __labelPropagation(self):
        """
        Loop principal de iterações do PCC com critério de early stop.
        """
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
            # heurística baseada em média de iterações por nó/partícula
            stop_max = round((node.amount / (part.amount * k_nn)) * round(es_chk * 0.1))
            max_mmpot = 0.0
            stop_cnt = 0

        for it in range(max_iter):
            _pcc_step(neib_list, neib_qt,
                      self.labels, self.p_grd, self.delta_v, self.c, self.zerovec,
                      part.curnode, part.label, part.strength, part.dist_table,
                      node.dominance, self.owndeg)

            if early_stop and it % 10 == 0:
                # potencial médio de dominância
                mmpot = np.mean(np.amax(node.dominance, 1))
                if mmpot > max_mmpot:
                    max_mmpot = mmpot
                    stop_cnt = 0
                else:
                    stop_cnt += 1
                    if stop_cnt > stop_max:
                        break

        # rotula nós originalmente não rotulados
        unlabeled = node.label == -1
        node.label[unlabeled] = self.unique_labels[
            np.argmax(node.dominance[unlabeled, :], axis=1)
        ]
               
        # normaliza owndeg por linha (como no MATLAB)
        row_sums = np.sum(self.owndeg, axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        self.owndeg = self.owndeg / row_sums

    def __genParticles(self) -> Particles:
        """
        Cria as partículas a partir dos nós rotulados.
        """
    
        homenode = np.where(self.labels != -1)[0].astype(np.int64)
        curnode = homenode.copy()
        label = self.labels[self.labels != -1].astype(np.int64)
        amount = int(homenode.shape[0])
        strength = np.full(amount, 1.0, dtype=np.float64)
    
        # número de nós vem do grafo
        n_nodes = self.neib_list.shape[0]
    
        # distância máxima limitada a 255 (uint8)
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
        """
        Inicializa dominâncias e rótulos dos nós.
        """
    
        amount = self.neib_list.shape[0]  # número de nós
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
