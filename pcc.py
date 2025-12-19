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
 2. Nearest Neighbors lists now use uint32 instead of the mix of int/int64.
 3. Nearest Neighbors list is now a matrix of uint32 instead of dic/list 
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

#import time
import numpy as np
from dataclasses import dataclass

from numba import njit  # quando for testar Numba

@njit  # depois de validar a versão pura em Python, você pode ligar o Numba
def _pcc_step(neib_list, neib_qt,
              labels, p_grd, delta_v, c, zerovec,
              part_curnode, part_label, part_strength, dist_table,
              dominance):
    """
    Executa UMA iteração do PCC sobre todas as partículas.

    Parâmetros (todos NumPy arrays / escalares):
      neib_list: int32/uint32 [n_nodes, max_deg]
      neib_qt:   int32/uint32 [n_nodes]
      labels:    int32        [n_nodes]
      p_grd:     float64
      delta_v:   float64
      c:         int (n_classes)
      zerovec:   float64      [c]
      part_curnode: int32     [n_particles]
      part_label:   int32     [n_particles]
      part_strength: float64  [n_particles]
      dist_table:    uint8    [n_nodes, n_particles]
      dominance:     float64  [n_nodes, c]
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
            label = part_label[p_i]

            # dom_list: dominância da classe da partícula em cada vizinho
            dom_list = dominance[neighbors, label]

            # dist_list: 1 / (1 + d)^2
            # (mantém a lógica da sua versão atual)
            d = dist_table[neighbors, p_i].astype(np.float64)
            dist_list = 1.0 / ((1.0 + d) * (1.0 + d))

            # slices acumulados
            prob = dom_list * dist_list
            slices = np.cumsum(prob)

            # roleta
            rand = np.random.uniform(0.0, slices[-1])
            choice = np.searchsorted(slices, rand)
            next_node = neighbors[choice]
        else:
            # --- random walk ---
            k = neighbors.shape[0]
            idx = np.random.randint(k)
            next_node = neighbors[idx]

        # --- update dominance / força / dist_table / posição da partícula ---

        # para nó não rotulado
        if labels[next_node] == -1:
            dom = dominance[next_node, :]
            step = part_strength[p_i] * (delta_v / (c - 1))

            # redução: dom - max(dom - step, 0)
            reduc = dom - np.maximum(dom - step, zerovec)

            # aplica redução
            dom -= reduc
            # soma tudo na classe da partícula
            dom[part_label[p_i]] += np.sum(reduc)

        # atualiza força
        part_strength[p_i] = dominance[next_node, part_label[p_i]]

        # dist_table
        current_node = curnode
        if dist_table[next_node, p_i] > dist_table[current_node, p_i] + 1:
            dist_table[next_node, p_i] = dist_table[current_node, p_i] + 1

        # move partícula se não houve choque
        if dominance[next_node, part_label[p_i]] == np.max(dominance[next_node, :]):
            part_curnode[p_i] = next_node

    # a função opera in-place; não precisa retornar nada


class ParticleCompetitionAndCooperation():

    def __init__(self):
        self.data = None

    def build_graph(self, data, k_nn=10):
        
        self.data = data
        self.k_nn = k_nn
        self.neib_list, self.neib_qt = self.__genGraph()
        
    def fit_predict(self, labels, p_grd=0.5, delta_v=0.1, max_iter=500000, early_stop=True, es_chk=2000):

        if (self.data is None):
            print("Error: You must build the graph first using build_graph(data)")
            return(-1)
        
        self.labels = labels
        self.p_grd = p_grd
        self.delta_v = delta_v
        self.max_iter = max_iter       
        self.early_stop = early_stop
        # early stop control, decrease it to run faster, but accuracy may be lower
        self.es_chk= es_chk        
        self.unique_labels = np.unique(self.labels) # list of classes
        self.unique_labels = self.unique_labels[self.unique_labels != -1] # excluding the "unlabeled label" (-1)
        self.c = len(self.unique_labels) # amount of classes                
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
      es_chk = self.es_chk
      max_iter = self.max_iter
      neib_list = self.neib_list
      neib_qt = self.neib_qt
      
      if early_stop:
          stop_max = round((node.amount/(part.amount*k_nn)) * round(es_chk * 0.1))
          max_mmpot = 0.0
          stop_cnt = 0
      
      for it in range(max_iter):
          _pcc_step(neib_list, neib_qt,
                    self.labels, self.p_grd, self.delta_v, self.c, self.zerovec,
                    part.curnode, part.label, part.strength, part.dist_table,
                    node.dominance)
      
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

    def __update(self, n_i, p_i):
               
        #local aliases
        labels = self.labels
        node = self.node
        part = self.part
        delta_v = self.delta_v
        c = self.c
        zerovec = self.zerovec
        
        # for unlabeled nodes, perform the dominance vector update
        if(labels[n_i] == -1):
            # calculate the dominance levels reduction according to particle 
            # strength and delta_v, taking care that no reductions would
            # set a level below zero.
            # Note: numpy.clip is cleaner but much slower them np.maximum + np.zeros
            deltadom = node.dominance[n_i,:] - np.maximum(
                node.dominance[n_i,:] - part.strength[p_i]*(delta_v/(c-1)), zerovec)
            # reducing the domination levels according to the calculated 
            # reduction. Don't worry about reducing the level corresponding
            # to the particle label, it will be re-added later.
            node.dominance[n_i,:] -= deltadom
            # everything that was reduced from all the levels is added to the
            # level corresponding to the particle label.
            node.dominance[n_i,part.label[p_i]] += np.sum(deltadom)
        
        part.strength[p_i] = node.dominance[n_i,part.label[p_i]]

        # update distance table
        current_node = part.curnode[p_i]
        if(part.dist_table[n_i,p_i] > (part.dist_table[current_node,p_i]+1)):
            part.dist_table[n_i,p_i] = part.dist_table[current_node,p_i]+1

        # if there isn't a shock, move the particle to the new node
        # note: argmax is cleaner, but slower; np.amax is also slower than max
        if(node.dominance[n_i,part.label[p_i]] == np.max(node.dominance[n_i,:])):
            part.curnode[p_i] = n_i

    def __genParticles(self):
        
        @dataclass
        class Particles():
            homenode = np.where(self.labels!=-1)[0]
            # it is important to copy the vector instead of referencing it,
            # otherwise the home nodes would change with the current nodes
            curnode = homenode.copy() 
            label = self.labels[self.labels!=-1]
            strength = np.full(len(label),1,dtype=np.float64)
            amount = len(homenode) # amount of particles
            
            # I changed distance table from 'int' to 'uint8' to save on memory space, 
            # but this makes the pow calculation in greedy walk much slower, since
            # it probably converts uint8 to float before the operation.         
            # As a workaround I am explicitly converting from uint8 to int in the pow()
            # function, which makes greedy walk only a little slower.
            # This could be an option in the future, to be used only with large datasets.
            
            dist_table = np.full(shape=(len(self.data),amount), fill_value=min(len(self.data)-1,255),dtype=np.uint8)
    
            for h,i in zip(homenode,range(amount)):
                dist_table[h,i] = 0
            
        part = Particles()

        return part


    def __genNodes(self):
       
        @dataclass
        class Nodes():
            amount = len(self.data)
            dominance = np.full(shape=(amount,len(self.unique_labels)), fill_value=float(1/self.c),dtype=np.float64)
            # it is important to copy the labels instead of referencing them
            # otherwise, the input vector would be changed.
            label = self.labels.copy()
            dominance[label != -1] = 0
            for l in np.unique(label[label!=-1]):
                dominance[label == l,l] = 1
            
        node = Nodes()

        return node


    def __genGraph(self):

        from sklearn.neighbors import NearestNeighbors
        # find the k-neirest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k_nn+1, algorithm='auto', n_jobs=-1).fit(self.data)
        neib_list = np.uint32(nbrs.kneighbors(self.data, return_distance=False))
        # discard self-distance
        neib_list = np.delete(neib_list, 0, 1) 
        
        # define the total amount of nodes
        qt_node = len(neib_list)
        # define the amount of neighbors of each node (intially it is k)
        neib_qt = np.full(qt_node, self.k_nn, dtype=np.uint32)

        # keep track of how many columns the neib_list matrix has
        ind_cols = self.k_nn

        # add the reciprocal connections
        for i in range(0,qt_node):
            for j in range(0,self.k_nn):
                target = neib_list[i,j]
                # check if there is space for the new element, if not increase matrix size
                if neib_qt[target]==ind_cols:
                    # increase by 20% + 1
                    new_cols_qt = round(ind_cols * 0.2) + 1
                    # add the new columns to the matrix
                    neib_list = np.append(neib_list, np.empty([qt_node, new_cols_qt],dtype=np.uint32), axis=1)
                    # increase the cols counter
                    ind_cols += new_cols_qt
                # add the reciprocal connection
                neib_list[target, neib_qt[target]] = i
                # increase the amount of neighbors of the target
                neib_qt[target] += 1
                
        # remove the duplicate neighbors for each node
        for i in range(0,qt_node):
            # generate the list of unique neighbors
            unique = np.unique(neib_list[i,:neib_qt[i]])
            # get the amount of unique neighbors
            neib_qt[i] = len(unique)
            # copy the list of unique neighbors to the matrix row
            neib_list[i,:neib_qt[i]] = unique
            
        # remove the now unused last columns
        ind_cols = max(neib_qt)
        neib_list = np.delete(neib_list, np.s_[ind_cols:], 1)
        
        return neib_list, neib_qt
