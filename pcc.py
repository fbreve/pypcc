"""
Semi-Supervised Learning with Particle Competition and Cooperation
==================================================================

If you use this algorithm, please cite:
Breve, Fabricio Aparecido; Zhao, Liang; Quiles, Marcos Gonçalves; Pedrycz, Witold; Liu, Jiming, 
"Particle Competition and Cooperation in Networks for Semi-Supervised Learning," 
Knowledge and Data Engineering, IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012
doi: 10.1109/TKDE.2011.119

To Do:
 1. fit, predict doesn't make much sense in transductive learning, it would
    be better to have a method to generate the graph, and another to run
    particle competition and cooperation. 
    Idea: graph_gen(data) - optional (k_nn)
          fit_predict(data,label) - optional (k_nn, p_grd, delta_v, max_iter, etc...)
    maybe automatically check if data and k didn't change and automatically
    call graph_gen if needed.
    note: fit_predict matches the scikit-learn scheme
    could use a flag to not reset particles, nodes, distances, and just continue
    the iterations.
 2. The distance table is a particle attribute, it should be in the Particle class.
  
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
 2. Neirest Neighbors lists now use uint32 instead of the mix of int/int64.
 3. Neirest Neighbors list is now a matrix of uint32 instead of dic/list 
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

"""

#import time
import numpy as np
from dataclasses import dataclass

class ParticleCompetitionAndCooperation():

    def __init__(self, k_nn=10, p_grd=0.5, delta_v=0.1, max_iter=500000, es_chk=2000):

        self.k_nn = k_nn
        self.p_grd = p_grd
        self.delta_v = delta_v
        self.max_iter = max_iter
        # early stop control, decrease it to run faster, but accuracy may be lower
        self.es_chk= 2000 
        self.c = 0
        self.labels = None
        self.data = None
        self.unique_labels = None
        self.spent = 0
        self.neib_list = None
        self.neib_qt = None


    def fit(self, data, labels):

        #start = time.time()

        self.data = data
        self.labels = labels
        self.unique_labels = np.unique(self.labels) # list of classes
        self.unique_labels = self.unique_labels[self.unique_labels != -1] # excluding the "unlabeled label" (-1)
        self.c = len(self.unique_labels) # amount of classes        
        self.part = self.__genParticles()
        self.node = self.__genNodes()
        self.dist_table = self.__genDistTable()

        self.neib_list, self.neib_qt = self.__genGraph()

        #end = time.time()

        #print('finished with time: '+"{0:.5f}".format(end-start)+'s')


    def predict(self, data):

        #start = time.time()

        self.__labelPropagation()

        #end = time.time()

        # time spent with greedy walk        
        #print('finished with time: '+"{0:.5f}".format(self.spent)+'s')
        
        # time spent in total
        #print('finished with time: '+"{0:.5f}".format(end-start)+'s')

        return self.node.label


    
    def __labelPropagation(self):
       
        # this is to avoid re-creating this zero-ed array for every particle 
        # move in the update function
        self.zerovec = np.zeros(self.c)
                
        # maximum amount of stop creteria positive checks before stopping early
        stop_max = round((self.node.amount/self.part.amount) * round(self.es_chk * 0.1));
        # mean of each node maximum dominance level
        max_mmpot = 0;
        # counter of stop creteria positive checks
        stop_cnt = 0;        

        for it in range(0,self.max_iter):

            for p_i in range(0,self.part.amount):
                # get the current node the particle is visiting
                curnode = self.part.curnode[p_i]
                # get the list of neighbors of the particle current node
                curnode_neib = self.neib_list[curnode][0:self.neib_qt[curnode]]
                
                # generating one random number at a time is slow in MATLAB, but
                # in Python I couldn't notice any difference between this and
                # generating a random numbers vector in the outer loop.

                if(np.random.random() < self.p_grd):                    
                    next_node = self.__greedyWalk(p_i, curnode_neib)
                else:
                    next_node = self.__randomWalk(curnode_neib)

                self.__update(next_node, p_i)
                
            # check stop criteria
            if (np.mod(it,10)==0):
                # get mean of all nodes maximum dominance level
                mmpot = np.mean(np.amax(self.node.dominance,1))   
                # check if it is larger than the maximum we've seen so far
                if (mmpot>max_mmpot):
                    # update the maximum we've seen                   
                    max_mmpot = mmpot;
                    # reset the counter of stop criterion positive check
                    stop_cnt = 0;
                else:
                    # increase the counter of stop criterion positive check
                    stop_cnt += 1;
                    # if the counter of positive checks is larger than the 
                    # threshold, we stop earlier
                    if (stop_cnt > stop_max):
                        break

        # labeling the unlabeled nodes
        unlabeled = self.node.label==-1        
        self.node.label[unlabeled] = self.unique_labels[np.argmax(self.node.dominance[unlabeled,:],axis=1)]


    def __update(self, n_i, p_i):
               
        # for unlabeled nodes, perform the dominance vector update
        if(self.labels[n_i] == -1):
            # calculate the dominance levels reduction according to particle 
            # strength and delta_v, taking care that no reductions would
            # set a level below zero.
            # Note: numpy.clip is cleaner but much slower them np.maximum + np.zeros
            deltadom = self.node.dominance[n_i,:] - np.maximum(
                self.node.dominance[n_i,:] - self.part.strength[p_i]*(self.delta_v/(self.c-1)), self.zerovec)
            # reducing the domination levels according to the calculated 
            # reduction. Don't worry about reducing the level corresponding
            # to the particle label, it will be re-added later.
            self.node.dominance[n_i,:] -= deltadom
            # everything that was reduced from all the levels is added to the
            # level corresponding to the particle label.
            self.node.dominance[n_i,self.part.label[p_i]] += sum(deltadom)
        
        self.part.strength[p_i] = self.node.dominance[n_i,self.part.label[p_i]]

        # update distance table
        current_node = self.part.curnode[p_i]
        if(self.dist_table[n_i,p_i] > (self.dist_table[current_node,p_i]+1)):
            self.dist_table[n_i,p_i] = self.dist_table[current_node,p_i]+1

        # if there isn't a shock, move the particle to the new node
        # note: argmax is cleaner, but slower; np.amax is also slower than max
        if(self.node.dominance[n_i,self.part.label[p_i]] == max(self.node.dominance[n_i,:])):
            self.part.curnode[p_i] = n_i


    def __greedyWalk(self, p_i, neighbors):
    
        # p_i is the particle index
        # neighbors is the list of neighbors of the current node

        #start = time.time()
        
        # get the particle label
        label = self.part.label[p_i]
                                                                
        # get the domination levels for the particle in all neighbors of the
        # current node
        dom_list = self.node.dominance[neighbors,label]
        
        # get the distance from all labels of the current node and apply
        # the exponential and inversion.
        # for some reason, 1/pow(x,2) is more efficient than pow(x,-2) in Python
        # probably because the pow() is performed with integers in the first case,
        # and it has to be converted to float in the second case.        
        dist_list = 1/pow(1 + self.dist_table[neighbors,p_i].astype(int),2)
        
        # let's calculate the neighbor probabilty and keep the accumulated 
        # probabilities for a more efficient roullete step
        slices = np.cumsum(np.multiply(dom_list,dist_list))
        
        # randomly choose a neighbor given the probabilities
        rand = np.random.uniform(0,slices[-1])
                  
        # find which neighbor corresponds to the random number chosen
        # np.searchsorted uses binary search, which must be fast for large amounts of neighbors
        choice = np.searchsorted(slices, rand)
            
        #end = time.time()

        #self.spent += end - start
            
        return neighbors[choice]


    def __randomWalk(self, neighbors):
        
        return neighbors[np.random.choice(len(neighbors))]


    def __genParticles(self):
        
        @dataclass
        class Particles():
            homenode = np.where(self.labels!=-1)[0]
            # it is important to copy the vector instead of referencing it,
            # otherwise the home nodes would change with the current nodes
            curnode = np.copy(homenode) 
            label = self.labels[self.labels!=-1]
            strength = np.full(len(label),1,dtype=float)
            amount = len(homenode) # amount of particles
            
        part = Particles()

        return part


    def __genNodes(self):

        # matrix to hold nodes domination levels, first column hold the labels
        # holding labels as float64 is a waste of space, change that!
        
        @dataclass
        class Nodes():
            amount = len(self.data)
            dominance = np.full(shape=(amount,len(self.unique_labels)), fill_value=float(1/self.c),dtype=float)
            # it is important to copy the labels instead of referencing them
            # otherwise, the input vector would be changed.
            label = np.copy(self.labels)
            dominance[label != -1] = 0
            for l in np.unique(label[label!=-1]):
                dominance[label == l,l] = 1
            
        node = Nodes()

        return node


    def __genDistTable(self):

        # I changed distance table from 'int' to 'uint8' to save on memory space, 
        # but this makes the pow calculation in greedy walk much slower, since
        # it probably converts uint8 to float before the operation.         
        # As a workaround I am explicitly converting from uint8 to int in the pow()
        # function, which makes greedy walk only a little slower.
        # This could be an option in the future, to be used only with large datasets.
        
        dist_table = np.full(shape=(len(self.data),self.part.amount), fill_value=min(len(self.data)-1,255),dtype='uint8')

        for h,i in zip(self.part.homenode,range(self.part.amount)):
            dist_table[h,i] = 0

        return dist_table

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
