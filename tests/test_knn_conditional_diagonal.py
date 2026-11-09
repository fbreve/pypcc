"""Permanent-style exact graph invariants for conditional kNN diagonal cleanup."""
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pcc_graph import build_knn_graph, build_graph_from_edge_index


def historical_reference(x,k):
    # Independent old algorithm with unconditional diagonal cleanup.
    # Scientific auditor additionally imports the EXACT original Git source.
    x=np.asarray(x)
    nbrs=NearestNeighbors(n_neighbors=k+1,algorithm="auto",
                          metric="minkowski",p=2,n_jobs=1).fit(x)
    indices=nbrs.kneighbors(x,return_distance=False)[:,1:]
    n=len(x)
    rows=np.repeat(np.arange(n,dtype=np.int64),k)
    cols=indices.ravel().astype(np.int64)
    adj=csr_matrix((np.ones(len(rows),dtype=np.int8),(rows,cols)),
                   shape=(n,n),dtype=np.int8)
    adj=adj+adj.T
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:]=1
    degrees=np.diff(adj.indptr).astype(np.int64)
    out=np.full((n,int(degrees.max())),-1,dtype=np.int64)
    ix=np.repeat(np.arange(n,dtype=adj.indices.dtype),degrees)
    offset=(np.arange(adj.indices.size,dtype=adj.indptr.dtype)
            -np.repeat(adj.indptr[:-1],degrees))
    out[ix,offset]=adj.indices
    return out,degrees,indices


class ConditionalKnnDiagonalTests(unittest.TestCase):
    def compare(self,x,k):
        x=np.asarray(x)
        before=x.copy()
        old_q,old_d,original_indices=historical_reference(x,k)
        new_q,new_d=build_knn_graph(x,k_nn=k,n_jobs=1)
        np.testing.assert_array_equal(new_q,old_q)
        np.testing.assert_array_equal(new_d,old_d)
        np.testing.assert_array_equal(x,before)
        self.assertEqual(new_q.dtype,np.int64)
        self.assertEqual(new_d.dtype,np.int64)
        self.assertTrue(new_q.flags.c_contiguous)
        self.assertTrue(new_d.flags.c_contiguous)
        self.assertTrue(new_q.flags.writeable)
        self.assertTrue(new_d.flags.writeable)
        mask=np.arange(new_q.shape[1])[None,:]<new_d[:,None]
        self.assertTrue(np.all(new_q[~mask]==-1))
        return int(np.sum(original_indices==np.arange(len(x))[:,None]))

    def test_real_ordered_graphs_on_k1_k3_k10(self):
        for loader in (load_iris,load_wine,load_breast_cancer,load_digits):
            x=StandardScaler().fit_transform(loader(return_X_y=True)[0])
            for k in (1,3,10):
                with self.subTest(name=loader.__name__,k=k):
                    self.compare(x,k)

    def test_equal_distance_ties_and_duplicates_include_actual_self_edges(self):
        x=np.zeros((64,4),dtype=np.float64)
        counts=[]
        for k in (1,3,10):
            with self.subTest(k=k):
                counts.append(self.compare(x,k))
        self.assertGreater(sum(counts),0,
                           "duplicate/nearest-tie scenario must exercise real diagonal")

    def test_noncontiguous_real_feature_matrix(self):
        rng=np.random.default_rng(51011)
        x=rng.normal(size=(140,9))
        x[30:52]=x[0]
        x=x[:,::2]  # noncontiguous view; do not modify or force new neighbor order
        self.assertFalse(x.flags.c_contiguous)
        for k in (1,3,10):
            with self.subTest(k=k):
                self.compare(x,k)

    def test_original_invalid_arguments_and_edge_index_builder_untouched(self):
        x=np.arange(24,dtype=np.float64).reshape((12,2))
        with self.assertRaisesRegex(ValueError,"2-D"):
            build_knn_graph(x.reshape(-1),k_nn=3)
        for k in (0,-1,12):
            with self.subTest(k=k),self.assertRaises(ValueError):
                build_knn_graph(x,k_nn=k)
        # Conditional skip must not modify other graph constructor's handling
        # of duplicate/loop edges, reciprocal adjacency and isolate padding.
        edges=np.array([[0,0,1,1,2,2],[0,1,0,1,0,0]],dtype=np.int64)
        neibs,degrees=build_graph_from_edge_index(5,edges)
        np.testing.assert_array_equal(degrees,np.array([2,1,1,0,0]))
        self.assertEqual(neibs.dtype,np.int64)
        self.assertEqual(neibs.shape,(5,2))
        np.testing.assert_array_equal(neibs[:3,:],np.array([[1,2],[0,-1],[0,-1]]))
        np.testing.assert_array_equal(neibs[3:,:],np.full((2,2),-1,dtype=np.int64))


if __name__=="__main__":
    unittest.main()
