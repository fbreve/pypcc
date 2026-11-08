"""Retain original ordered CSR graph construction and padding semantics.

The reference intentionally uses the previous per-vertex NumPy list approach,
not the new flat CSR offset formula, to catch ordering/shape regressions.
"""
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris,load_wine,load_breast_cancer,load_digits

from pcc_graph import build_knn_graph,build_graph_from_edge_index


def previous_ordered_scatter(adj,edge_index=False):
    n=adj.shape[0]
    if edge_index and not n:
        return np.empty((0,0),dtype=np.int64),np.empty(0,dtype=np.int64)
    deg=np.diff(adj.indptr).astype(np.int64)
    width=int(deg.max())
    if edge_index:
        width=max(width,1)
    out=np.full((n,width),-1,dtype=np.int64)
    rows=np.repeat(np.arange(n,dtype=np.int64),deg)
    cols=(np.concatenate([np.arange(int(d),dtype=np.int64) for d in deg])
          if n else np.empty(0,dtype=np.int64))
    if not edge_index or rows.size:
        out[rows,cols]=adj.indices.astype(np.int64)
    return out,deg


def previous_knn_graph(x,k,p):
    n=len(x)
    nbrs=NearestNeighbors(n_neighbors=k+1,algorithm="auto",
                          metric="minkowski",p=p,n_jobs=1).fit(x)
    indices=nbrs.kneighbors(x,return_distance=False)[:,1:]
    rows=np.repeat(np.arange(n,dtype=np.int64),k)
    cols=indices.ravel().astype(np.int64)
    adj=csr_matrix((np.ones(len(rows),dtype=np.int8),(rows,cols)),
                   shape=(n,n),dtype=np.int8)
    adj=adj+adj.T
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:]=1
    return previous_ordered_scatter(adj)


def previous_edge_graph(n,edges):
    if not n:
        return np.empty((0,0),dtype=np.int64),np.empty(0,dtype=np.int64)
    src,dst=np.asarray(edges,dtype=np.int64)
    rows=np.concatenate((src,dst))
    cols=np.concatenate((dst,src))
    adj=csr_matrix((np.ones(len(rows),dtype=np.bool_),(rows,cols)),
                   shape=(n,n),dtype=np.bool_)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:]=1
    return previous_ordered_scatter(adj,True)


class OrderedCsrScatterCompatibilityTests(unittest.TestCase):
    def assert_graph_exact(self,original,candidate):
        self.assertEqual(len(original),len(candidate))
        for old,new in zip(original,candidate):
            self.assertEqual(old.dtype,new.dtype)
            self.assertEqual(old.shape,new.shape)
            np.testing.assert_array_equal(old,new)

    def test_original_knn_with_ties_duplicates_and_noncontiguous_inputs(self):
        examples=(
            np.array([[0.],[0.],[1.],[1.],[2.],[4.],[8.]],dtype=float),
            np.asfortranarray(np.arange(120,dtype=float).reshape(24,5)),
            np.array([[0,0],[0,0],[1,0],[1,0],
                      [0,1],[0,1],[1,1],[1,1]],dtype=float),
        )
        for x in examples:
            for k in sorted({1,min(3,len(x)-1),len(x)-1}):
                for p in (1,2):
                    with self.subTest(n=len(x),k=k,p=p):
                        expected=previous_knn_graph(x,k,p)
                        actual=build_knn_graph(x,k_nn=k,p=p,n_jobs=1)
                        self.assert_graph_exact(expected,actual)

    def test_original_four_real_dataset_graphs_are_bitwise_unchanged(self):
        for loader in (load_iris,load_wine,load_breast_cancer,load_digits):
            x=StandardScaler().fit_transform(
                np.asarray(loader().data,dtype=np.float64))
            for k in (1,3,10):
                with self.subTest(dataset=loader.__name__,k=k):
                    self.assert_graph_exact(
                        previous_knn_graph(x,k,2),
                        build_knn_graph(x,k_nn=k,p=2,n_jobs=1))

    def test_original_edge_index_loops_isolates_duplicates_and_padding(self):
        cases=(
            (0,np.empty((2,0),dtype=np.int64)),
            (1,np.empty((2,0),dtype=np.int64)),
            (8,np.empty((2,0),dtype=np.int64)),
            (8,np.array([[0,0,1,1,2,3,3,5],
                         [1,1,0,2,1,3,5,5]],dtype=np.int64)),
            (8,np.array([[7,7,7,2,3,3,4,5],
                         [0,0,1,3,2,3,6,6]],dtype=np.int64)),
        )
        for n,edge_index in cases:
            with self.subTest(nodes=n,edges=edge_index.shape[1]):
                self.assert_graph_exact(
                    previous_edge_graph(n,edge_index),
                    build_graph_from_edge_index(n,edge_index))
        empty,degree=build_graph_from_edge_index(
            8,np.empty((2,0),dtype=np.int64))
        self.assertEqual(empty.shape,(8,1))
        np.testing.assert_array_equal(empty,-np.ones((8,1),dtype=np.int64))
        np.testing.assert_array_equal(degree,np.zeros(8,dtype=np.int64))


if __name__=="__main__":
    unittest.main()
