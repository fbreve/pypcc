"""Edge-index conditional CSR diagonal cleanup safety characterization."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from pcc_graph import build_graph_from_edge_index,build_knn_graph


def original_edge_index_builder(n,edges):
    e=np.asarray(edges,dtype=np.int64)
    if e.ndim!=2 or e.shape[0]!=2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if n==0:
        return np.empty((0,0),dtype=np.int64),np.empty(0,dtype=np.int64)
    src,dst=e
    rows=np.concatenate((src,dst))
    cols=np.concatenate((dst,src))
    adj=csr_matrix(
        (np.ones(len(rows),dtype=np.bool_),(rows,cols)),
        shape=(n,n),dtype=np.bool_)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:]=1
    deg=np.diff(adj.indptr).astype(np.int64)
    width=max(int(deg.max()),1)
    matrix=np.full((n,width),-1,dtype=np.int64)
    rr=np.repeat(np.arange(n,dtype=np.int64),deg)
    cc=np.arange(adj.indices.size,dtype=np.int64)-np.repeat(
        adj.indptr[:-1],deg)
    if len(rr):
        matrix[rr,cc]=adj.indices.astype(np.int64)
    return matrix,deg


class ConditionalDiagonalGraphTests(unittest.TestCase):
    def assert_same(self,n,edges):
        a=original_edge_index_builder(n,edges)
        b=build_graph_from_edge_index(n,edges)
        for old,new in zip(a,b):
            self.assertEqual(old.dtype,new.dtype)
            self.assertEqual(old.shape,new.shape)
            np.testing.assert_array_equal(old,new)

    def test_empty_graphs_and_many_isolates(self):
        for n in (0,1,2,16,1000,10000):
            with self.subTest(n=n):
                self.assert_same(n,np.empty((2,0),dtype=np.int64))

    def test_self_loops_duplicates_reciprocals_and_padding(self):
        cases=(
            (1,np.array([[0,0,0],[0,0,0]],dtype=np.int64)),
            (5,np.array([[0,1,1,2,2,2,3,3],[1,0,2,2,4,2,3,3]])),
            (12,np.array([[4,4,4,4,5,5,8,9,9,9,11],
                          [5,5,4,7,4,4,9,8,9,11,9]])),
        )
        for n,edges in cases:
            with self.subTest(n=n,m=edges.shape[1]):
                self.assert_same(n,edges)

    def test_stable_random_sparse_and_dense_graphs(self):
        rng=np.random.default_rng(28311)
        for n,m in ((300,10),(300,1000),(1000,5000),
                    (3000,100),(3000,15000)):
            with self.subTest(n=n,m=m):
                src=rng.integers(0,n,size=m,dtype=np.int64)
                dst=rng.integers(0,n,size=m,dtype=np.int64)
                dst=np.where(dst==src,(dst+1)%n,dst)
                edges=np.vstack((src,dst))
                self.assertFalse(np.any(edges[0]==edges[1]))
                self.assert_same(n,edges)
                # Introduce self-loops in the same graph and test the old
                # diagonal-cleaning path as well.
                edges_with_loops=edges.copy()
                edges_with_loops[1,:min(m,25)]=edges_with_loops[0,:min(m,25)]
                self.assert_same(n,edges_with_loops)

    def test_malformed_edge_inputs_not_silently_accepted(self):
        cases=(
            (3,np.empty((3,0),dtype=np.int64)),
            (-1,np.empty((2,0),dtype=np.int64)),
            (3,np.array([[0,3],[1,2]],dtype=np.int64)),
            (3,np.array([[0,-1],[1,2]],dtype=np.int64)),
        )
        for n,edges in cases:
            with self.subTest(n=n,edge_shape=edges.shape):
                outcomes=[]
                for fn in (original_edge_index_builder,build_graph_from_edge_index):
                    try:
                        fn(n,edges)
                        outcomes.append(None)
                    except Exception as e:
                        outcomes.append((type(e).__name__,str(e)))
                self.assertEqual(outcomes[0],outcomes[1])
                self.assertIsNotNone(outcomes[0])


if __name__=="__main__":
    unittest.main()
