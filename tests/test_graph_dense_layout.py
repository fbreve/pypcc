"""Exact ordered CSR-offset scatter with ties, isolates and edge cases."""
import unittest
import numpy as np

from pcc_graph import build_knn_graph,build_graph_from_edge_index


def expected_dense_from_adjacency(adjacency,edge_index=False):
    n=len(adjacency)
    if edge_index and n==0:
        return np.empty((0,0),dtype=np.int64),np.empty(0,dtype=np.int64)
    degrees=np.array([len(x) for x in adjacency],dtype=np.int64)
    width=int(max(degrees,default=0))
    if edge_index:
        width=max(width,1)
    out=np.full((n,width),-1,dtype=np.int64)
    for i,neigh in enumerate(adjacency):
        out[i,:len(neigh)]=sorted(neigh)
    return out,degrees


class GraphDenseOffsetTests(unittest.TestCase):
    def test_exact_handwritten_directed_input_symmetrization_and_isolates(self):
        cases=[
            (0,np.empty((2,0),dtype=np.int64),[]),
            (5,np.empty((2,0),dtype=np.int64),[[] for _ in range(5)]),
            (6,np.array([[0,0,1,1,2,3,4,4,4],
                         [1,1,0,2,1,3,4,2,2]]),
             [[1],[0,2],[1,4],[],[2],[]]),
            (3,np.array([[0,0,0,1,1,2],
                         [0,0,1,0,2,1]]),
             [[1],[0,2],[1]]),
            (7,np.array([[0,0,1,1,1,5,5,6],
                         [4,6,4,2,3,6,6,5]]),
             [[4,6],[2,3,4],[1],[1],[0,1],[6],[0,5]]),
        ]
        for n,edges,adj in cases:
            with self.subTest(n=n,edges=edges.shape[1]):
                expected=expected_dense_from_adjacency(adj,True)
                actual=build_graph_from_edge_index(n,edges)
                for a,b in zip(actual,expected):
                    self.assertEqual(a.dtype,np.dtype("int64"))
                    np.testing.assert_array_equal(a,b)

    def test_canonical_union_knn_undirected_ties_and_duplicate_rows(self):
        data=[
            np.array([[0.],[1.],[3.],[9.]]),
            np.array([[0,0],[0,0],[1,0],[1,0],
                      [0,1],[0,1],[1,1],[1,1]],dtype=np.float64),
            np.asfortranarray(np.arange(120).reshape(24,5).astype(float)),
            np.column_stack([np.repeat(np.arange(7),3),
                             np.tile([0.,1.,2.],7)]),
        ]
        for x in data:
            for k in (1,min(3,len(x)-1),len(x)-1):
                for power in (1,2):
                    with self.subTest(n=len(x),k=k,p=power):
                        neib,deg=build_knn_graph(x,k_nn=k,p=power)
                        self.assertEqual(neib.dtype,np.dtype("int64"))
                        self.assertEqual(deg.dtype,np.dtype("int64"))
                        self.assertEqual(neib.shape[0],len(x))
                        # With duplicate samples sklearn may place a
                        # duplicate before the query itself; the ORIGINAL
                        # source discards [:, 0] then removes diagonal
                        # entries. Degree can therefore be below k.
                        self.assertTrue(np.all((0<=deg)&(deg<len(x))))
                        for v in range(len(x)):
                            ns=neib[v,:deg[v]]
                            self.assertTrue(np.all(np.diff(ns)>=0))
                            self.assertNotIn(v,ns)
                            np.testing.assert_array_equal(
                                neib[v,deg[v]:],
                                np.full(neib.shape[1]-deg[v],-1))
                            for u in ns:
                                self.assertIn(v,neib[u,:deg[u]])

    def test_no_hidden_empty_edge_scatter_mismatch(self):
        e=np.empty((2,0),dtype=np.int64)
        for n in (0,1,2,12):
            with self.subTest(n=n):
                graph,degrees=build_graph_from_edge_index(n,e)
                self.assertEqual(graph.shape,(0,0) if n==0 else (n,1))
                self.assertEqual(degrees.shape,(n,))
                self.assertTrue(np.all(graph==-1))
                self.assertTrue(np.all(degrees==0))


if __name__=="__main__":
    unittest.main()
