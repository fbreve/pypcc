"""Original graph-array contract with reused degrees and native CSR index width."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from pcc_graph import build_knn_graph,build_graph_from_edge_index


def previous_scatter(adj,edge_index):
    n=adj.shape[0]
    degrees=np.diff(adj.indptr).astype(np.int64)
    width=(max(int(degrees.max()),1) if edge_index else int(degrees.max()))
    out=np.full((n,width),-1,dtype=np.int64)
    rr=np.repeat(np.arange(n,dtype=np.int64),degrees)
    cc=(np.arange(adj.indices.size,dtype=np.int64)
        -np.repeat(adj.indptr[:-1],degrees))
    if not edge_index or rr.size:
        out[rr,cc]=adj.indices.astype(np.int64)
    return out,degrees


def previous_edge(n,edge_index):
    e=np.asarray(edge_index,dtype=np.int64)
    if e.ndim!=2 or e.shape[0]!=2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if n==0:
        return np.empty((0,0),dtype=np.int64),np.empty(0,dtype=np.int64)
    src,dst=e
    rows=np.concatenate((src,dst))
    cols=np.concatenate((dst,src))
    adj=csr_matrix((np.ones(len(rows),dtype=np.bool_),(rows,cols)),
                   shape=(n,n),dtype=np.bool_)
    if np.any(src==dst):
        adj.setdiag(0)
        adj.eliminate_zeros()
    adj.data[:]=1
    return previous_scatter(adj,True)


def previous_knn(x,k,p=2):
    from sklearn.neighbors import NearestNeighbors
    n=len(x)
    nb=NearestNeighbors(n_neighbors=k+1,algorithm="auto",
                        metric="minkowski",p=p,n_jobs=1).fit(x)
    idx=nb.kneighbors(x,return_distance=False)[:,1:]
    rr=np.repeat(np.arange(n,dtype=np.int64),k)
    cc=idx.ravel().astype(np.int64)
    adj=csr_matrix((np.ones(len(rr),dtype=np.int8),(rr,cc)),
                   shape=(n,n),dtype=np.int8)
    adj=adj+adj.T
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:]=1
    return previous_scatter(adj,False)


class ReusedDegreesGraphCompatibilityTests(unittest.TestCase):
    def compare(self,old,new):
        for a,b in zip(old,new):
            self.assertEqual(a.shape,b.shape)
            self.assertEqual(a.dtype,np.dtype("int64"))
            self.assertEqual(b.dtype,np.dtype("int64"))
            np.testing.assert_array_equal(a,b)

    def test_original_all_four_real_graphs_all_k(self):
        for loader in (datasets.load_iris,datasets.load_wine,
                       datasets.load_breast_cancer,datasets.load_digits):
            x=StandardScaler().fit_transform(
                np.asarray(loader().data,dtype=np.float64))
            for k in (1,3,10):
                with self.subTest(data=loader.__name__,k=k):
                    self.compare(previous_knn(x,k),
                                 build_knn_graph(x,k_nn=k,n_jobs=1))
                    old=previous_knn(x,k)
                    rr=np.repeat(np.arange(len(x),dtype=np.int64),old[1])
                    valid=(np.arange(old[0].shape[1])[None,:]
                           <old[1][:,None])
                    e=np.vstack((rr,old[0][valid].astype(np.int64)))
                    self.compare(previous_edge(len(x),e),
                                 build_graph_from_edge_index(len(x),e))

    def test_tied_duplicates_minkowski_and_noncontiguous(self):
        x=np.array([[0.,0.],[0.,0.],[0.,1.],[0.,1.],
                    [1.,0.],[1.,0.],[1.,1.],[1.,1.],
                    [2.,2.],[2.,2.]])
        for features in (x,np.asfortranarray(x),x[:,::-1]):
            for k in (1,3,len(x)-1):
                for p in (1,2):
                    with self.subTest(k=k,p=p,fortran=features.flags.f_contiguous):
                        self.compare(previous_knn(features,k,p),
                                     build_knn_graph(features,k_nn=k,p=p,n_jobs=1))

    def test_all_edge_types_and_isolates(self):
        cases=(
            (0,np.empty((2,0),dtype=np.int64)),
            (1,np.empty((2,0),dtype=np.int64)),
            (1000,np.empty((2,0),dtype=np.int64)),
            (9,np.array([[0,0,1,2,3,3,4,8,8],
                         [1,1,0,3,2,3,4,8,7]],dtype=np.int64)),
            (32,np.array([[0,1,1,1,7,9,9,9,18,19],
                          [1,0,1,2,7,9,1,3,19,18]],dtype=np.int64)),
        )
        for n,e in cases:
            with self.subTest(n=n,edges=e.shape[1]):
                self.compare(previous_edge(n,e),
                             build_graph_from_edge_index(n,e))

    def test_advanced_numpy_indices_accept_int32_when_scipy_uses_int32(self):
        adj=csr_matrix(
            (np.ones(5,dtype=np.bool_),
             (np.array([0,1,1,2,2]),np.array([1,0,2,1,3]))),
            shape=(5,5),dtype=np.bool_)
        if adj.indices.dtype!=np.dtype("int32"):
            self.skipTest("SciPy explicitly selected wide index dtype on this host")
        deg=np.diff(adj.indptr).astype(np.int64)
        row=np.repeat(np.arange(5,dtype=adj.indices.dtype),deg)
        col=(np.arange(adj.indices.size,dtype=adj.indptr.dtype)
             -np.repeat(adj.indptr[:-1],deg))
        self.assertEqual(row.dtype,np.dtype("int32"))
        self.assertEqual(col.dtype,np.dtype("int32"))
        out=np.full((5,int(deg.max())),-1,dtype=np.int64)
        out[row,col]=adj.indices
        old=previous_scatter(adj,False)[0]
        np.testing.assert_array_equal(out,old)
        self.assertEqual(out.dtype,np.dtype("int64"))


if __name__=="__main__":
    unittest.main()
