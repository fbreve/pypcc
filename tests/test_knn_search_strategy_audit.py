"""Cheap semantic checks for exact-neighbor alternative auditing.

No git history or optional experiment tooling is needed in ordinary regression.
"""
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from experiments.audit_knn_search_strategies import (
    ALTERNATIVES,DATASETS,K_VALUES,METHODS,SYNTHETIC_NAMES,
    difference,edge_set,synthetic_inputs,engine_raw_order,inputs,
)
from pcc_graph import build_knn_graph


class NearestNeighborSearchAuditTests(unittest.TestCase):
    def test_frozen_algorithm_cohort_and_adversarial_inputs(self):
        self.assertEqual(METHODS,
                         ("auto","brute","kd_tree","ball_tree"))
        self.assertEqual(K_VALUES,(1,3,10))
        self.assertEqual(DATASETS,
                         ("iris","wine","breast_cancer","digits"))
        data=synthetic_inputs()
        self.assertEqual(tuple(data),SYNTHETIC_NAMES)
        for name,x in data.items():
            with self.subTest(name=name):
                self.assertEqual(x.shape[0],12)
                self.assertTrue(np.isfinite(x).all())
        self.assertTrue(data["fortran_noncontiguous"].flags.f_contiguous)
        self.assertEqual(len(ALTERNATIVES),3)

    def test_exact_comparison_includes_degree_dtype_padding_and_order(self):
        orig=(np.array([[1,2,-1],[0,2,3],[0,1,-1],[1,-1,-1]],
                       dtype=np.int64),
              np.array([2,3,2,1],dtype=np.int64))
        self.assertTrue(difference(orig,orig)["exact"])
        reversed_list=(orig[0].copy(),orig[1].copy())
        reversed_list[0][0,:2]=[2,1]
        diff=difference(orig,reversed_list)
        self.assertFalse(diff["exact"])
        self.assertEqual(diff["missing_edges"],0)
        self.assertEqual(diff["extra_edges"],0)
        changed_padding=(orig[0].copy(),orig[1].copy())
        changed_padding[0][0,2]=123
        self.assertFalse(difference(orig,changed_padding)["exact"])
        changed_degree=(orig[0].copy(),orig[1].copy())
        changed_degree[1][0]=1
        self.assertFalse(difference(orig,changed_degree)["exact"])
        added=(orig[0].copy(),orig[1].copy())
        added[0][0,:2]=[1,3]
        self.assertGreater(difference(orig,added)["missing_edges"],0)
        self.assertGreater(difference(orig,added)["extra_edges"],0)

    def test_alternative_nn_queries_are_real_sklearn_calls(self):
        # No approximation or changed query semantics; these engine names
        # are native sklearn.NearestNeighbors choices.
        x,_=inputs("iris")
        for k in K_VALUES:
            for method in METHODS:
                with self.subTest(k=k,method=method):
                    resolved,idx=engine_raw_order(x,k,method)
                    self.assertIn(resolved,ALTERNATIVES)
                    self.assertEqual(idx.shape,(len(x),k))
                    self.assertTrue(np.all((idx>=0)&(idx<len(x))))

    def test_original_production_graph_is_reciprocal(self):
        x,_=inputs("wine")
        for k in K_VALUES:
            graph,deg=build_knn_graph(x,k_nn=k,n_jobs=1)
            self.assertEqual(graph.dtype,np.dtype("int64"))
            self.assertEqual(deg.dtype,np.dtype("int64"))
            for i in range(len(deg)):
                for j in graph[i,:deg[i]]:
                    self.assertNotEqual(i,j)
                    self.assertIn(i,graph[j,:deg[j]])
                self.assertTrue(np.all(graph[i,deg[i]:]==-1))


if __name__=="__main__":
    unittest.main()
