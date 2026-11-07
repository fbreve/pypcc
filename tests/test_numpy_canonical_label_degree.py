"""Synthetic ranking and real canonical graph checks for label placement."""
import unittest

import numpy as np

from experiments.audit_numpy_canonical_label_degree import (
    DATASETS,SEEDS,STRATEGIES,CLASSES,ROWS,
    LABELS_PER_CLASS,class_pools,dataset_graph,select_label_masks,
    home_diagnostics,
)


class CanonicalDegreeHomeTests(unittest.TestCase):
    def test_ranked_degree_pools_use_id_ties_and_are_disjoint(self):
        truth=np.array([0]*12+[1]*12,dtype=np.int64)
        degree=np.array([12,10,10,15,20,14,10,20,20,12,14,10] * 2,
                        dtype=np.int64)
        p=class_pools(truth,degree)
        self.assertEqual(p[0]["low_degree_quartile"].tolist(),[1,2,6])
        self.assertEqual(p[0]["high_degree_quartile"].tolist(),[4,7,8])
        self.assertEqual(p[1]["low_degree_quartile"].tolist(),[13,14,18])
        self.assertEqual(p[1]["high_degree_quartile"].tolist(),[16,19,20])
        self.assertEqual(len(p[0]["random"]),12)
        a,common,meta=select_label_masks(truth,degree,24103)
        b,common2,meta2=select_label_masks(truth,degree,24103)
        self.assertTrue(all(
            np.array_equal(a[s],b[s]) for s in STRATEGIES))
        self.assertTrue(np.array_equal(common,common2))
        self.assertEqual(meta,meta2)
        self.assertEqual(int(np.sum(common)),len(truth)-
                         len(set(np.concatenate([
                             np.flatnonzero(labels!=-1)
                             for labels in a.values()]))))
        for strategy in STRATEGIES:
            self.assertEqual(int(np.sum(a[strategy]!=-1)),4)
            self.assertTrue(np.all(a[strategy][common]==-1))
            for cls in (0,1):
                ids=set(np.flatnonzero(a[strategy]==cls))
                self.assertEqual(len(ids),2)
                self.assertTrue(ids.issubset(set(p[cls][strategy])))

    def test_frozen_real_graphs_and_candidate_distribution(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                graph,degrees,truth,meta,sha=dataset_graph(dataset)
                self.assertEqual(meta["n_nodes"],ROWS[dataset])
                self.assertEqual(meta["ncomponents"],1)
                self.assertTrue(meta["reciprocal"])
                self.assertEqual(meta["component_sizes"],[ROWS[dataset]])
                self.assertGreaterEqual(meta["min_degree"],10)
                self.assertEqual(len(sha),64)
                p=class_pools(truth,degrees)
                for cls in range(CLASSES[dataset]):
                    eligible=np.flatnonzero(truth==cls)
                    count=max(2,int(np.ceil(len(eligible)/4)))
                    self.assertEqual(len(p[cls]["low_degree_quartile"]),count)
                    self.assertEqual(len(p[cls]["high_degree_quartile"]),count)
                    self.assertTrue(set(p[cls]["low_degree_quartile"]).isdisjoint(
                        set(p[cls]["high_degree_quartile"])))
                for seed in SEEDS:
                    masks,common,pool_json=select_label_masks(
                        truth,degrees,seed)
                    other,common2,_=select_label_masks(truth,degrees,seed)
                    self.assertTrue(np.array_equal(common,common2))
                    self.assertEqual(len(pool_json),CLASSES[dataset])
                    self.assertGreater(int(np.sum(common)),0)
                    for s in STRATEGIES:
                        labels=masks[s]
                        self.assertTrue(np.array_equal(labels,other[s]))
                        self.assertTrue(np.all(labels[common]==-1))
                        self.assertEqual(int(np.sum(labels!=-1)),
                                         2*CLASSES[dataset])
                        diag=home_diagnostics(graph,degrees,truth,labels)
                        self.assertEqual(len(diag["homes"]),2*CLASSES[dataset])
                        self.assertGreaterEqual(
                            diag["selected_degree_min"],10)
                        self.assertTrue(
                            0<=diag["selected_neighbor_label_agreement"]<=1)

    def test_common_evaluation_intersection_excludes_all_selected_homes(self):
        _,degrees,truth,_,_=dataset_graph("wine")
        masks,common,_=select_label_masks(truth,degrees,24223)
        selected=np.logical_or.reduce([
            labels!=-1 for labels in masks.values()])
        self.assertTrue(np.array_equal(common,~selected))
        self.assertEqual(int(np.sum(common)+np.sum(selected)),len(truth))
        self.assertTrue(all(np.all(labels[common]==-1)
                            for labels in masks.values()))


if __name__=="__main__":
    unittest.main()
