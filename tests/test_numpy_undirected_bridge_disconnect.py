"""Structural and per-iteration crossing checks for split/bridge experiment."""
import unittest

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from experiments.audit_numpy_undirected_bridge_disconnect import (
    build_variants, labelled_nodes, CrossingObserver,
)


class UndirectedBridgeTests(unittest.TestCase):
    def test_real_digit_graph_is_exact_one_edge_counterfactual(self):
        data=load_digits()
        x=StandardScaler().fit_transform(np.asarray(data.data,dtype=float))
        variants,edge=build_variants(x,np.asarray(data.target,dtype=np.int64))
        split=variants["split_disconnected"][1]
        joined=variants["single_bridge"][1]
        full=variants["full_union10"][1]
        self.assertEqual(split["component_sizes"],[899,898])
        self.assertEqual(split["ncomponents"],2)
        self.assertEqual(joined["ncomponents"],1)
        self.assertEqual(full["ncomponents"],1)
        self.assertEqual(joined["undirected_edges"]-split["undirected_edges"],1)
        self.assertEqual(edge[0]%2,0)
        self.assertEqual(edge[1]%2,1)
        self.assertTrue(all(v[1]["min_degree"]>0 for v in variants.values()))

    def test_class_homes_avoid_missing_classes_on_a_partition(self):
        truth=np.asarray(load_digits().target,dtype=np.int64)
        for seed in (21107,21221,21317,21401):
            labels=labelled_nodes(truth,seed)
            self.assertEqual(int(np.sum(labels!=-1)),20)
            for cls in range(10):
                for side in (0,1):
                    self.assertEqual(int(np.count_nonzero(
                        (labels==cls)&(np.arange(len(labels))%2==side))),1)

    def setup_sample(self,variant="single_bridge"):
        mapped=np.array([0,0,1,1],dtype=np.int64)
        # Reciprocal undirected intra-half edges (0--2,1--3), plus
        # precisely one 0--1 cross-partition bridge.
        graph=np.array([[2,1],[3,0],[0,-1],[1,-1]],dtype=np.int64)
        degrees=np.array([2,2,1,1],dtype=np.int64)
        observer=CrossingObserver(
            mapped,graph,degrees,variant,(0,1),cap=10,window=2)
        dom=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        pos=np.array([0,1,2,3],dtype=np.int64)
        strengths=np.array([1.,1.,1.,1.])
        distances=np.zeros((4,4),dtype=np.uint8)
        ownership=np.zeros((4,2),dtype=float)
        rng=np.random.RandomState(111).get_state()
        return observer,dom,pos,strengths,distances,ownership,rng

    def send(self,sample,iteration,info=None):
        observer,dom,pos,strength,dist,own,rng=sample
        observer.capture(iteration,dom,pos,strength,dist,own,rng,info)

    def test_cross_away_and_return_not_lost_by_window_endpoint(self):
        sample=self.setup_sample()
        obs,_,pos,*_=sample
        for it in range(11):
            if it==1:
                pos[0]=1
            if it==2:
                pos[0]=0
            self.send(sample,it)
        self.send(sample,10,dict(effective_iterations=10,
                                 stop_reason="max_iter"))
        summary=obs.summary()
        self.assertEqual(summary["physical_iterations"],11)
        self.assertEqual(summary["cross_partition_moves"],2)
        self.assertEqual(summary["cross_partition_moves_by_class"],[2,0])
        self.assertEqual(summary["cross_partition_moves_by_particle"],[2,0,0,0])
        self.assertEqual(summary["total_accepted_moves"],2)
        self.assertEqual(summary["n_windows"],5)
        self.assertEqual(summary["zero_move_class_windows"],9)
        with self.assertRaises(AssertionError):
            self.send(sample,10,dict(effective_iterations=10,
                                     stop_reason="max_iter"))

    def test_disconnected_cross_is_rejected_even_when_position_changes(self):
        sample=self.setup_sample("split_disconnected")
        self.send(sample,0)
        sample[2][0]=1
        with self.assertRaisesRegex(AssertionError,"crossed disconnected"):
            self.send(sample,1)

    def test_only_declared_bridge_may_transfer_between_partitions(self):
        sample=self.setup_sample()
        self.send(sample,0)
        sample[2][2]=3
        with self.assertRaisesRegex(AssertionError,"off the unique bridge"):
            self.send(sample,1)

    def test_skipped_iterations_rejected(self):
        sample=self.setup_sample()
        self.send(sample,0)
        with self.assertRaisesRegex(AssertionError,"Duplicate or missed"):
            self.send(sample,2)


if __name__=="__main__":
    unittest.main()
