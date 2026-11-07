"""Synthetic and real-structure gates for frozen nested bridge doses."""
import unittest
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from experiments.audit_numpy_undirected_multibridge import (
    DOSES,SEEDS,make_graphs,MultiBridgeObserver,
)
from experiments.audit_numpy_undirected_bridge_disconnect import (
    labelled_nodes,edges_from_neighbor_arrays,
)


class NestedBridgeTests(unittest.TestCase):
    def test_real_graphs_preserve_internal_edges_and_nested_one_edge_reference(self):
        digits=load_digits()
        truth=np.asarray(digits.target,dtype=np.int64)
        x=StandardScaler().fit_transform(np.asarray(digits.data,dtype=float))
        graphs,bridges,digest=make_graphs(x,truth)
        self.assertEqual(len(bridges),32)
        self.assertEqual(len(set(bridges)),32)
        self.assertEqual(len({v for edge in bridges for v in edge}),64)
        internal=edges_from_neighbor_arrays(*graphs[0][0])
        self.assertEqual(graphs[0][1]["component_sizes"],[899,898])
        for b in DOSES:
            graph,meta,sha=graphs[b]
            e=edges_from_neighbor_arrays(*graph)
            self.assertEqual(e,internal|set(bridges[:b]))
            self.assertEqual(len(e)-len(internal),b)
            self.assertEqual(meta["ncomponents"],2 if b==0 else 1)
            self.assertEqual(meta["directed_edges"],
                             graphs[0][1]["directed_edges"]+2*b)
        self.assertEqual(bridges[0] in edges_from_neighbor_arrays(*graphs[1][0]),True)

    def test_same_class_home_per_parity_side_for_all_new_seeds(self):
        truth=np.asarray(load_digits().target,dtype=np.int64)
        for seed in SEEDS:
            mask=labelled_nodes(truth,seed)
            self.assertEqual(int(np.count_nonzero(mask!=-1)),20)
            for cls in range(10):
                for parity in (0,1):
                    self.assertEqual(np.count_nonzero(
                        (mask==cls)&(np.arange(len(mask))%2==parity)),1)

    def sample(self,allowed,cap=10,window=2):
        mapped=np.array([0,0,1,1],dtype=np.int64)
        graph=np.array([[1,2],[0,3],[0,3],[1,2]],dtype=np.int64)
        deg=np.full(4,2,dtype=np.int64)
        obs=MultiBridgeObserver(mapped,graph,deg,allowed,cap=cap,window=window)
        dom=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        positions=np.array([0,1,2,3],dtype=np.int64)
        strengths=np.ones(4)
        dist=np.zeros((4,4),dtype=np.uint8)
        ownership=np.zeros((4,2))
        state=np.random.RandomState(5).get_state()
        return obs,dom,positions,strengths,dist,ownership,state

    def send(self,sample,it,info=None):
        obs,dom,pos,strength,dist,own,rng=sample
        obs.capture(it,dom,pos,strength,dist,own,rng,info)

    def test_out_and_back_counts_every_native_crossing_not_endpoints(self):
        s=self.sample([(0,1)])
        obs,_,pos,*_=s
        for it in range(11):
            if it==1:
                pos[0]=1
            if it==2:
                pos[0]=0
            self.send(s,it)
        self.send(s,10,dict(effective_iterations=10,stop_reason="max_iter"))
        d=obs.summary()
        self.assertEqual(d["physical_iterations"],11)
        self.assertEqual(d["cross_partition_moves"],2)
        self.assertEqual(d["cross_partition_moves_by_particle"],[2,0,0,0])
        self.assertEqual(d["cross_partition_moves_by_class"],[2,0])
        self.assertEqual(d["crossing_count_by_bridge"],{"0-1":2})
        self.assertEqual(d["first_crossing_iteration"],1)
        self.assertEqual(d["particles_ever_crossed"],1)
        self.assertEqual(d["bridges_ever_traversed"],1)
        self.assertEqual(d["cross_partition_windows"][0]["total"],2)
        self.assertEqual(sum(x["total"] for x in d["cross_partition_windows"]),2)
        self.assertEqual(d["total_accepted_moves"],2)
        with self.assertRaisesRegex(AssertionError,"Invalid final"):
            self.send(s,10,dict(effective_iterations=10))

    def test_forbidden_edge_detected_and_zero_bridge_graph_never_crosses(self):
        s=self.sample([],cap=2,window=1)
        self.send(s,0)
        s[2][0]=1
        with self.assertRaisesRegex(AssertionError,"Forbidden cross"):
            self.send(s,1)
        s=self.sample([(0,1)],cap=2,window=1)
        self.send(s,0)
        s[2][2]=3
        with self.assertRaisesRegex(AssertionError,"Forbidden cross"):
            self.send(s,1)

    def test_skipped_callbacks_rejected(self):
        s=self.sample([],cap=2,window=1)
        self.send(s,0)
        with self.assertRaisesRegex(AssertionError,"Duplicate or missed"):
            self.send(s,2)


if __name__=="__main__":
    unittest.main()
