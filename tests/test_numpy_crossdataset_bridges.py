"""Cross-dataset, independent Wine/Breast Cancer nested-bridge gates."""
import unittest
import numpy as np

from experiments.audit_numpy_crossdataset_bridges import (
    ARMS,DATASETS,DOSES,K,NCLASSES,NODES,PARTITION_SIZES,SEEDS,
    build_graphs,inputs,select_labels,
)
from experiments.audit_numpy_undirected_bridge_disconnect import (
    edges_from_neighbor_arrays,
)


class CrossDatasetBridgeTests(unittest.TestCase):
    def test_production_union_graphs_nested_on_both_real_datasets(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                x,truth,parts=inputs(dataset)
                graphs,bridges,baseline_sha=build_graphs(x,truth,parts)
                self.assertEqual(len(bridges),32)
                self.assertEqual(len(set(bridges)),32)
                self.assertEqual(len({v for edge in bridges for v in edge}),64)
                self.assertEqual(set(graphs),set(ARMS))
                internal=edges_from_neighbor_arrays(*graphs["B0"][0])
                self.assertEqual(graphs["B0"][1]["component_sizes"],
                                 PARTITION_SIZES[dataset])
                for dose in DOSES:
                    ((graph,degrees),meta,digest,allowed)=graphs[f"B{dose}"]
                    e=edges_from_neighbor_arrays(graph,degrees)
                    self.assertEqual(e,internal|set(bridges[:dose]))
                    self.assertEqual(allowed,bridges[:dose])
                    self.assertEqual(meta["cross_partition_edges"],dose)
                    self.assertEqual(meta["ncomponents"],2 if dose==0 else 1)
                    self.assertEqual(meta["undirected_edges"],
                                     graphs["B0"][1]["undirected_edges"]+dose)
                    self.assertEqual(meta["directed_edges"],
                                     graphs["B0"][1]["directed_edges"]+2*dose)
                    self.assertEqual(len(digest),64)
                ((graph,degrees),meta,digest,cross)=graphs["full_union10"]
                e=edges_from_neighbor_arrays(graph,degrees)
                self.assertEqual(meta["ncomponents"],1)
                self.assertGreater(len(cross),0)
                self.assertEqual(
                    set(cross),{edge for edge in e
                                if edge[0]%2!=edge[1]%2})
                self.assertEqual(len(digest),64)

    def test_each_class_has_one_supervised_vertex_per_component(self):
        for dataset in DATASETS:
            _,truth,parts=inputs(dataset)
            for seed in SEEDS:
                with self.subTest(dataset=dataset,seed=seed):
                    labels=select_labels(truth,seed)
                    nc=NCLASSES[dataset]
                    self.assertEqual(int(np.count_nonzero(labels!=-1)),2*nc)
                    for cls in range(nc):
                        for side in (0,1):
                            self.assertEqual(int(np.count_nonzero(
                                (labels==cls)&
                                (np.arange(len(labels))%2==side))),1)

    def test_previous_synthetic_crossing_contract_still_passes(self):
        # These parent tests are run independently by the workflow; here
        # assert native collector is inherited rather than redesigned.
        from experiments.audit_numpy_undirected_multibridge import (
            MultiBridgeObserver,
        )
        from experiments.audit_numpy_undirected_sparsity_mobility import (
            MobilityObserver,
        )
        self.assertTrue(issubclass(MultiBridgeObserver,MobilityObserver))


if __name__=="__main__":
    unittest.main()
