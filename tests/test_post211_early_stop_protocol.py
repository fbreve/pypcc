"""Low-cost checks of frozen current-master stop-on/fixed schedule and provenance."""
import unittest

from benchmarks.fixtures import Case, REAL_DATASETS
from benchmarks.runner import PARAMETERS
from experiments.audit_post211_early_stop import (
    BASE, DATASETS, SEEDS, BACKENDS, MODES, ITERATIONS,
    PROTECTED, REPS, original_source_guard)


class Post211EarlyStopProtocolTests(unittest.TestCase):
    def test_precise_workload_and_one_original_graph_per_dataset(self):
        self.assertEqual(BASE,"d25d5446b7fa51cc0264a496b7eea87dbb11f1fc")
        self.assertEqual(DATASETS,("wine","breast_cancer","digits"))
        self.assertEqual(SEEDS,(45013,45117))
        self.assertEqual(BACKENDS,("numpy","numba","cython"))
        self.assertEqual(MODES,("early_stopping","fixed"))
        self.assertEqual(ITERATIONS,30000)
        self.assertEqual(REPS,3)
        self.assertEqual(PARAMETERS,
                         dict(p_grd=.5,delta_v=.1,deltap=1.,dexp=2.,es_chk=2000))
        ncases=0
        for ds in DATASETS:
            nodes,feats,classes,_=REAL_DATASETS[ds]
            for seed in SEEDS:
                c=Case(ds,nodes,feats,classes,k_nn=10,labels_per_class=3,
                       data_seed=0,label_seed=seed)
                c.validate()
                self.assertEqual(c.particles,3*classes)
                ncases+=1
        self.assertEqual(ncases,6)
        self.assertEqual(ncases*len(BACKENDS)*len(MODES)*REPS,108)
        self.assertEqual(ncases*len(BACKENDS)*2,36)
        self.assertEqual(ncases*len(BACKENDS),18)
        self.assertIn("pcc.py",PROTECTED)
        self.assertIn("benchmarks",PROTECTED)

    def test_balance_across_repetitions(self):
        for rep in range(REPS):
            order=BACKENDS[rep:]+BACKENDS[:rep]
            self.assertEqual(set(order),set(BACKENDS))
            self.assertEqual(len(order),3)
            stop_order=MODES if rep%2==0 else MODES[::-1]
            self.assertEqual(set(stop_order),set(MODES))
        for backend in BACKENDS:
            self.assertEqual(sum((BACKENDS[r:]+BACKENDS[:r])[0]==backend
                                 for r in range(REPS)),1)

    def test_source_guard_identifies_experiment_only_changes(self):
        original_source_guard()


if __name__=="__main__":
    unittest.main()
