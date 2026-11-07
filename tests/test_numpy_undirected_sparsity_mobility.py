"""Synthetic streaming observer movement, lockout and callback-order tests."""
import unittest

import numpy as np

from experiments.audit_numpy_undirected_sparsity_mobility import MobilityObserver
from experiments.numpy_continuous_observer import rng_equal


class SyntheticMobilityTests(unittest.TestCase):
    def data(self,cap=12,window=2):
        mapped=np.array([0,0,1,1],dtype=np.int64)
        graph=np.array([[1,3],[0,2],[1,3],[2,0]],dtype=np.int64)
        degree=np.full(4,2,dtype=np.int64)
        dom=np.array([[1.,0.]]*4,dtype=np.float64)
        positions=np.array([0,1,2,3],dtype=np.int64)
        strengths=np.array([1.,1.,.005,.005],dtype=np.float64)
        distances=np.zeros((4,4),dtype=np.uint8)
        own=np.zeros((4,2),dtype=np.float64)
        rng=np.random.RandomState(2).get_state()
        obs=MobilityObserver(mapped,graph,degree,cap=cap,window=window)
        return obs,dom,positions,strengths,distances,own,rng

    def notify(self,obs,iteration,dom,pos,strength,dist,own,rng,info=None):
        obs.capture(iteration,dom,pos,strength,dist,own,rng,info)

    def test_out_and_back_hidden_at_window_endpoint_and_class_lockout(self):
        obs,dom,pos,strength,dist,own,rng=self.data()
        initial=pos.copy()
        for it in range(13):
            if it==1:
                pos[0]=1
            elif it==2:
                pos[0]=0
            self.notify(obs,it,dom,pos,strength,dist,own,rng)
        self.notify(obs,12,dom,pos,strength,dist,own,rng,
                    dict(effective_iterations=12,stop_reason="max_iter"))
        m=obs.summary()
        self.assertTrue(np.array_equal(pos,initial))
        self.assertEqual(m["total_accepted_moves"],2)
        self.assertEqual(m["total_moves_by_class"],[2,0])
        self.assertEqual(m["zero_movement_steps"],10)
        self.assertEqual(m["physical_iterations"],13)
        self.assertEqual(m["n_windows"],6)
        self.assertEqual(m["operationally_locked_classes"],1)
        self.assertEqual(m["first_class_lockout"],[None,10])
        self.assertEqual(m["first_recovered_move"],[None,None])
        self.assertEqual(m["never_moved_particles"],3)
        self.assertEqual(m["longest_individual_streak"],12)
        self.assertEqual(len(m["windows"]),12)
        self.assertEqual(m["windows"][0]["accepted_moves"],2)
        self.assertEqual(m["windows"][1]["accepted_moves"],0)
        self.assertTrue(m["windows"][9]["operational_lockout"])
        self.assertTrue(rng_equal(obs.final["rng"],rng))
        with self.assertRaisesRegex(AssertionError,"after true finalization"):
            self.notify(obs,12,dom,pos,strength,dist,own,rng)

    def test_recovery_after_operational_event_is_recorded(self):
        obs,dom,pos,strength,dist,own,rng=self.data()
        for it in range(13):
            if it==11:
                pos[2]=1
            self.notify(obs,it,dom,pos,strength,dist,own,rng)
        self.notify(obs,12,dom,pos,strength,dist,own,rng,
                    dict(effective_iterations=12,stop_reason="max_iter"))
        m=obs.summary()
        self.assertEqual(m["first_class_lockout"][1],10)
        self.assertEqual(m["first_recovered_move"][1],12)
        self.assertEqual(m["total_moves_by_class"],[0,1])

    def test_missing_iteration_and_duplicate_final_rejected(self):
        obs,dom,pos,strength,dist,own,rng=self.data()
        self.notify(obs,0,dom,pos,strength,dist,own,rng)
        with self.assertRaisesRegex(AssertionError,"Duplicate or missed"):
            self.notify(obs,2,dom,pos,strength,dist,own,rng)
        with self.assertRaisesRegex(AssertionError,"Invalid final"):
            self.notify(obs,12,dom,pos,strength,dist,own,rng,
                        dict(effective_iterations=12))

    def test_invalid_window_parameters_rejected(self):
        mapped=np.array([0,0,1,1])
        graph=np.array([[1,3],[0,2],[1,3],[2,0]])
        degrees=np.full(4,2)
        for cap,window in ((0,1),(5,0),(5,2)):
            with self.subTest(cap=cap,window=window),self.assertRaises(ValueError):
                MobilityObserver(mapped,graph,degrees,cap=cap,window=window)


if __name__=="__main__":
    unittest.main()
