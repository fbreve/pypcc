"""Fast synthetic tests for the bounded, no-RNG NumPy trajectory collector."""
import unittest

import numpy as np

from experiments.numpy_continuous_observer import (
    ContinuousNumpyObserver, clone_rng, rng_equal,
)


def synthetic_arrays():
    labels=np.array([-1,-1,0,1],dtype=np.int64)
    dominance=np.array(
        [[.5,.5],[.3,.7],[1.,0.],[0.,1.]],
        dtype=np.float64, order="F")
    positions=np.array([2,3],dtype=np.int64)
    strengths=np.ones(2,dtype=np.float64)
    distances=np.full((4,2),3,dtype=np.uint8)
    ownership=np.zeros((4,2),dtype=np.float64,order="F")
    rng=np.random.RandomState(7).get_state()
    return labels,dominance,positions,strengths,distances,ownership,rng


class TestNumpyContinuousObserver(unittest.TestCase):
    def test_initial_final_snapshot_copy_and_rng_state_are_immutable(self):
        labels,dom,pos,strength,dist,own,rng=synthetic_arrays()
        c=ContinuousNumpyObserver(labels,milestones=(0,1,11))
        first_rng=clone_rng(rng)
        c.capture(0,dom,pos,strength,dist,own,rng,None)
        before=c.samples[0]["state"]["dominance"].copy()
        dom[0,0]=.4
        dom[0,1]=.6
        pos[0]=1
        self.assertTrue(np.array_equal(
            c.samples[0]["state"]["dominance"],before))
        self.assertTrue(rng_equal(c.samples[0]["rng"],first_rng))
        self.assertTrue(c.wants(1))
        c.capture(1,dom,pos,strength,dist,own,rng,None)
        metadata=dict(
            effective_iterations=1,checks=0,stop_reason="max_iter",
            reached_max_iter=True,stop_max=1,stalled_checks=0)
        c.capture(1,dom,pos,strength,dist,own,rng,metadata)
        self.assertEqual(len(c.samples),2)
        self.assertEqual(c.final_info,metadata)
        self.assertEqual([r["iteration"] for r in c.summarize()],[0,1])
        self.assertIsNone(c.samples[0]["net_assignment_changes"])
        self.assertTrue(c.samples[-1]["final_info"])
        with self.assertRaisesRegex(AssertionError,"after finalization"):
            c.capture(2,dom,pos,strength,dist,own,rng,None)

    def test_final_unscheduled_is_captured_once(self):
        labels,dom,pos,strength,dist,own,rng=synthetic_arrays()
        c=ContinuousNumpyObserver(labels,milestones=(0,11))
        c.capture(0,dom,pos,strength,dist,own,rng,None)
        c.capture(7,dom,pos,strength,dist,own,rng,dict(
            effective_iterations=7,stop_reason="early_stop"))
        self.assertEqual([s["iteration"] for s in c.samples],[0,7])

    def test_bounds_and_no_corrupted_final_reuse(self):
        labels,dom,pos,strength,dist,own,rng=synthetic_arrays()
        with self.assertRaises(ValueError):
            ContinuousNumpyObserver(labels,milestones=(0,1,1))
        with self.assertRaises(ValueError):
            ContinuousNumpyObserver(labels,milestones=(0,1,2),
                                     max_observations=2)
        small=ContinuousNumpyObserver(labels,milestones=(0,),max_bytes=64)
        with self.assertRaisesRegex(ValueError,"byte budget"):
            small.capture(0,dom,pos,strength,dist,own,rng,None)
        good=ContinuousNumpyObserver(labels,milestones=(0,1))
        good.capture(0,dom,pos,strength,dist,own,rng,None)
        dom[0,0]=.6
        with self.assertRaisesRegex(AssertionError,"finalization changed"):
            good.capture(0,dom,pos,strength,dist,own,rng,
                         dict(effective_iterations=0))

    def test_rng_clone_and_comparison_include_continuation_state(self):
        _,_,_,_,_,_,rng=synthetic_arrays()
        copy=clone_rng(rng)
        self.assertTrue(rng_equal(copy,rng))
        changed=(rng[0],rng[1].copy(),rng[2]+1,rng[3],rng[4])
        self.assertFalse(rng_equal(copy,changed))
        other=(rng[0],rng[1].copy(),rng[2],rng[3],rng[4])
        other[1][0]^=np.uint32(1)
        self.assertFalse(rng_equal(copy,other))


if __name__=="__main__":
    unittest.main()
