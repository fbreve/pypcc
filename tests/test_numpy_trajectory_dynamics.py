"""Synthetic semantics for every-iteration scientific trajectory collector."""
import unittest

import numpy as np

from experiments.audit_numpy_trajectory_dynamics import (
    MAX_ITER, StreamingDynamicsObserver, identical_raw, raw_state,
)


class StreamingTrajectoryTests(unittest.TestCase):
    def test_hidden_reversal_and_stationary_particle_are_separate(self):
        mapped=np.array([-1,0,1],dtype=np.int64)
        obs=StreamingDynamicsObserver(mapped,stop_iteration=301,stop_max=1)
        dominance=np.array([[.6,.4],[1.,0.],[0.,1.]],dtype=float)
        positions=np.array([1,2],dtype=np.int64)
        strengths=np.ones(2)
        distances=np.zeros((3,2),dtype=np.uint8)
        ownership=np.zeros((3,2),dtype=float)
        rng=np.random.RandomState(99).get_state()
        for it in range(MAX_ITER+1):
            dominance[0]=[.4,.6] if 20<=it<40 else [.6,.4]
            positions[0]=0 if it==30 else 1
            obs.capture(it,dominance,positions,strengths,
                        distances,ownership,rng,None)
        meta={"effective_iterations":MAX_ITER,"stop_reason":"max_iter"}
        obs.capture(MAX_ITER,dominance,positions,strengths,
                    distances,ownership,rng,meta)
        m=obs.summary()
        self.assertEqual(m["native_callbacks"],6002)
        self.assertEqual(m["full_windows"],60)
        self.assertEqual(m["step_flip_events"],2)
        self.assertEqual(m["step_movement_events"],2)
        self.assertEqual(m["zero_motion_steps"],5999)
        self.assertEqual(m["quiet_endpoint_windows"],60)
        self.assertEqual(m["quiet_but_hidden_windows"],1)
        self.assertEqual(m["reverted_node_window_events"],1)
        self.assertEqual(m["particles_never_changed_position"],1)
        self.assertEqual(m["shadow_first_stop_iteration"],21)
        self.assertEqual(obs.stop_checkpoint["iteration"],301)
        self.assertEqual(obs.final_info,meta)

    def test_final_only_upgrades_and_missing_iteration_rejected(self):
        mapped=np.array([-1,0],dtype=np.int64)
        obs=StreamingDynamicsObserver(mapped,stop_iteration=1,stop_max=1)
        dom=np.array([[.5,.5],[1.,0.]])
        pos=np.array([1],dtype=np.int64)
        strength=np.ones(1)
        dist=np.zeros((2,1),dtype=np.uint8)
        own=np.zeros((2,2))
        rng=np.random.RandomState(2).get_state()
        obs.capture(0,dom,pos,strength,dist,own,rng,None)
        with self.assertRaisesRegex(AssertionError,"Missing"):
            obs.capture(2,dom,pos,strength,dist,own,rng,None)

    def test_raw_comparison_includes_ownership_and_labels(self):
        mapped=np.array([-1,0,1],dtype=np.int64)
        dom=np.array([[.4,.6],[1.,0.],[0.,1.]])
        pos=np.array([1,2])
        strength=np.ones(2)
        dist=np.zeros((3,2),dtype=np.uint8)
        own=np.zeros((3,2))
        a=raw_state(dom,pos,strength,dist,own,mapped)
        b=raw_state(dom,pos,strength,dist,own,mapped)
        self.assertTrue(identical_raw(a,b))
        own[0,1]+=1.
        c=raw_state(dom,pos,strength,dist,own,mapped)
        self.assertFalse(identical_raw(a,c))


if __name__=="__main__":
    unittest.main()
