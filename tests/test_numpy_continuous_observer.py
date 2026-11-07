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

    def test_failing_observer_aborts_before_first_particle_update(self):
        from pcc import ParticleCompetitionAndCooperation as PCC
        from pcc_numpy import pcc_propagate_numpy

        views = []
        calls = []

        class FailFirst:
            def wants(self, iteration):
                return iteration == 0

            def capture(self, *args):
                calls.append(args[0])
                raise RuntimeError("synthetic observer failure")

        class FailingObservedPCC(PCC):
            def _get_backend_fn(self):
                native = PCC._get_backend_fn(self)
                if native is not pcc_propagate_numpy:
                    raise AssertionError("NumPy backend unavailable")

                def fail(*args, **kwargs):
                    # Capture writable native buffers by reference plus copies;
                    # fit_predict rolls its own model fields back on failure.
                    buffers = [args[i] for i in (7, 9, 10, 11, 12)]
                    views.append((buffers, [x.copy() for x in buffers]))
                    return native(*args, observer=FailFirst(), **kwargs)

                fail.__name__ = native.__name__
                return fail

        model = FailingObservedPCC(impl="numpy")
        graph = np.array([
            [1,3],[0,2],[1,3],[0,2]], dtype=np.int64)
        model.set_graph(graph, np.full(4,2,dtype=np.int64),k_nn=2)
        with self.assertRaisesRegex(RuntimeError,"synthetic observer failure"):
            model.fit_predict(
                np.array([0,-1,-1,1],dtype=np.int64),
                seed=73,max_iter=20,early_stop=True)
        self.assertEqual(calls,[0])
        self.assertEqual(len(views),1)
        refs,original = views[0]
        for current,expected in zip(refs,original):
            self.assertTrue(
                np.array_equal(current,expected),
                "The failed initial callback must precede particle changes")


    def test_failed_capture_does_not_reserve_bytes_or_append_partial_row(self):
        labels,dom,pos,strength,dist,own,rng=synthetic_arrays()
        trial=ContinuousNumpyObserver(labels,milestones=(0,1))
        trial.capture(0,dom,pos,strength,dist,own,rng,None)
        row_bytes=trial.used_bytes
        self.assertEqual(row_bytes,sum(
            a.nbytes for a in trial.samples[0]["state"].values())
            + trial.samples[0]["rng"][1].nbytes)
        c=ContinuousNumpyObserver(
            labels,milestones=(0,1),max_bytes=2*row_bytes-1)
        c.capture(0,dom,pos,strength,dist,own,rng,None)
        with self.assertRaisesRegex(ValueError,"byte budget"):
            c.capture(1,dom,pos,strength,dist,own,rng,None)
        self.assertEqual(c.used_bytes,row_bytes)
        self.assertEqual([r["iteration"] for r in c.samples],[0])
        self.assertIsNone(c.final_info)
        invalid=ContinuousNumpyObserver(labels,milestones=(0,))
        broken=dom.copy()
        broken[0,0]=np.nan
        with self.assertRaisesRegex(AssertionError,"non-finite"):
            invalid.capture(0,broken,pos,strength,dist,own,rng,None)
        self.assertEqual(invalid.used_bytes,0)
        self.assertEqual(len(invalid.samples),0)

    def test_failure_after_first_iteration_is_not_atomic(self):
        from pcc import ParticleCompetitionAndCooperation as PCC
        from pcc_numpy import pcc_propagate_numpy

        captured=[]
        visited=[]

        class FailAfterOne:
            def wants(self,iteration):
                return iteration==1

            def capture(self,*args):
                visited.append(args[0])
                raise RuntimeError("intentional post-mutation observer failure")

        class FailingObservedPCC(PCC):
            def _get_backend_fn(self):
                native=PCC._get_backend_fn(self)
                if native is not pcc_propagate_numpy:
                    raise AssertionError("wrong backend")
                def fail(*args,**kwargs):
                    buffers=[args[i] for i in (7,9,10,11,12)]
                    captured.append((buffers,[x.copy() for x in buffers]))
                    return native(*args,observer=FailAfterOne(),**kwargs)
                return fail

        model=FailingObservedPCC(impl="numpy")
        graph=np.array([[1,3],[0,2],[1,3],[0,2]],dtype=np.int64)
        model.set_graph(graph,np.full(4,2,dtype=np.int64),k_nn=2)
        np.random.seed(73)
        with self.assertRaisesRegex(RuntimeError,"post-mutation"):
            model.fit_predict(np.array([0,-1,-1,1],dtype=np.int64),
                              seed=73,max_iter=20,early_stop=False)
        self.assertEqual(visited,[1])
        self.assertIsNone(model.part) # Existing high-level attribute rollback
        refs,before=captured[0]
        self.assertTrue(any(not np.array_equal(now,old)
                            for now,old in zip(refs,before)),
                        "A real particle step must have occurred")
        self.assertFalse(rng_equal(np.random.get_state(),
                                   np.random.RandomState(73).get_state()),
                         "Global RNG is not rolled back after a failed callback")

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
