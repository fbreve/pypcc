import unittest
import numpy as np
from experiments.convergence_measurements import maxima_gaps, norms, distribution, hidden_changes, is_checkpoint


class DiagnosticTests(unittest.TestCase):
    def test_reading_rng_and_calculations_do_not_draw_or_modify_rng(self):
        from numba import _helperlib
        from test_propagation_metrics import seed_only
        from test_sequential_safety import next_random
        seed_only(123)
        pointer=_helperlib.rnd_get_np_state_ptr()
        before=_helperlib.rnd_get_state(pointer)
        a=np.array([[.7,.3],[.4,.6]])
        distribution(a,np.array([True,True]))
        norms(a,a)
        self.assertEqual(before,_helperlib.rnd_get_state(pointer))
        observed=next_random()
        seed_only(123)
        self.assertEqual(observed,next_random())

    def test_top_two_gap_distribution_and_ties(self):
        a=np.array([[.8,.1,.1],[.5,.5,0],[.34,.33,.33]])
        maxima,gaps=maxima_gaps(a)
        np.testing.assert_allclose(maxima,[.8,.5,.34])
        np.testing.assert_allclose(gaps,[.7,0,.01])
        d=distribution(a,np.array([False,True,True]))
        self.assertEqual(d['low_gap_fraction_unlabeled'],1.)
        self.assertAlmostEqual(d['low_gap_fraction'],2/3)
        self.assertEqual(sum(d[key] for key in d if key.startswith('max_')),3)

    def test_matrix_norms_and_hidden_reversal(self):
        a=np.array([[.7,.3],[.2,.8]])
        b=np.array([[.5,.5],[.2,.8]])
        n=norms(a,b)
        self.assertAlmostEqual(n['matrix_mean_abs'],.1)
        self.assertAlmostEqual(n['matrix_frobenius'],np.sqrt(.08))
        self.assertAlmostEqual(n['matrix_max_abs'],.2)
        self.assertAlmostEqual(n['max_node_l1'],.4)
        self.assertEqual(hidden_changes(np.array([0,1,0]),np.array([0,0,0]),np.array([True,True,False])),1)

    def test_checkpoint_grid_and_fine_event_window(self):
        self.assertTrue(is_checkpoint(501,7981))
        self.assertFalse(is_checkpoint(511,7981))
        self.assertTrue(is_checkpoint(7481,7981))
        self.assertTrue(is_checkpoint(7981,7981))
        self.assertTrue(is_checkpoint(8481,7981))
        self.assertFalse(is_checkpoint(8491,7981))
        self.assertTrue(is_checkpoint(25011,7981))
        self.assertTrue(is_checkpoint(30000,7981))


if __name__=='__main__':unittest.main()
