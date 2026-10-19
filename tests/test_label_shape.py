"""Public label structure must be checked before state changes or dispatch."""
import copy
import unittest
import warnings
from unittest.mock import patch
import numpy as np
from numba import njit
from pcc import ParticleCompetitionAndCooperation as PCC


@njit
def seed_native():
    np.random.seed(19)


@njit
def draw_native():
    return np.random.random()


class LabelShapeTests(unittest.TestCase):
    backends = ('numpy', 'numba', 'cython')

    def model(self, backend):
        m = PCC(impl=backend)
        self.assertEqual(m._get_backend_fn().__module__,
                         {'numpy':'pcc_numpy','numba':'pcc_numba','cython':'pcc_step'}[backend])
        m.set_graph(np.array([[2],[2],[0]]),np.ones(3,dtype=np.int64))
        return m

    def equal_state(self, actual, expected):
        if isinstance(actual,np.ndarray):
            np.testing.assert_array_equal(actual,expected)
        elif isinstance(actual,dict):
            self.assertEqual(actual.keys(),expected.keys())
            for k in actual:self.equal_state(actual[k],expected[k])
        elif hasattr(actual,'__dict__'):
            self.equal_state(actual.__dict__,expected.__dict__)
        else:self.assertEqual(actual,expected)

    def reject(self,m,labels):
        old=dict(m.__dict__);values=copy.deepcopy(old)
        np.random.seed(19);seed_native();numpy_state=np.random.get_state()
        with patch.object(m,'_get_backend_fn') as dispatch, patch(
                'pcc.np.ascontiguousarray', side_effect=AssertionError('unexpected label conversion')) as convert:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with self.assertRaisesRegex(ValueError,'labels.*one-dimensional.*3'):
                    m.fit_predict(labels,max_iter=2,early_stop=False)
            dispatch.assert_not_called()
            convert.assert_not_called()
        self.assertEqual(m.__dict__.keys(),old.keys())
        for k in old:self.assertIs(m.__dict__[k],old[k],k)
        self.equal_state(m.__dict__,values)
        actual=np.random.get_state()
        np.testing.assert_array_equal(actual[1],numpy_state[1]);self.assertEqual(actual[2:],numpy_state[2:])
        next_draw=draw_native();seed_native();self.assertEqual(next_draw,draw_native())

    def cases(self):
        return ([7,8], [7,8,-1,-1], [], [[7],[8],[-1]],
                np.array([7,8]),np.array([7,8,-1,-1]),np.empty(0),
                np.array([[7,8,-1]]),np.array([[[7,8,-1]]]),np.array(7),7,
                [[np.nan],[np.inf],[-1]],np.array([np.nan,np.inf]),
                np.array([[np.nan,np.inf,-1]]),[-1],[[7,7,7]])

    def test_invalid_shapes_on_first_call_are_atomic(self):
        for backend in self.backends:
            m=self.model(backend)
            for labels in self.cases():
                with self.subTest(backend=backend,shape=np.shape(labels)):
                    self.reject(m,labels)

    def test_invalid_shapes_after_valid_fit_preserve_all_state(self):
        for backend in self.backends:
            m=self.model(backend)
            m.fit_predict([0,1,-1],max_iter=5,early_stop=False)
            for labels in self.cases():
                with self.subTest(backend=backend,shape=np.shape(labels)):
                    self.reject(m,labels)

    def fit_state(self,m,labels):
        fn=m._get_backend_fn();info={}
        def capture(*args):info.update(fn(*args,return_info=True))
        np.random.seed(19);seed_native()
        with patch.object(m,'_get_backend_fn',return_value=capture):
            prediction=m.fit_predict(labels,max_iter=35,early_stop=True,es_chk=1)
        return ([a.copy() for a in (prediction,m.node.dominance,m.part.curnode,
                m.part.strength,m.part.dist_table,m.owndeg)],info,np.random.random(4),draw_native())

    def test_recovery_and_valid_list_array_parity(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                m=self.model(backend)
                self.fit_state(m,[0,1,-1])
                self.reject(m,[7,8])
                a=self.fit_state(m,[7,8,-1])
                b=self.fit_state(self.model(backend),np.array([7,8,-1]))
                for x,y in zip(a[0],b[0]):np.testing.assert_array_equal(x,y)
                self.assertEqual(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(a[3],b[3])

    def test_valid_degenerate_labels_still_return_original_and_clear(self):
        for backend in self.backends:
            m=self.model(backend)
            for labels in ([-1]*3,np.array([7,-1,-1])):
                self.fit_state(m,[0,1,-1])
                with self.assertWarnsRegex(UserWarning,'Returning original labels'):
                    self.assertIs(m.fit_predict(labels),labels)
                self.assertIsNone(m.part);self.assertIsNone(m.node)
                np.testing.assert_array_equal(m.labels,labels)
