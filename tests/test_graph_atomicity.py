"""Graph preparation failures must not commit any instance fields."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from pcc import ParticleCompetitionAndCooperation as PCC
from tests import test_label_shape as label_tests


class GraphAtomicityTests(unittest.TestCase):
    backends=('numpy','numba','cython')

    def model(self,backend,stage='graph'):
        helper=label_tests.LabelShapeTests()
        if stage=='empty':return PCC(impl=backend,n_jobs=2)
        m=helper.model(backend);m.n_jobs=2
        if stage in ('fit','degenerate'):
            helper.fit_state(m,[0,1,-1])
        if stage=='degenerate':
            with self.assertWarns(UserWarning):m.fit_predict([0,-1,-1])
        return m

    def reject(self,m,operation):
        helper=label_tests.LabelShapeTests()
        old=dict(m.__dict__);values=copy.deepcopy(old)
        np.random.seed(19);label_tests.seed_native();random=np.random.get_state()
        with patch.object(m,'_get_backend_fn') as dispatch:
            with self.assertRaises((ValueError,TypeError,AttributeError,IndexError,OverflowError)):
                operation()
            dispatch.assert_not_called()
        self.assertEqual(m.__dict__.keys(),old.keys())
        for k in old:self.assertIs(m.__dict__[k],old[k],k)
        helper.equal_state(m.__dict__,values)
        now=np.random.get_state()
        np.testing.assert_array_equal(now[1],random[1]);self.assertEqual(now[2:],random[2:])
        draw=label_tests.draw_native();label_tests.seed_native();self.assertEqual(draw,label_tests.draw_native())

    def build_failures(self,m):
        data=np.array([[0.],[1.],[2.]])
        return [lambda:m.build_graph(data.ravel(),1),lambda:m.build_graph(data[None,:,:],1),
                lambda:m.build_graph(data,0),lambda:m.build_graph(data,3),
                lambda:m.build_graph(data,1.5),lambda:m.build_graph(data,'bad'),
                lambda:m.build_graph(np.array([[0.],[np.nan],[2.]]),1),
                lambda:m.build_graph(np.array([[0.],[np.inf],[2.]]),1),
                lambda:m.build_graph(np.array([['bad'],['data'],['here']]),1),
                lambda:m.build_graph(data,1,n_jobs='bad')]

    def set_failures(self,m):
        neighbors=np.zeros((4,1),dtype=np.int64);degrees=np.ones(4,dtype=np.int64)
        return [lambda:m.set_graph(neighbors,degrees,k_nn='bad'),
                lambda:m.set_graph(neighbors,np.array(['bad']*4)),
                lambda:m.set_graph(neighbors,np.empty(0,dtype=np.int64)),
                lambda:m.set_graph(np.full((4,1),'bad',dtype=object),degrees),
                lambda:m.set_graph(np.full((4,1),10**100,dtype=object),degrees),
                lambda:m.set_graph(np.full((4,1),np.nan),degrees),
                lambda:m.set_graph(neighbors,np.full(4,np.inf)),
                lambda:m.set_graph(neighbors,degrees,k_nn=np.nan),
                lambda:m.set_graph([[0],[0],[0]],np.ones(3,dtype=np.int64)),
                lambda:m.set_graph(np.array([1.]),np.ones(3,dtype=np.int64))]

    def test_build_failures_preserve_every_lifecycle_stage(self):
        for backend in self.backends:
            for stage in ('empty','graph','fit','degenerate'):
                m=self.model(backend,stage)
                for index,operation in enumerate(self.build_failures(m)):
                    with self.subTest(backend=backend,stage=stage,case=index):self.reject(m,operation)

    def test_set_failures_preserve_every_lifecycle_stage(self):
        for backend in self.backends:
            for stage in ('empty','graph','fit','degenerate'):
                m=self.model(backend,stage)
                for index,operation in enumerate(self.set_failures(m)):
                    with self.subTest(backend=backend,stage=stage,case=index):self.reject(m,operation)

    def test_recovery_after_rejected_graph_change(self):
        helper=label_tests.LabelShapeTests()
        for backend in self.backends:
            for method in ('build','set'):
                with self.subTest(backend=backend,method=method):
                    m=self.model(backend,'fit')
                    operation=(lambda:m.build_graph(np.ones((2,2)),k_nn=3)) if method=='build' else (
                        lambda:m.set_graph(np.zeros((4,1),dtype=np.int64),np.ones(4,dtype=np.int64),k_nn='bad'))
                    self.reject(m,operation)
                    a=helper.fit_state(m,[7,8,-1]);b=helper.fit_state(self.model(backend),[7,8,-1])
                    for x,y in zip(a[0],b[0]):np.testing.assert_array_equal(x,y)
                    self.assertEqual(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(a[3],b[3])

    def test_build_success_preserves_ownership_and_job_selection(self):
        data=np.array([[0.],[1.],[2.]])
        for jobs,expected in ((None,2),(1,1)):
            m=self.model('numpy','fit');old_node,old_part=m.node,m.part
            neighbors=np.array([[2],[2],[0]]);degrees=np.ones(3,dtype=np.int64)
            with patch('pcc.build_knn_graph',return_value=(neighbors,degrees)) as build:
                self.assertIsNone(m.build_graph(data,1,n_jobs=jobs))
            self.assertIs(m.data,data);self.assertIs(m.neib_list,neighbors);self.assertIs(m.neib_qt,degrees)
            self.assertEqual(m.k_nn,1);self.assertEqual(m.n_jobs,2)
            self.assertEqual(build.call_args.kwargs,{'n_jobs':expected})
            self.assertIs(m.node,old_node);self.assertIs(m.part,old_part)

    def test_set_success_preserves_copies_and_finite_coercions(self):
        m=self.model('numpy','fit');old_node,old_part=m.node,m.part
        neighbors=np.array([[2.9],[2.9],[0.9]])
        degrees=np.array([1.9]*3)
        for k in (None,1.9):
            self.assertIsNone(m.set_graph(neighbors,degrees,k_nn=k))
            np.testing.assert_array_equal(m.neib_list,neighbors.astype(np.int64))
            np.testing.assert_array_equal(m.neib_qt,degrees.astype(np.int64))
            self.assertFalse(np.shares_memory(m.neib_list,neighbors));self.assertFalse(np.shares_memory(m.neib_qt,degrees))
            self.assertIsNone(m.data);self.assertEqual(m.k_nn,1)
            self.assertIs(m.node,old_node);self.assertIs(m.part,old_part)
        # Preserve acceptance policy; this PR does not add structural/index validation.
        m.set_graph(np.array([[9999]],dtype=np.int64),np.array([2,0],dtype=np.int64),k_nn=-2)
        np.testing.assert_array_equal(m.neib_list,[[9999]]);self.assertEqual(m.k_nn,-2)

    def test_real_builder_and_setter_have_same_valid_fit(self):
        helper=label_tests.LabelShapeTests();data=np.array([[0.],[1.],[3.]])
        for backend in self.backends:
            with self.subTest(backend=backend):
                m=self.model(backend,'empty');m.build_graph(data,1)
                other=self.model(backend,'empty');other.set_graph(m.neib_list,m.neib_qt,k_nn=1)
                a=helper.fit_state(m,[0,1,-1]);b=helper.fit_state(other,[0,1,-1])
                for x,y in zip(a[0],b[0]):np.testing.assert_array_equal(x,y)
                self.assertEqual(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(a[3],b[3])
