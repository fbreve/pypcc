"""Installing a graph ends the previous fit, without mutating retained objects."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from pcc import ParticleCompetitionAndCooperation as PCC
from tests import test_label_shape as label_tests
from tests import test_graph_atomicity as atomic_tests


class GraphFitInvalidationTests(unittest.TestCase):
    fields=('labels','unique_labels','c','p_grd','delta_v','deltap','dexp',
            'max_iter','early_stop','es_chk','part','node','zerovec','owndeg',
            'label_to_idx','idx_to_label')
    dynamic=('mapped_labels','dist_weights','buf_dom_row','buf_reduc','buf_dom_list',
             'buf_dist_list','buf_prob','buf_slices')
    backends=('numpy','numba','cython')

    def model(self,backend,stage='fit'):
        m=PCC(impl=backend,n_jobs=1)
        self.assertEqual(m._get_backend_fn().__module__,
                         {'numpy':'pcc_numpy','numba':'pcc_numba','cython':'pcc_step'}[backend])
        if stage=='empty':return m
        m.set_graph(np.array([[1],[0],[3],[2]]),np.ones(4,dtype=np.int64))
        if stage in ('fit','degenerate'):
            self.fit(m,np.array([0,-1,1,-1]))
        if stage=='degenerate':
            with self.assertWarns(UserWarning):m.fit_predict([0,-1,-1,-1])
        return m

    def fit(self,m,labels):
        fn=m._get_backend_fn();info={}
        def capture(*args):info.update(fn(*args,return_info=True))
        np.random.seed(19);label_tests.seed_native()
        with patch.object(m,'_get_backend_fn',return_value=capture):
            prediction=m.fit_predict(labels,max_iter=15,early_stop=True,es_chk=1)
        arrays=[a.copy() for a in (prediction,m.node.label,m.node.dominance,
                m.part.homenode,m.part.curnode,m.part.label,m.part.strength,m.part.dist_table,
                m.mapped_labels,m.owndeg,m.dist_weights)]
        return arrays,info,np.random.random(4),label_tests.draw_native()

    def install(self,m,method,variant):
        data=np.array({'same':[0.,1.,10.,11.],'rewired':[0.,10.,1.,11.],
                       'larger':[0.,1.,10.,11.,20.,21.],'smaller':[0.,1.],
                       'degree':[0.,1.,10.,11.]}[variant])[:,None]
        k=3 if variant=='degree' else 1
        if method=='build':
            self.assertIsNone(m.build_graph(data,k))
            self.assertIs(m.data,data)
        else:
            n=len(data)
            if variant=='rewired':neighbors=np.array([[2],[3],[0],[1]])
            elif variant=='degree':neighbors=np.array([[j for j in range(n) if i!=j] for i in range(n)])
            else:neighbors=np.array([[i+1 if i%2==0 else i-1] for i in range(n)])
            degrees=np.full(n,k,dtype=np.int64)
            self.assertIsNone(m.set_graph(neighbors,degrees,k_nn=k))
            self.assertIsNone(m.data)
            np.testing.assert_array_equal(m.neib_list,neighbors)
            np.testing.assert_array_equal(m.neib_qt,degrees)
            self.assertFalse(np.shares_memory(m.neib_list,neighbors))
            self.assertFalse(np.shares_memory(m.neib_qt,degrees))
        self.assertEqual(m.k_nn,k)
        return len(data)

    def assert_cleared(self,m):
        for name in self.fields:self.assertIsNone(getattr(m,name),name)
        for name in self.dynamic:self.assertFalse(hasattr(m,name),name)

    def test_success_invalidates_all_fit_fields_without_mutating_old_objects(self):
        helper=label_tests.LabelShapeTests()
        for backend in self.backends:
            for method in ('build','set'):
                for variant in ('same','rewired','larger','smaller','degree'):
                    for stage in ('empty','graph','fit','degenerate'):
                        with self.subTest(backend=backend,method=method,variant=variant,stage=stage):
                            m=self.model(backend,stage)
                            old={name:getattr(m,name) for name in self.fields+self.dynamic if hasattr(m,name)}
                            self.assertEqual(set(m.__dict__)-{'impl','update_mode','n_jobs','data','k_nn','neib_list','neib_qt'},set(old))
                            values=copy.deepcopy(old);config=(m.impl,m.update_mode,m.n_jobs)
                            np.random.seed(19);label_tests.seed_native();rng=np.random.get_state()
                            with patch.object(m,'_get_backend_fn') as dispatch:
                                self.install(m,method,variant)
                                dispatch.assert_not_called()
                            self.assert_cleared(m)
                            helper.equal_state(old,values)
                            self.assertEqual((m.impl,m.update_mode,m.n_jobs),config)
                            now=np.random.get_state()
                            np.testing.assert_array_equal(now[1],rng[1]);self.assertEqual(now[2:],rng[2:])
                            draw=label_tests.draw_native();label_tests.seed_native();self.assertEqual(draw,label_tests.draw_native())

    def test_repeated_installation_is_idempotent_and_refit_matches_fresh(self):
        for backend in self.backends:
            for method in ('build','set'):
                for variant in ('same','rewired','larger','smaller','degree'):
                    with self.subTest(backend=backend,method=method,variant=variant):
                        m=self.model(backend)
                        n=self.install(m,method,variant);self.assert_cleared(m)
                        self.install(m,method,variant);self.assert_cleared(m)
                        labels=np.full(n,-1,dtype=np.int64);labels[0]=7;labels[-1]=12
                        a=self.fit(m,labels)
                        fresh=self.model(backend,'empty');self.install(fresh,method,variant)
                        b=self.fit(fresh,labels)
                        for x,y in zip(a[0],b[0]):np.testing.assert_array_equal(x,y)
                        self.assertEqual(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(a[3],b[3])

    def test_preparation_failure_after_refit_preserves_state_and_recovers(self):
        atomic=atomic_tests.GraphAtomicityTests()
        for backend in self.backends:
            for method in ('build','set'):
                with self.subTest(backend=backend,method=method):
                    m=self.model(backend);self.install(m,method,'larger')
                    labels=np.array([0,-1,-1,-1,-1,1]);self.fit(m,labels)
                    operation=(lambda:m.build_graph(np.ones((2,2)),3)) if method=='build' else (
                        lambda:m.set_graph(np.zeros((2,1),dtype=np.int64),np.ones(2,dtype=np.int64),k_nn='bad'))
                    atomic.reject(m,operation)
                    a=self.fit(m,labels);fresh=self.model(backend,'empty');self.install(fresh,method,'larger')
                    b=self.fit(fresh,labels)
                    for x,y in zip(a[0],b[0]):np.testing.assert_array_equal(x,y)
                    self.assertEqual(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(a[3],b[3])

    def test_zero_node_set_and_numpy_synchronous_configuration(self):
        for backend in self.backends:
            m=self.model(backend)
            m.set_graph(np.empty((0,0),dtype=np.int64),np.empty(0,dtype=np.int64),k_nn=1)
            self.assert_cleared(m)
            labels=np.empty(0,dtype=np.int64)
            with self.assertWarns(UserWarning):self.assertIs(m.fit_predict(labels),labels)
        m=PCC(impl='numpy',update_mode='synchronous',n_jobs=1)
        m.set_graph(np.array([[1],[0]]),np.ones(2,dtype=np.int64))
        m.fit_predict([0,1],max_iter=1,early_stop=False)
        m.set_graph(np.array([[1],[0]]),np.ones(2,dtype=np.int64))
        self.assert_cleared(m);self.assertEqual(m.update_mode,'synchronous')
        np.testing.assert_array_equal(m.fit_predict([0,1],max_iter=1),[0,1])
