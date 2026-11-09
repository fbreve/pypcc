"""Exact inverse mapped-label return semantics under experimental vectorization."""
import unittest
from unittest.mock import patch
import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC


def neighbors_ring(n):
    ids = np.arange(n, dtype=np.int64)
    return np.stack(((ids-1) % n, (ids+1) % n), axis=1)


class InverseLabelMappingTests(unittest.TestCase):
    def checked_fit(self, nodes, classes, positions, original_labels=None,
                    internal_view=False):
        if original_labels is None:
            original_labels = np.arange(classes, dtype=np.int64)*271 - 1000000
        labels = np.full(nodes, -1, dtype=np.int64)
        labels[:classes] = original_labels
        proposed = np.asarray(positions, dtype=np.int64)
        self.assertEqual(proposed.shape, labels.shape)
        expected = np.full(nodes, -1, dtype=np.int64)
        for idx, value in enumerate(original_labels):
            expected[proposed == idx] = value
        m = PCC(impl="numpy")
        m.set_graph(neighbors_ring(nodes), np.full(nodes,2,dtype=np.int64),k_nn=2)
        originals = proposed.copy()
        observed = []
        def stub(self, seed=None):
            if internal_view:
                container=np.empty(nodes*2,dtype=np.int64)
                container[::2] = proposed
                self.node.label = container[::2]
                observed.append(container)
            else:
                self.node.label = proposed.copy()
            observed.append(self.node.label)
        with patch.object(PCC, "_ParticleCompetitionAndCooperation__labelPropagation",stub):
            outcome = m.fit_predict(labels,max_iter=1,early_stop=False,seed=38059)
        np.testing.assert_array_equal(outcome, expected)
        np.testing.assert_array_equal(m.node.label, originals)
        np.testing.assert_array_equal(m.unique_labels, np.sort(original_labels))
        self.assertEqual(m.idx_to_label,
                         {idx:label for idx,label in enumerate(np.sort(original_labels))})
        self.assertEqual(outcome.dtype,np.int64)
        self.assertEqual(outcome.shape,(nodes,))
        self.assertTrue(outcome.flags.c_contiguous)
        self.assertTrue(outcome.flags.writeable)
        self.assertFalse(np.shares_memory(outcome,m.node.label))
        self.assertFalse(np.shares_memory(outcome,labels))
        return m, outcome

    def test_original_guard_and_vectorized_boundary(self):
        for n,c in ((150,3),(569,2),(1023,10),(1024,7),(1024,8),
                    (1024,10),(1797,10),(2048,50),(30000,10)):
            with self.subTest(nodes=n,classes=c):
                indices=np.arange(n,dtype=np.int64) % c
                self.checked_fit(n,c,indices)

    def test_invalid_internal_indices_and_int64_extremes(self):
        low=np.iinfo(np.int64).min
        high=np.iinfo(np.int64).max
        originals=np.concatenate((np.array([low,-909,0,low+1,high],dtype=np.int64),
                                  np.arange(3,dtype=np.int64)+121))
        originals=np.sort(originals)
        for n,c in ((150,8),(1023,8),(1024,8),(2048,8)):
            with self.subTest(n=n):
                indices=np.arange(n,dtype=np.int64)%c
                indices[:10]=[0,1,2,7,-1,-2,c,c+1,low,high]
                self.checked_fit(n,c,indices,originals,internal_view=True)

    def test_no_fit_zero_or_one_class_returns_original_reference(self):
        for n in (150,1024):
            model=PCC(impl="numpy")
            model.set_graph(neighbors_ring(n),np.full(n,2,dtype=np.int64),k_nn=2)
            for classes in (0,1):
                labels=np.full(n,-1,dtype=np.int64)
                if classes:labels[0]=28
                with self.subTest(n=n,classes=classes):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        actual=model.fit_predict(labels,seed=38059,max_iter=0)
                    self.assertIs(actual,labels)
                    self.assertNotIn("mapped_labels",model.__dict__)

    def test_real_native_backend_outputs_unchanged_on_large_class_case(self):
        n=1024
        originals=np.arange(10,dtype=np.int64)*89-10000
        labels=np.full(n,-1,dtype=np.int64)
        for c in range(10):
            labels[3*c:3*c+3]=originals[c]
        expected_input=labels.copy()
        for backend in ("numpy","numba","cython"):
            with self.subTest(backend=backend):
                m=PCC(impl=backend)
                m.set_graph(neighbors_ring(n),np.full(n,2,dtype=np.int64),k_nn=2)
                prediction=m.fit_predict(labels,max_iter=11,early_stop=False,seed=38059)
                self.assertEqual(prediction.shape,(n,))
                self.assertEqual(prediction.dtype,np.int64)
                self.assertTrue(prediction.flags.c_contiguous)
                self.assertFalse(np.shares_memory(prediction,m.node.label))
                np.testing.assert_array_equal(labels,expected_input)
                np.testing.assert_array_equal(prediction[:30],labels[:30])
                expected=np.full(n,-1,dtype=np.int64)
                for idx, value in m.idx_to_label.items():
                    expected[m.node.label==idx]=value
                np.testing.assert_array_equal(prediction,expected)


if __name__=="__main__":
    unittest.main()
