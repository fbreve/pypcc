"""Original PCC numeric behavior with in-place postpropagation owndeg normalization."""
import unittest

import numpy as np

from pcc import ParticleCompetitionAndCooperation as PCC


class InplaceOwndegNormalizationTests(unittest.TestCase):
    def test_real_model_inplace_owndeg_preserves_normalization(self):
        # Use a stub solely at the normal backend boundary to control the
        # owndeg matrix; run the actual original model initialization and
        # postprocessing, not a duplicated postprocessing implementation.
        model=PCC(impl="numpy")
        neighbors=np.array([[1,2],[0,2],[0,1],[2,-1]],
                           dtype=np.int64)
        degrees=np.array([2,2,2,1],dtype=np.int64)
        model.set_graph(neighbors,degrees,k_nn=2)
        raw=np.array([[0.0,0.0],[1.0,2.0],[3.0,1.0],
                      [0.0,10.0]],dtype=np.float64,order="F")
        captured={}
        def stub(*args,**kwargs):
            own=args[12]
            self.assertEqual(own.shape,raw.shape)
            self.assertTrue(own.flags.f_contiguous)
            own[:]=raw
            captured["backend_owndeg"]=own
        model._get_backend_fn=lambda:stub
        labels=np.array([0,1,-1,-1],dtype=np.int64)
        predicted=model.fit_predict(labels,max_iter=2,early_stop=False,seed=7)
        sums=raw.sum(axis=1,keepdims=True)
        sums[sums==0.0]=1.0
        expected=raw/sums
        self.assertIs(model.owndeg,captured["backend_owndeg"],
                      "Normalization should reuse the model's original buffer")
        np.testing.assert_array_equal(model.owndeg,expected)
        self.assertEqual(model.owndeg.dtype,np.dtype("float64"))
        self.assertTrue(model.owndeg.flags.f_contiguous)
        self.assertEqual(predicted.shape,(4,))
        np.testing.assert_array_equal(predicted[:2],labels[:2])

    def test_inplace_division_is_bitwise_equal_for_zero_rows_and_layouts(self):
        rng=np.random.default_rng(31991)
        for n,c in ((0,2),(1,2),(178,3),(569,2),(1797,10)):
            original=rng.random((n,c))
            if n:
                original[::7,:]=0.0
            for kind,data in (("C",np.ascontiguousarray(original)),
                              ("F",np.asfortranarray(original))):
                with self.subTest(rows=n,classes=c,layout=kind):
                    ref=data.copy(order="K")
                    new=data.copy(order="K")
                    denom=np.sum(ref,axis=1,keepdims=True)
                    denom[denom==0.0]=1.0
                    ref=ref/denom
                    denom_new=np.sum(new,axis=1,keepdims=True)
                    denom_new[denom_new==0.0]=1.0
                    saved=new
                    np.divide(new,denom_new,out=new)
                    self.assertIs(new,saved)
                    self.assertEqual(new.dtype,ref.dtype)
                    self.assertEqual(new.shape,ref.shape)
                    np.testing.assert_array_equal(ref,new)


if __name__=="__main__":
    unittest.main()
