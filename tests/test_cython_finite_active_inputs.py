"""Cython probability inputs must be finite before native mutation.

This exercises BOTH direct Cython entry points, including the recently merged
default-on potential cache, and deliberately leaves unused buffer padding.
"""
import unittest

import numpy as np

try:
    import pcc_step
except ImportError:
    pcc_step = None


def sample():
    # Four-node reciprocal undirected ring with two labeled homes.
    graph = np.array([[1, 2], [0, 3], [0, 3], [1, 2]], dtype=np.int64)
    degree = np.full(4, 2, dtype=np.int64)
    labels = np.array([0, 1, -1, -1], dtype=np.int64)
    strength = np.ones(2, dtype=np.float64)
    dom = np.asfortranarray([
        [1., 0.], [0., 1.], [.5, .5], [.5, .5],
    ])
    arrays = dict(
        graph=graph, degree=degree, labels=labels,
        zerovec=np.zeros(2, dtype=np.float64),
        position=np.array([0, 1], dtype=np.int64),
        particle_class=np.array([0, 1], dtype=np.int64),
        strength=strength,
        distance=np.zeros((4, 2), dtype=np.uint8, order="F"),
        dominance=dom, ownership=np.zeros((4, 2), dtype=np.float64, order="F"),
        dom_row=np.zeros(2, dtype=np.float64),
        reduc=np.zeros(2, dtype=np.float64),
        dom_list=np.zeros(2, dtype=np.float64),
        dist_list=np.zeros(2, dtype=np.float64),
        prob=np.zeros(2, dtype=np.float64),
        slices=np.zeros(2, dtype=np.float64),
        weights=1.0 / (np.arange(258, dtype=np.float64) + 1.0) ** 2.0,
    )
    return arrays


def invoke(kind, arrays):
    x=arrays
    mandatory=(
        x["graph"],x["degree"],x["labels"],0.5,0.1,2,x["zerovec"],
        x["position"],x["particle_class"],x["strength"],x["distance"],
        x["dominance"],x["ownership"],
    )
    if kind=="step":
        return pcc_step.pcc_step(
            *mandatory,1.0,2.0,
            x["prob"],x["slices"],x["weights"])
    return pcc_step.pcc_propagate(
        *mandatory,1.0,2.0,
        x["dom_row"],x["reduc"],x["dom_list"],x["dist_list"],
        x["prob"],x["slices"],x["weights"],
        15,1,2000,100,return_info=True,seed=101)


@unittest.skipIf(pcc_step is None, "Cython extension not built")
class TestCythonFiniteActiveProbabilityInputs(unittest.TestCase):
    def test_nan_and_infinity_are_rejected_before_mutation(self):
        positions=(
            ("part_strength", "strength", (0,)),
            ("dominance", "dominance", (2,0)),
            ("dist_weights", "weights", (0,)),
            ("dist_weights", "weights", (255,)),
        )
        for kind in ("step", "propagate"):
            for field, buffer, position in positions:
                for invalid in (np.nan, np.inf, -np.inf):
                    with self.subTest(
                        kind=kind, field=field,
                        position=position, invalid=invalid
                    ):
                        x=sample()
                        x[buffer][position]=invalid
                        before={
                            name:np.asarray(value).copy()
                            for name,value in x.items()
                            if isinstance(value,np.ndarray)
                        }
                        with self.assertRaisesRegex(ValueError,field):
                            invoke(kind,x)
                        for name,original in before.items():
                            np.testing.assert_array_equal(
                                x[name],original,
                                err_msg=f"Partial mutation in {kind}: {name}")

    def test_unused_padding_may_contain_nan_or_infinity(self):
        for kind in ("step","propagate"):
            with self.subTest(kind=kind):
                x=sample()
                x["strength"]=np.r_[x["strength"],np.nan]
                dom=np.full((5,3),np.nan,dtype=np.float64,order="F")
                dom[:4,:2]=x["dominance"]
                x["dominance"]=dom
                x["weights"][256]=np.inf
                x["weights"][257]=np.nan
                # Only first 4 rows/2 classes, 2 particles and 256
                # distance-weight entries are read by this kernel.
                invoke(kind,x)
                self.assertTrue(np.all(
                    np.isfinite(x["dominance"][:4,:2])))
                self.assertTrue(np.isnan(x["strength"][-1]))
                self.assertTrue(np.isnan(x["dominance"][-1,0]))

    def test_uninitialized_scratch_is_not_rejected_as_active_state(self):
        for kind in ("step","propagate"):
            with self.subTest(kind=kind):
                x=sample()
                for key in (
                    "dom_row","reduc","dom_list","dist_list","prob","slices"
                ):
                    x[key][:]=np.nan
                # Preflight must not reject buffers written during the step.
                invoke(kind,x)

    def test_no_stop_still_rejects_active_nan(self):
        x=sample()
        x["strength"][0]=np.nan
        args=(
            x["graph"],x["degree"],x["labels"],0.5,0.1,2,
            x["zerovec"],x["position"],x["particle_class"],x["strength"],
            x["distance"],x["dominance"],x["ownership"],1.0,2.0,
            x["dom_row"],x["reduc"],x["dom_list"],x["dist_list"],
            x["prob"],x["slices"],x["weights"],15,0,2000,100,
        )
        with self.assertRaisesRegex(ValueError,"part_strength"):
            pcc_step.pcc_propagate(*args,return_info=True,seed=101)


if __name__=="__main__":
    unittest.main()
