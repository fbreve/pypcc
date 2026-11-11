"""Permanent native Cython regression: canonical initial row-max cache shortcut.

Separate source-exact Linux/Windows whole-fit CPU GO evidence is in science #241.
No change to graph, particle dynamics, native RNG, stopping, or direct low-level
Cython calls with arbitrary already-populated dominance matrices.
"""
import unittest
from unittest.mock import patch

import numpy as np

import pcc
from pcc import ParticleCompetitionAndCooperation as PCC

try:
    import pcc_step
except ImportError:
    pcc_step = None


class CustomPCC(PCC):
    """A subclass must retain the original general-purpose Cython path."""


def make_graph(n):
    row = np.arange(n, dtype=np.int64)[:, None]
    return np.ascontiguousarray((row + np.array([1, -1, 2, -2])) % n), \
        np.full(n, 4, dtype=np.int64)


def make_labels(n, classes):
    labels = np.full(n, -1, dtype=np.int64)
    for cls in range(classes):
        labels[cls * 3: cls * 3 + 3] = cls
    return labels


def bitwise_state(model, prediction):
    arrays = {
        "predictions": prediction,
        "node_labels": model.node.label,
        "dominance": model.node.dominance,
        "part_homenode": model.part.homenode,
        "part_curnode": model.part.curnode,
        "part_label": model.part.label,
        "part_strength": model.part.strength,
        "dist_table": model.part.dist_table,
        "owndeg": model.owndeg,
        "mapped_labels": model.mapped_labels,
    }
    return {
        name: (arr.shape, arr.dtype.str, np.ascontiguousarray(arr).tobytes())
        for name, arr in arrays.items()
    }


@unittest.skipUnless(pcc._HAS_CYTHON and pcc_step is not None,
                     "compiled Cython extension unavailable")
class CanonicalCythonRowmaxTests(unittest.TestCase):
    def make_model(self, cls=PCC, n=80, classes=10):
        graph, degrees = make_graph(n)
        model = cls(impl="cython", update_mode="sequential")
        model.set_graph(graph, degrees, k_nn=4)
        self.assertIs(model._get_backend_fn(), pcc_step.pcc_propagate)
        return model, make_labels(n, classes)

    def fit_and_compare(self, classes, max_iter, early_stop):
        # Intercept only to record real low-level stop metadata and verify the
        # actual Cython dispatcher receives the canonical hint, NEVER a
        # requested-backend string/fallback or subclass path.
        true_cython = pcc_step.pcc_propagate
        records = []

        def observe(*args, **kwargs):
            enabled = kwargs.get("experimental_canonical_rowmax_init", False)
            kwargs["return_info"] = True
            outcome = true_cython(*args, **kwargs)
            records.append((enabled, outcome))
            return outcome

        with patch.object(pcc_step, "pcc_propagate", observe):
            direct, labels = self.make_model(PCC, classes=classes)
            inherited, labels2 = self.make_model(CustomPCC, classes=classes)
            np.testing.assert_array_equal(labels, labels2)
            kwargs = dict(
                p_grd=.5, delta_v=.1, deltap=1., dexp=2.,
                max_iter=max_iter, early_stop=early_stop, es_chk=2000, seed=29
            )
            result_a = direct.fit_predict(labels, **kwargs)
            result_b = inherited.fit_predict(labels, **kwargs)

        self.assertEqual(len(records), 2)
        self.assertIs(records[0][0], bool(early_stop))
        self.assertIs(records[1][0], False)
        self.assertEqual(records[0][1], records[1][1], "native stopping changed")
        self.assertEqual(records[0][1]["effective_iterations"],
                         max_iter if not records[0][1]["stop_reason"] == "early_stop"
                         else records[0][1]["effective_iterations"])
        self.assertEqual(bitwise_state(direct, result_a),
                         bitwise_state(inherited, result_b))
        self.assertTrue(direct.node.dominance.flags.f_contiguous)
        self.assertTrue(inherited.node.dominance.flags.f_contiguous)
        self.assertEqual(direct.neib_list.dtype, np.dtype("int64"))
        # Both models own independent state; no hidden cross-model aliasing.
        self.assertFalse(np.shares_memory(direct.node.dominance,
                                          inherited.node.dominance))

    def test_exact_cython_base_activates_only_on_real_cached_path(self):
        for classes in (2, 10):
            for steps in (0, 1, 5, 21):
                with self.subTest(classes=classes, steps=steps):
                    self.fit_and_compare(classes, steps, True)

    def test_fixed_iterations_leave_lowlevel_native_kernel_unchanged(self):
        for classes in (2, 10):
            with self.subTest(classes=classes):
                self.fit_and_compare(classes, 5, False)

    def test_canonical_row_max_follows_initial_ieee_values(self):
        for classes in (2, 3, 10, 50):
            n = 3 * classes + 17
            labels = make_labels(n, classes)
            original = np.full((n, classes), 1.0 / classes,
                               dtype=np.float64, order="F")
            known = np.flatnonzero(labels != -1)
            original[known, :] = 0.0
            original[known, labels[known]] = 1.0
            expected = np.where(labels == -1, 1.0 / classes, 1.0)
            self.assertEqual(np.max(original, axis=1).tobytes(),
                             expected.tobytes())

    def test_direct_lowlevel_arbitrary_dominance_uses_legacy_max_scan(self):
        model, labels = self.make_model(classes=10)
        model.fit_predict(labels, max_iter=5, early_stop=True, seed=29)
        n, classes = model.node.dominance.shape
        p = model.part
        native = pcc_step.pcc_propagate

        def direct(initial_dominance, hint):
            # Cloned input arrays avoid any alias or cross-run contamination.
            curr = p.curnode.copy()
            strengths = p.strength.copy()
            distances = p.dist_table.copy(order="F")
            dom = initial_dominance.copy(order="F")
            owndeg = model.owndeg.copy(order="F")
            kw = dict(return_info=True, seed=29,
                      experimental_rowmax_cache=True,
                      experimental_verify_rowmax_cache=True)
            if hint is not None:
                kw["experimental_canonical_rowmax_init"] = hint
            info = native(
                model.neib_list, model.neib_qt, model.mapped_labels,
                .5, .1, classes, model.zerovec,
                curr, p.label.copy(), strengths, distances, dom, owndeg,
                1., 2., model.buf_dom_row.copy(), model.buf_reduc.copy(),
                model.buf_dom_list.copy(), model.buf_dist_list.copy(),
                model.buf_prob.copy(), model.buf_slices.copy(),
                model.dist_weights.copy(), 21, True, 2000, 1, **kw
            )
            state = (curr, strengths, distances, dom, owndeg)
            values = tuple((a.shape, a.dtype.str,
                            np.ascontiguousarray(a).tobytes()) for a in state)
            return info, values

        arbitrary = model.node.dominance.copy(order="F")
        arbitrary[0, :] = np.linspace(.01, .25, classes)
        # A low-level caller can supply any legal finite dominance state.
        # Omitting the hint MUST run and verify the full legacy cache scan.
        self.assertEqual(direct(arbitrary, None), direct(arbitrary, False))
        canonical = np.full((n, classes), 1.0 / classes, order="F")
        mapped = model.mapped_labels
        known = np.flatnonzero(mapped != -1)
        canonical[known, :] = 0.0
        canonical[known, mapped[known]] = 1.0
        # The explicit opt-in must equal a full row-max scan even under
        # per-checkpoint native legacy-vs-cached verification.
        self.assertEqual(direct(canonical, False), direct(canonical, True))

    def test_numpy_backend_is_not_given_cython_native_keyword(self):
        graph, degrees = make_graph(80)
        labels = make_labels(80, 10)
        model = PCC(impl="numpy", update_mode="sequential")
        model.set_graph(graph, degrees, k_nn=4)
        self.assertIsNot(model._get_backend_fn(), pcc_step.pcc_propagate)
        pred = model.fit_predict(labels, max_iter=0, early_stop=True, seed=29)
        self.assertEqual(pred.shape, labels.shape)
        np.testing.assert_array_equal(pred[labels != -1], labels[labels != -1])


if __name__ == "__main__":
    unittest.main()
