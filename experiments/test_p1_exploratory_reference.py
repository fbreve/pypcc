"""Only synthetic examples; the P1 scientific design is NOT yet approved."""
import copy
import hashlib
import math
from pathlib import Path
import unittest

from experiments import p1_exploratory_reference as ref


def sha(text):
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def synthetic_table():
    return [
        {
            "dataset_id": ds, "replicate": rep, "backend": be,
            "graph_sha256": sha(f"synthetic-graph|{ds}"),
            "mask_sha256": sha(f"synthetic-mask|{ds}|{rep}"),
            "flip_fraction": rep/16.0 + list(ref.BACKENDS).index(be)/100.0,
        }
        for ds in ref.IDS
        for rep in range(ref.N_REPLICATES)
        for be in ref.BACKENDS
    ]


class P1ExploratorySyntheticTests(unittest.TestCase):
    def test_proposed_masks_do_not_sample_any_labels(self):
        self.assertEqual(ref.proposed_mask_seed_words(0,0), [20261112,0,0])
        self.assertEqual(ref.proposed_mask_seed_words(2,7), [20261112,2,7])
        for args in [(-1,0),(3,0),(0,8),(0,-1)]:
            with self.assertRaises(ref.P1ScientificInvalid):
                ref.proposed_mask_seed_words(*args)

    def test_backend_propagation_seeds_are_unique_nonzero_uint32(self):
        seeds = [
            ref.proposed_propagation_seed(ds, rep, be)
            for ds in ref.IDS for rep in range(8) for be in ref.BACKENDS
        ]
        self.assertEqual(len(seeds), 72)
        self.assertEqual(len(set(seeds)), 72)
        self.assertTrue(all(1 <= x <= 2**32-1 for x in seeds))
        self.assertEqual(seeds, [
            ref.proposed_propagation_seed(ds, rep, be)
            for ds in ref.IDS for rep in range(8) for be in ref.BACKENDS
        ])
        with self.assertRaises(ref.P1ScientificInvalid):
            ref.proposed_propagation_seed(545,0,ref.BACKENDS[0])
        with self.assertRaises(ref.P1ScientificInvalid):
            ref.proposed_propagation_seed(212,8,ref.BACKENDS[0])
        with self.assertRaises(ref.P1ScientificInvalid):
            ref.proposed_propagation_seed(212,0,"unknown")

    def test_primary_synthetic_flip_excludes_labeled_nodes_and_tie_rule(self):
        a = [[0.8,0.2], [0.5,0.5], [0.9,0.1], [0.2,0.8]]
        b = [[0.2,0.8], [0.1,0.9], [0.3,0.7], [0.3,0.7]]
        observed = ref.synthetic_primary_flips(a,b,[0])
        self.assertEqual(observed["unlabeled_n"], 3)
        self.assertEqual(observed["flip_n"], 2)
        self.assertEqual(observed["flip_fraction"], 2/3)
        self.assertEqual(observed["exact_ties_5000_n"], 1)
        self.assertEqual(observed["exact_ties_50000_n"], 0)
        self.assertEqual(observed["model_fits_executed"], 0)
        self.assertEqual(observed["window"], [5000,50000])

    def test_primary_rejects_inconsistent_or_nonfinite_dominance(self):
        a = [[0.5,0.5],[0.1,0.9],[0.3,0.7]]
        for labels in [[0,0],[-1],[3],[0,1,2],["0"]]:
            with self.assertRaises(ref.P1ScientificInvalid):
                ref.synthetic_primary_flips(a,a,labels)
        invalid = [
            [[0,0],[0.1,0.9],[0.3,0.7]],
            [[0.5,0.5],[math.nan,0.9],[0.3,0.7]],
            [[0.5,0.5],[0.1,-0.9],[0.3,0.7]],
            [[0.5,0.5],[0.1,0.9,0],[0.3,0.7]],
        ]
        for bad in invalid:
            with self.assertRaises(ref.P1ScientificInvalid):
                ref.synthetic_primary_flips(a,bad,[0])

    def test_complete_case_summary_has_only_descriptive_statistics(self):
        summary=ref.synthetic_72_case_descriptive_summary(synthetic_table())
        self.assertEqual(summary["n_proposed_cases"],72)
        self.assertFalse(summary["CI_or_equivalence_claim"])
        self.assertFalse(summary["independent_node_replicates_claimed"])
        self.assertFalse(summary["execution_authorized"])
        self.assertEqual(summary["datasets_selected"],0)
        self.assertEqual(summary["real_PCC_fits_executed"],0)
        self.assertIsNone(summary["P2_execution_sha256"])
        for be in ref.BACKENDS:
            self.assertEqual(
                len(summary["backends"][be]["dataset_summaries"]),3)
            self.assertAlmostEqual(
                summary["backends"][be]["unweighted_dataset_mean"],
                3.5/16+list(ref.BACKENDS).index(be)/100,
            )

    def test_missing_duplicate_or_unpaired_cases_block_full_summary(self):
        good = synthetic_table()
        with self.assertRaises(ref.P1ScientificInvalid):
            ref.synthetic_72_case_descriptive_summary(good[:-1])
        broken = copy.deepcopy(good)
        broken[0] = copy.deepcopy(broken[1])
        with self.assertRaises(ref.P1ScientificInvalid):
            ref.synthetic_72_case_descriptive_summary(broken)
        for key, value in [
            ("flip_fraction",math.nan),
            ("flip_fraction",1.1),
            ("mask_sha256","a"*64),
            ("graph_sha256","b"*64),
            ("dataset_id",545),
            ("backend","auto"),
        ]:
            broken=copy.deepcopy(good)
            broken[0][key]=value
            with self.assertRaises(ref.P1ScientificInvalid):
                ref.synthetic_72_case_descriptive_summary(broken)

    def test_no_model_dataset_or_rng_imports(self):
        source=Path(ref.__file__).read_text(encoding="utf-8")
        for forbidden in [
            "import numpy","import sklearn","import pandas","import pcc",
            "import numba","fit_predict(","pcc_propagate(",
            "urlopen(", "requests.get(", "np.random", "Generator(",
        ]:
            self.assertNotIn(forbidden,source)
        self.assertIn('"real_PCC_fits_executed": 0',source)
        self.assertIn('"execution_authorized": False',source)


if __name__=="__main__":
    unittest.main()
