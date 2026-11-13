"""P2-07B synthetic metadata-contract/adversarial tests only, no benchmark fits."""
import copy
import unittest

from experiments import p2_07b_isolated_synthetic_resource_pilot as testee


def fake_record(backend):
    value = {
        "nodes": testee.NODES,
        "particles": testee.LABELED_PER_CLASS*2,
        "milestones": [0,1,10,50,100,250,500,testee.ITERATIONS],
    }
    if backend == "numpy":
        value.update(
            "iterations", testee.ITERATIONS,
        )
    return value


class SyntheticResourceBudgetTests(unittest.TestCase):
    def test_exact_separate_three_original_pins_and_iteration_budget(self):
        self.assertEqual(set(testee.BENCHMARKS), {"numpy", "numba", "cython"})
        self.assertEqual(testee.NODES, 500)
        self.assertEqual(testee.ITERATIONS, 1000)
        self.assertEqual(testee.LABELED_PER_CLASS, 3)
        self.assertEqual(testee.REPEATS, 2)
        for backend, entry in testee.BENCHMARKS.items():
            self.assertEqual(len(entry["git_blob"]), 40)
            self.assertEqual(len(entry["observer_git_blob"]), 40)
            self.assertEqual(len(entry["commit"]), 40)
            self.assertEqual(entry["file"].split("/")[-1],
                             "bench_live_"+backend+"_fullfit.py")

    def test_accept_original_numpy_paired_result_and_reject_invalid_clock(self):
        base = {
            "nodes": 500, "particles": 6, "repeats": 2,
            "iterations": 1000,
            "milestones": [0,1,10,50,100,250,500,1000],
            "snapshot_bytes_per_observed_fit": 11000,
            "samples_cpu_ns": {
                "baseline": [1000000, 1100000],
                "observed": [1200000, 1300000],
            },
            "median_baseline_cpu_s": .00105,
            "median_observed_cpu_s": .00125,
            "observed_over_baseline_cpu_ratio": 1.1904761904761905,
        }
        summary = testee.verify_original_result("numpy", base)
        self.assertEqual(summary["median_baseline_CPU_seconds_per_complete_synthetic_fit"],
                         .00105)
        broken = copy.deepcopy(base)
        broken["samples_cpu_ns"]["baseline"][1] = 0
        with self.assertRaises(testee.SyntheticBudgetBlocked):
            testee.verify_original_result("numpy", broken)
        broken = copy.deepcopy(base)
        broken["milestones"] = [0, 1, 10]
        with self.assertRaises(testee.SyntheticBudgetBlocked):
            testee.verify_original_result("numpy", broken)

    def test_accept_compiled_paired_samples_and_disallow_wrong_cython_mode(self):
        common = {
            "nodes": 500, "particles": 6,
            "fits_per_sample": testee.COMPILED_BATCH,
            "milestones": [0,1,10,50,100,250,500,1000],
            "snapshot_bytes_per_fit": 21000,
            "pairs": [
                {"baseline": {"cpu_ns": 300000},
                 "observed": {"cpu_ns": 320000}},
                {"baseline": {"cpu_ns": 310000},
                 "observed": {"cpu_ns": 340000}},
            ],
            "median_baseline_cpu_s_per_fit": .0000305,
            "median_observed_cpu_s_per_fit": .000033,
        }
        numba = {**common,
                 "fits_per_batch": common["fits_per_sample"],
                 "observed_over_baseline_ratio": 1.081967,
                 "nopython_observer_verified": True}
        numba.pop("fits_per_sample")
        cython = {**common,
                  "mode": "no-stop",
                  "observed_effective_iterations": 1000,
                  "overhead_percent_ratio_of_medians": 8.1967}
        self.assertGreater(
            testee.verify_original_result("numba", numba)[
                "median_observer_CPU_seconds_per_complete_synthetic_fit"], 0)
        self.assertGreater(
            testee.verify_original_result("cython", cython)[
                "median_observer_CPU_seconds_per_complete_synthetic_fit"], 0)
        broken = {**numba, "nopython_observer_verified": False}
        with self.assertRaises(testee.SyntheticBudgetBlocked):
            testee.verify_original_result("numba", broken)
        broken = {**cython, "mode": "early-cached"}
        with self.assertRaises(testee.SyntheticBudgetBlocked):
            testee.verify_original_result("cython", broken)

    def test_reject_wrong_backend_and_scope(self):
        sample = {
            "nodes": 500, "particles": 6,
            "repeats": 2, "iterations": 1000,
            "milestones": [0,1,10,50,100,250,500,1000],
        }
        with self.assertRaises(testee.SyntheticBudgetBlocked):
            testee.verify_original_result("unexpected", sample)
        with self.assertRaises(testee.SyntheticBudgetBlocked):
            testee.verify_original_result("numpy", {**sample, "nodes": 900})

    def test_source_name_hashes_are_distinct_and_bound_to_observer(self):
        paths = [entry["file"] for entry in testee.BENCHMARKS.values()]
        observers = [entry["observer"] for entry in testee.BENCHMARKS.values()]
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(len(set(observers)), 3)
        self.assertTrue(all(x.startswith("experiments/live_") for x in observers))


if __name__ == "__main__":
    unittest.main()
