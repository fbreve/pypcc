"""Small deterministic checks; long study/downloads never run in CI."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pcc_numba
from experiments.stopping_patience import (Observer, adjusted_threshold, original_threshold,
    run, rng_state, restore_rng, accuracy, STATE_INDICES)
from test_sequential_safety import state, buffers, next_random
from test_propagation_metrics import seed_only
from experiments.patience_summary import summaries, FIELDS


def fixture(plateau=False):
    a = state()
    a[3] = .5
    if plateau:
        a[0][:] = 0
        a[2][0] = 0
        a[11][0] = [1., 0.]
        a[3] = 0.
    initial = a + buffers(a) + [100, True, 2000, 1]
    case = dict(dataset='fixture', seed=42, labels=a[2].copy(), truth=np.array([0, 1, 0]))
    return initial, case


class PatienceTests(unittest.TestCase):
    def test_formula_and_rounding(self):
        self.assertEqual(original_threshold(178, 9, 10), 396)
        self.assertEqual(original_threshold(1797, 30, 10), 1198)
        self.assertEqual(original_threshold(150, 9, 10), 333)
        self.assertEqual(original_threshold(569, 6, 10), 1897)
        self.assertEqual(original_threshold(3, 1, 0), 1)
        self.assertEqual(original_threshold(3, 1, 10, isolated=True), 1)
        self.assertEqual(adjusted_threshold(1, .25), 1)
        self.assertEqual(adjusted_threshold(10, .25), 2)  # 2.5, ties to even
        self.assertEqual(adjusted_threshold(14, .25), 4)  # 3.5, ties to even
        self.assertEqual(adjusted_threshold(333, 1.), 333)

    def test_strict_record_reset_first_stop_and_invalid_schedule(self):
        observer = Observer(1)
        fired, _, record = observer.check(1, .6)
        self.assertTrue(record)
        self.assertEqual(fired, [])
        observer.check(11, .6)
        self.assertEqual(observer.counter, 1)
        fired, _, _ = observer.check(21, .59)
        self.assertIn(1., fired)
        self.assertEqual(observer.events[1.]['iteration'], 21)
        observer.check(31, np.nextafter(.6, 1.))
        self.assertEqual(observer.counter, 0)
        self.assertEqual(observer.resets, 2)
        observer.check(41, .6)
        observer.check(51, .6)
        self.assertEqual(observer.events[1.]['iteration'], 21)
        with self.assertRaises(ValueError):
            observer.check(52, .6)

    def test_rng_round_trip_has_no_draw(self):
        seed_only(9)
        before = rng_state()
        value = next_random()
        restore_rng(before)
        self.assertEqual(next_random(), value)

    def test_accuracy_counts_ignore_initially_labeled_nodes(self):
        initial, case = fixture(True)
        correct, acc = accuracy(np.array([1, 1, 1]), case)
        self.assertEqual(correct, 1)
        self.assertEqual(acc, .5)

    def test_exact_original_event_state_and_rng(self):
        initial, case = fixture(True)
        original = copy.deepcopy(initial)
        seed_only(case['seed'])
        metrics = pcc_numba.pcc_propagate_numba(*original, return_info=True)
        original_rng = rng_state()
        self.assertEqual(metrics['effective_iterations'], 21)
        with tempfile.TemporaryDirectory() as root:
            result, a, verify = run(initial, case, 100, root, 'original', seed_only,
                                   spacing=10, recovery_spacing=50, verify_points=(21, 100))
            event = next(row for row in result['rows'] if row['multiplier'] == 1.)
            self.assertEqual(event['stop_iteration'], 21)
            self.assertEqual(event['checks'], 3)
            self.assertEqual(event['accuracy_unlabeled'], .5)
            for row in result['rows']:
                self.assertEqual(row['stop_iteration'], 1 + 10 * (row['adjusted_threshold'] + 1))
            self.assertEqual(result['iterations'], 100)  # continue after every predicate fires
            self.assertEqual(result['checks'], 10)
            self.assertTrue(all(v['state_equal'] and v['rng_equal'] for v in verify))
            # Check saved trajectory's exact first event against the early-stop kernel.
            replay = copy.deepcopy(initial)
            seed_only(case['seed'])
            pcc_numba._pcc_propagate_numba_sequential(*replay[:22], 21, False, *replay[24:], False)
            for i in STATE_INDICES:
                np.testing.assert_array_equal(original[i], replay[i])
            self.assertEqual(rng_state(), original_rng)

    def test_censoring_and_continuous_state_rng_preservation(self):
        initial, case = fixture()
        with tempfile.TemporaryDirectory() as root:
            result, observed, verified = run(initial, case, 12, root, 'censor', seed_only,
                         spacing=5, recovery_spacing=7, references=(7,), verify_points=(1, 7, 12))
            final_rng = rng_state()
            self.assertEqual(result['checks'], 2)  # 1 and 11, not 12
            self.assertTrue(all(row['status'] == 'right_censored' for row in result['rows']))
            self.assertTrue(all(row['stop_iteration'] is None for row in result['rows']))
            base = copy.deepcopy(initial)
            seed_only(42)
            pcc_numba._pcc_propagate_numba_sequential(*base[:22], 12, False, *base[24:], False)
            self.assertEqual(final_rng, rng_state())
            for i in STATE_INDICES:
                np.testing.assert_array_equal(observed[i], base[i])
            self.assertEqual(len(verified), 3)

    def test_recovery_and_completed_case_skip_are_exact(self):
        initial, case = fixture()
        with tempfile.TemporaryDirectory() as root:
            uninterrupted, expected, _ = run(initial, case, 100, Path(root)/'base', 'same', seed_only,
                        spacing=10, recovery_spacing=20, verify_points=(21, 100))
            expected_rng = rng_state()
            recovery = Path(root)/'resume'
            with self.assertRaises(InterruptedError):
                run(initial, case, 100, recovery, 'same', seed_only,
                    spacing=10, recovery_spacing=20, interrupt_at=40, verify_points=(21, 100))
            # Trailing partial work after checkpoint must not be counted twice.
            with (recovery/'checks.csv').open('a') as stream:
                stream.write('partial,row\n')
            seed_only(999)
            result, observed, verified = run(initial, case, 100, recovery, 'same', seed_only,
                        spacing=10, recovery_spacing=20, verify_points=(21, 100))
            self.assertTrue({21, 100}.issubset(v['iteration'] for v in verified))
            self.assertTrue(any(v['original_stop_matches'] for v in verified))
            self.assertEqual(rng_state(), expected_rng)
            self.assertEqual(result['rows'], uninterrupted['rows'])
            for i in STATE_INDICES:
                np.testing.assert_array_equal(expected[i], observed[i])
            self.assertEqual((recovery/'checks.csv').read_bytes(), (Path(root)/'base'/'checks.csv').read_bytes())
            self.assertEqual((recovery/'samples.csv').read_bytes(), (Path(root)/'base'/'samples.csv').read_bytes())
            before_rng = rng_state()
            skipped, a, _ = run(initial, case, 100, recovery, 'same', seed_only)
            self.assertIsNone(a)
            self.assertEqual(skipped['rows'], result['rows'])
            self.assertEqual(before_rng, rng_state())
            with self.assertRaises(ValueError):
                run(initial, case, 100, recovery, 'different', seed_only)
            with self.assertRaises(ValueError):
                run(initial, case, 101, recovery, 'same', seed_only)

    def test_equal_dataset_weight_not_pooled_weight(self):
        rows = []
        for name, count, value in [('A', 1, 0.), ('B', 3, 1.)]:
            for seed in range(count):
                row = dict(dataset=name, seed=seed, multiplier=1., status='stopped',
                           classes=2, nodes=100, labeled_fraction=.06)
                row.update({field: value for field in FIELDS})
                rows.append(row)
        datasets, overall = summaries(rows)
        self.assertEqual(overall[0]['accuracy_difference_dataset_mean_mean'], .5)
        self.assertEqual(overall[0]['stop_rate'], 1.)
        self.assertEqual(len(datasets), 2)

    def test_resume_after_final_checkpoint_before_completion_marker(self):
        initial, case = fixture()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(InterruptedError):
                run(initial, case, 12, root, 'final-checkpoint', seed_only, interrupt_at=12)
            expected_rng = rng_state()
            result, _, _ = run(initial, case, 12, root, 'final-checkpoint', seed_only)
            self.assertEqual(result['iterations'], 12)
            self.assertEqual(rng_state(), expected_rng)

    def test_arff_delimiter_spaces_not_dataset_subsampling(self):
        import io
        from scipy.io import arff
        from experiments.patience_data import arff_text
        raw = b'@relation example\n@attribute x numeric\n@attribute class {0,1}\n@data\n1, 0\n2, 1\n'
        data, _ = arff.loadarff(io.StringIO(arff_text(raw)))
        np.testing.assert_array_equal(data['x'], [1., 2.])
        np.testing.assert_array_equal(data['class'], [b'0', b'1'])


if __name__ == '__main__':
    unittest.main()
