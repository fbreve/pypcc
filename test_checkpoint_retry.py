import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments.checkpoint_retry import retry_writer
import experiments.stopping_patience as experiment
from test_stopping_patience import fixture
from test_propagation_metrics import seed_only


def locked():
    error = PermissionError('Simulated Windows sharing/access failure')
    error.winerror = 5
    return error


class CheckpointRetryTests(unittest.TestCase):
    def test_retry_preserves_trajectory_rng_and_csv(self):
        initial, case = fixture()
        original = experiment.atomic_pickle
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory)/'plain'
            result, expected, _ = experiment.run(initial, case, 100, plain, 'same', seed_only,
                                                spacing=10, recovery_spacing=20)
            expected_rng = experiment.rng_state()
            calls = 0

            def failing_write(path, state):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise locked()
                return original(path, state)

            # The original operation fails on the reproduced error.
            with self.assertRaises(PermissionError):
                failing_write(None, None)
            calls = 0
            with patch.object(experiment, 'atomic_pickle', retry_writer(failing_write, sleeper=lambda _: None)):
                retried, actual, _ = experiment.run(initial, case, 100, Path(directory)/'retry', 'same', seed_only,
                                                   spacing=10, recovery_spacing=20)
            self.assertEqual(result['rows'], retried['rows'])
            self.assertEqual(expected_rng, experiment.rng_state())
            for i in experiment.STATE_INDICES:
                np.testing.assert_array_equal(expected[i], actual[i])
            for name in ('checks.csv', 'samples.csv'):
                self.assertEqual((plain/name).read_bytes(), (Path(directory)/'retry'/name).read_bytes())

    def test_permanent_error_is_bounded_other_errors_not_retried(self):
        attempts = []

        def fail(*args):
            attempts.append(1)
            raise locked()
        with self.assertRaises(PermissionError):
            retry_writer(fail, attempts=3, sleeper=lambda _: None)(None, None)
        self.assertEqual(len(attempts), 3)
        with self.assertRaises(ValueError):
            retry_writer(lambda *_: None, attempts=0)
        with self.assertRaises(PermissionError):
            retry_writer(lambda *_: (_ for _ in ()).throw(PermissionError('other')))(None, None)


if __name__ == '__main__':
    unittest.main()
