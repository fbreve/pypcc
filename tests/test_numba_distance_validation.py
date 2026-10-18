"""R01: validate distance indices before unchecked Numba lookups, without copies."""
import copy
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numba import njit

import pcc_numba
from pcc_numpy import pcc_step_numpy
from pcc_metrics import propagation_info
from tests.test_sequential_safety import state, buffers, next_random


@njit
def seed_only(seed):
    np.random.seed(seed)


def invoke(args, propagation=False):
    if propagation:
        return pcc_numba.pcc_propagate_numba(
            *args, *buffers(args), 30, True, 2000, 1, return_info=True)
    return pcc_numba.pcc_step_numba(*args)


class NumbaDistanceValidationTests(unittest.TestCase):
    def test_other_backends_already_reject_original_crash_tables(self):
        import pcc_step
        for value in (-10**12, 10**12):
            for fn, error in ((pcc_step_numpy, IndexError), (pcc_step.pcc_step, ValueError)):
                args = state()
                args[10] = np.full((3, 1), value, dtype=np.int64)
                with self.subTest(backend=fn.__module__, value=value), self.assertRaises(error):
                    fn(*args)

    def test_incompatible_tables_rejected_before_native_dispatch(self):
        invalid = [None, np.empty(0, dtype=np.uint8),
                   np.empty((3, 1, 1), dtype=np.uint8),
                   np.empty((2, 1), dtype=np.uint8),
                   np.empty((3, 0), dtype=np.uint8),
                   np.zeros((3, 1), dtype=bool),
                   np.zeros((3, 1), dtype=np.float64),
                   np.full((3, 1), np.nan),
                   np.full((3, 1), np.inf),
                   np.zeros((3, 1), dtype=complex),
                   np.zeros((3, 1), dtype=object)]
        for dtype, values in ((np.int64, (-10**12, -1, 256, 10**12)),
                              (np.uint64, (256, 2**64 - 1))):
            invalid += [np.full((3, 1), value, dtype=dtype) for value in values]
        for propagation in (False, True):
            target = ('_pcc_propagate_numba_sequential' if propagation
                      else '_pcc_step_numba_sequential')
            for table in invalid:
                with self.subTest(propagation=propagation, table=str(table)):
                    args = state()
                    args[10] = table
                    before = [a.copy() for a in (args[7], args[9], args[11], args[12])]
                    seed_only(19)
                    with patch.object(pcc_numba, target) as native:
                        with self.assertRaises(ValueError):
                            invoke(args, propagation)
                        native.assert_not_called()
                    observed_random = next_random()
                    seed_only(19)
                    self.assertEqual(observed_random, next_random())
                    for old, actual in zip(before, (args[7], args[9], args[11], args[12])):
                        np.testing.assert_array_equal(old, actual)

    def test_valid_integer_layouts_preserve_state_rng_and_stop(self):
        for dtype in (np.uint8, np.int8, np.uint16, np.int32, np.int64, np.uint64):
            for layout in ('C', 'F', 'strided', 'negative-stride'):
                for propagation in (False, True):
                    with self.subTest(dtype=dtype, layout=layout, propagation=propagation):
                        args = state()
                        args[7] = np.array([0, 1], dtype=np.int64)
                        args[8] = np.array([0, 1], dtype=np.int64)
                        args[9] = np.ones(2)
                        # int8 cannot encode 255, but is a compatible integer table.
                        value = 127 if dtype == np.int8 else 255
                        table = np.full((3, 2), value, dtype=dtype, order=layout if layout in ('C', 'F') else 'C')
                        table[0, 0] = 0
                        table[1, 1] = 0
                        if layout == 'strided':
                            storage = np.zeros((6, 4), dtype=dtype)
                            storage[::2, ::2] = table
                            table = storage[::2, ::2]
                        elif layout == 'negative-stride':
                            table = table[::-1].copy()[::-1]
                        args[10] = table
                        reference = copy.deepcopy(args)
                        seed_only(19)
                        if propagation:
                            raw = pcc_numba._pcc_propagate_numba_sequential(
                                *reference, *buffers(reference), 30, True, 2000, 1, True)
                            expected_info = propagation_info(raw[0], True, 30, 1, raw[1])
                        else:
                            pcc_numba._pcc_step_numba_sequential(*reference, *buffers(reference))
                            expected_info = None
                        continuation = [next_random() for _ in range(5)]
                        seed_only(19)
                        info = invoke(args, propagation)
                        self.assertEqual(info, expected_info)
                        self.assertIs(args[10], table)
                        for i in (7, 9, 10, 11, 12):
                            np.testing.assert_array_equal(args[i], reference[i])
                        self.assertEqual(continuation, [next_random() for _ in range(5)])

    def test_zero_particles_accepts_empty_active_distance_region(self):
        for dtype in (np.uint8, np.int64):
            args = state()
            args[7] = np.empty(0, dtype=np.int64)
            args[8] = np.empty(0, dtype=np.int64)
            args[9] = np.empty(0)
            args[10] = np.empty((3, 0), dtype=dtype)
            seed_only(19)
            invoke(args)
            observed_random = next_random()
            seed_only(19)
            self.assertEqual(observed_random, next_random())

    def test_valid_padded_table_checks_only_active_region_and_keeps_identity(self):
        for dtype in (np.uint8, np.int64):
            args = state()
            table = np.full((4, 2), 255 if dtype == np.uint8 else 10**12, dtype=dtype)
            table[:3, :1] = np.array([[0], [255], [255]], dtype=dtype)
            args[10] = table
            original = table.copy()
            kernel = pcc_numba._pcc_step_numba_sequential
            with patch.object(pcc_numba, '_pcc_step_numba_sequential', wraps=kernel) as native:
                invoke(args)
                self.assertIs(native.call_args.args[10], table)
            np.testing.assert_array_equal(table[3:], original[3:])
            np.testing.assert_array_equal(table[:, 1:], original[:, 1:])

    def test_original_crash_inputs_are_rejected_in_isolated_process(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        result = subprocess.run(
            [sys.executable, '-m', 'tests.test_numba_distance_validation', '--crash-probe'],
            capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('rejected 4 unsafe calls', result.stdout)


def crash_probe():
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetErrorMode(0x8003)
    for propagation in (False, True):
        for value in (-10**12, 10**12):
            args = state()
            args[10] = np.full((3, 1), value, dtype=np.int64)
            try:
                invoke(args, propagation)
            except ValueError:
                continue
            raise AssertionError('unsafe distance table was accepted')
    print('rejected 4 unsafe calls')


if __name__ == '__main__':
    if sys.argv[1:] == ['--crash-probe']:
        crash_probe()
    else:
        unittest.main()
