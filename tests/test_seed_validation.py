"""Seed validation must agree between the public API and direct backends."""
import unittest
import numpy as np
from pcc_validation import validate_seed


class SeedValidationTests(unittest.TestCase):
    def test_valid_values(self):
        for seed in (None, 0, 1, 2**32 - 1, np.int64(42)):
            with self.subTest(seed=seed):
                result = validate_seed(seed)
                self.assertEqual(result, None if seed is None else int(seed))
                if result is not None:
                    self.assertIs(type(result), int)

    def test_invalid_values(self):
        for seed in (-1, 2**32, 1.5, True, np.bool_(False), "42",
                     float("nan"), float("inf")):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed must be an integer"):
                    validate_seed(seed)


if __name__ == "__main__":
    unittest.main()
