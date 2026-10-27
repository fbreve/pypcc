"""Direct propagation iteration controls reject invalid values consistently."""
import unittest
import numpy as np
from pcc_validation import validate_propagation_controls


class PropagationControlValidationTests(unittest.TestCase):
    def test_valid_controls(self):
        for controls in ((0, 0, 0), (1, 2000, 1),
                         (np.int64(10), np.int64(20), np.int64(2)),
                         (-2, -20, -1)):
            with self.subTest(controls=controls):
                self.assertIsNone(validate_propagation_controls(*controls))

    def test_invalid_controls(self):
        for index, name in enumerate(("max_iter", "es_chk", "stop_max")):
            for value in (1.5, True, float("nan"), "2"):
                with self.subTest(name=name, value=value):
                    controls = [10, 2000, 1]
                    controls[index] = value
                    with self.assertRaisesRegex(ValueError, name):
                        validate_propagation_controls(*controls)


if __name__ == "__main__":
    unittest.main()
