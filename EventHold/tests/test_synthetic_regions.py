import unittest

import numpy as np

from eventhold.evaluate_synthetic_regions import segment_mean


class SyntheticRegionMetricTests(unittest.TestCase):
    def test_segment_mean_is_explicit_and_does_not_pad(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(segment_mean(values, 1, 3), 2.5)
        self.assertEqual(segment_mean(values, 0, 4), 2.5)


if __name__ == "__main__":
    unittest.main()
