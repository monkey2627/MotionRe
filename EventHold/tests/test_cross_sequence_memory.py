import unittest

import torch

from eventhold.baseline import CausalBaseline
from eventhold.diagnose_cross_sequence_memory import select_cross_donor


class CrossSequenceMemoryTests(unittest.TestCase):
    def test_cross_sequence_donor_uses_hidden_distance_only(self):
        model = CausalBaseline(hidden=2)
        clean = torch.zeros(1, 1, 2)
        pool = [
            ("a.npz", 0, torch.tensor([[[1.0, 0.0]]])),
            ("b.npz", 0, torch.tensor([[[0.0, 3.0]]])),
        ]
        distance, source, frame, donor = select_cross_donor(model, "all", clean, pool)
        self.assertEqual(source, "b.npz")
        self.assertEqual(frame, 0)
        self.assertAlmostEqual(distance, 3.0)
        torch.testing.assert_close(donor, pool[1][2])


if __name__ == "__main__":
    unittest.main()
