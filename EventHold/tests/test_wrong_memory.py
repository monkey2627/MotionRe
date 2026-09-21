import unittest

import numpy as np
import torch

from eventhold.baseline import CausalBaseline
from eventhold.diagnose_wrong_memory import (
    donor_indices,
    first_sustained_below,
    inject_state,
    summarize,
)
from eventhold.region_event_model import RegionEventWrite


class WrongMemoryDiagnosticTests(unittest.TestCase):
    def test_b0_only_allows_full_state_transplant(self):
        model = CausalBaseline(hidden=6)
        clean = torch.zeros(2, 1, 6)
        donor = torch.ones(2, 1, 6)
        result = inject_state(model, clean, donor, "all")
        torch.testing.assert_close(result, donor)
        self.assertEqual(float(clean.sum()), 0.0)
        with self.assertRaises(ValueError):
            inject_state(model, clean, donor, "lower")

    def test_m1_region_transplant_does_not_change_other_partitions(self):
        model = RegionEventWrite(hidden=9)
        clean = torch.zeros(1, 9)
        donor = torch.arange(9, dtype=torch.float32).view(1, 9)
        lower = inject_state(model, clean, donor, "lower")
        upper = inject_state(model, clean, donor, "upper")
        torch.testing.assert_close(lower[:, :3], donor[:, :3])
        torch.testing.assert_close(lower[:, 3:], clean[:, 3:])
        torch.testing.assert_close(upper[:, 3:6], donor[:, 3:6])
        torch.testing.assert_close(upper[:, :3], clean[:, :3])
        torch.testing.assert_close(upper[:, 6:], clean[:, 6:])

    def test_recovery_requires_a_complete_sustained_run(self):
        values = np.array([4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0])
        valid = np.ones(len(values), dtype=bool)
        self.assertEqual(first_sustained_below(values, valid, 2.0, 3), 4)
        valid[5] = False
        self.assertIsNone(first_sustained_below(values, valid, 2.0, 3))

    def test_donor_pool_never_uses_state_after_event_start(self):
        self.assertEqual(donor_indices(10, 4), [0, 4, 8, 10])
        self.assertTrue(all(index <= 10 for index in donor_indices(10, 4)))

    def test_summary_keeps_censored_events_in_denominator(self):
        base = {
            "scope": "all",
            "subject_key": "s1",
            "initial_output_delta_deg": 4.0,
            "initial_gt_excess_deg": 2.0,
        }
        rows = [
            dict(base, effective_wrong_injection=True, recovery_2deg_seconds=1.5, unrecovered_2deg=False),
            dict(base, effective_wrong_injection=True, recovery_2deg_seconds=None, unrecovered_2deg=True),
            dict(base, effective_wrong_injection=False, recovery_2deg_seconds=None, unrecovered_2deg=False),
        ]
        summary = summarize(rows, 2.0)["all"]
        self.assertEqual(summary["effective_wrong_injections"], 2)
        self.assertEqual(summary["unrecovered"], 1)
        self.assertEqual(summary["unrecovered_fraction"], 0.5)
        self.assertEqual(summary["median_recovery_seconds_among_recovered"], 1.5)


if __name__ == "__main__":
    unittest.main()
