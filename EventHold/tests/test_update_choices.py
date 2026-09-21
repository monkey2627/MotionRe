import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from eventhold.diagnose_update_choices import rollouts


def rotations(degrees):
    return Rotation.from_euler('z', degrees, degrees=True).as_matrix()[:, None]


class ChoiceDiagnosticTests(unittest.TestCase):
    def test_oracle_keeps_good_and_replaces_bad_memory(self):
        q = rotations([0, 40, 80, 90])
        gt = rotations([0, 0, 90, 90])
        result, decisions = rollouts(q, gt, np.ones(4, bool))
        self.assertEqual(decisions.tolist(), [True, False, True, True])
        np.testing.assert_allclose(result['gt_greedy_keep_write'], rotations([0, 0, 80, 90]))

    def test_nonoracle_paths_do_not_read_ground_truth(self):
        q = rotations([5, 20, 60])
        a, _ = rollouts(q, rotations([0, 0, 0]), np.ones(3, bool))
        b, _ = rollouts(q, rotations([90, 90, 90]), np.ones(3, bool))
        for name in ['write', 'keep_entry', 'fixed_so3_ema']:
            np.testing.assert_allclose(a[name], b[name])

    def test_invalid_reference_never_drives_oracle(self):
        q = rotations([0, 90]);gt = q.copy();gt[1] = np.nan
        result, decision = rollouts(q, gt, np.array([True, False]))
        self.assertFalse(decision[1])
        np.testing.assert_allclose(result['gt_greedy_keep_write'][1], q[0])

    def test_interpolation_endpoints_and_rotation_validity(self):
        q = rotations([5, 80, 179.9]);valid = np.ones(3, bool)
        a, _ = rollouts(q, q, valid, alpha=0)
        b, _ = rollouts(q, q, valid, alpha=1)
        np.testing.assert_allclose(a['fixed_so3_ema'], a['keep_entry'])
        np.testing.assert_allclose(b['fixed_so3_ema'], q, atol=1e-12)
        c, _ = rollouts(q, q, valid)
        np.testing.assert_allclose(np.linalg.det(c['fixed_so3_ema']), 1, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
