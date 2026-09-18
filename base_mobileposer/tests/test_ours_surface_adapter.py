import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from mobileposer.test_ours_surface import ROLES, bone_orientation, fit_heading, yaw_matrix, humanpose_reference


class AdapterTests(unittest.TestCase):
    def test_anatomical_mapping(self):
        self.assertEqual(ROLES['lw'], 'LEFT_LOWER_ARM')
        self.assertEqual(ROLES['lu'], 'LEFT_UPPER_ARM')
        self.assertEqual(ROLES['ls'], 'LEFT_LOWER_LEG')
        self.assertEqual(ROLES['lf'], 'LEFT_FOOT')

    def test_down_reference_arms_and_upright_legs(self):
        # Server identity is arms-down, whereas SMPL identity is T-pose.
        np.testing.assert_allclose(bone_orientation(np.eye(3), 'lu') @ [1, 0, 0], [0, -1, 0], atol=1e-7)
        np.testing.assert_allclose(bone_orientation(np.eye(3), 'rw') @ [-1, 0, 0], [0, -1, 0], atol=1e-7)
        np.testing.assert_allclose(bone_orientation(np.eye(3), 'lt'), np.eye(3), atol=1e-7)

    def test_heading_recovered_without_body_labels(self):
        raw = Rotation.random(150, random_state=42).as_matrix()
        mount = Rotation.from_rotvec([.3, -.8, .4]).as_matrix()
        heading = yaw_matrix(1.17)
        adjusted = heading @ raw @ mount
        recovered, diagnostics = fit_heading(raw, adjusted)
        np.testing.assert_allclose(recovered, heading, atol=1e-5)
        self.assertLess(diagnostics['delta_residual_p95_deg'], .001)

    def test_pure_yaw_cannot_calibrate_heading(self):
        raw = Rotation.from_rotvec(np.linspace(0, 2, 100)[:, None] * [0, 1, 0]).as_matrix()
        with self.assertRaisesRegex(ValueError, 'Unobservable'):
            fit_heading(raw, yaw_matrix(.8) @ raw)

    def test_humanpose_reference_preserves_geometry_and_timestamps(self):
        bones = ['pelvis', 'femur_l', 'femur_r', 'tibia_l', 'tibia_r', 'talus_l', 'talus_r',
                 'toes_l', 'toes_r', 'lumbar_body', 'thorax', 'head', 'humerus_l', 'humerus_r',
                 'ulna_l', 'ulna_r', 'hand_l', 'hand_r']
        positions = [{'x': float(i), 'y': float(i+1), 'z': float(i+2)} for i in range(18)]
        package = {'bones': bones, 'coordinateSystem': 'unity_world_xyz', 'frames': [
            {'time': t, 'positions': positions, 'present': [True]*18} for t in (0., .034, .065)]}
        ref, ids, times = humanpose_reference(package)
        np.testing.assert_allclose(times, [0., .034, .065])
        np.testing.assert_allclose(ref[0, ids[12]], [-12., 13., 14.])
        np.testing.assert_allclose(np.linalg.norm(ref[0, ids[12]]-ref[0, ids[14]]), np.sqrt(12), rtol=1e-6)
        package['frames'][1]['present'][0] = False
        with self.assertRaisesRegex(ValueError, 'missing bones'):
            humanpose_reference(package)


if __name__ == '__main__':
    unittest.main()
