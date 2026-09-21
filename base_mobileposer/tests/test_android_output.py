"""Checks for the independent exported-SMPL validation command."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mobileposer.verify_android_output import verify


class AndroidOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / 'reference.json'
        self.output = self.root / 'output.jsonl'
        rotations = np.tile(np.eye(3), (24, 1, 1))
        rotations[0] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.record = {
            'model_type': 'smpl', 'layout': 'legs_waist', 'frame_index': 0,
            'rotation_units': 'radians', 'global_orient': [0, 0, math.pi / 2],
            'body_pose': [0] * 69, 'transl': [0] * 3, 'betas': [0] * 10,
            'pose_time_seconds': 0, 'smpl_local_rotations': rotations.flatten().tolist(),
        }
        self.reference.write_text(json.dumps({'layout': 'legs_waist', 'frames': [
            {'pose_time_seconds': 0, 'smpl_local_rotations': rotations.tolist()}]}))

    def write_output(self):
        self.output.write_text(json.dumps(self.record) + '\n')

    def test_axis_angle_and_matrix_both_match(self):
        self.write_output()
        self.assertTrue(verify(self.output, self.reference)['passed'])

    def test_wrong_axis_angle_is_rejected_even_if_matrix_matches(self):
        self.record['global_orient'] = [0, 0, 0]
        self.write_output()
        with self.assertRaises(AssertionError):
            verify(self.output, self.reference)

    def test_empty_capture_is_not_success(self):
        self.output.write_text('')
        with self.assertRaises(AssertionError):
            verify(self.output, self.reference, allow_sparse=True)


if __name__ == '__main__':
    unittest.main()
