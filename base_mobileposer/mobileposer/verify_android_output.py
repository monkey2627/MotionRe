"""Compare an Android SMPL JSONL file with the exported PyTorch reference."""
import argparse
import json
from pathlib import Path

import numpy as np


def axis_angle_matrix(angles):
    result = []
    for vector in np.asarray(angles, dtype=np.float64).reshape(24, 3):
        angle = np.linalg.norm(vector)
        if angle < 1e-12:
            result.append(np.eye(3))
            continue
        x, y, z = vector / angle
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        result.append(np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew))
    return np.stack(result)


def verify(output, reference, tolerance=1e-4, allow_sparse=False):
    expected = json.loads(reference.read_text())
    frames = expected['frames']
    count, max_error = 0, 0.0
    previous_index = -1
    with output.open() as stream:
        for ordinal, line in enumerate(stream):
            actual = json.loads(line)
            index = actual['frame_index']
            assert isinstance(index, int) and previous_index < index
            previous_index = index
            if not allow_sparse:
                assert index == ordinal
            if index >= len(frames):
                raise ValueError('Use a single-pass output, not a looping benchmark')
            assert actual['model_type'] == 'smpl'
            assert actual['layout'] == expected['layout']
            assert actual['rotation_units'] == 'radians'
            assert len(actual['global_orient']) == 3 and len(actual['body_pose']) == 69
            assert actual['transl'] == [0, 0, 0] and actual['betas'] == [0] * 10
            assert abs(actual['pose_time_seconds'] - frames[index]['pose_time_seconds']) < 1e-8
            rotations = axis_angle_matrix(actual['global_orient'] + actual['body_pose'])
            raw = np.asarray(actual['smpl_local_rotations']).reshape(24, 3, 3)
            assert np.isfinite(rotations).all() and np.isfinite(raw).all()
            target = np.asarray(frames[index]['smpl_local_rotations'])
            max_error = max(max_error, float(np.abs(rotations - target).max()), float(np.abs(raw - target).max()))
            count += 1
    assert count > 0
    if not allow_sparse:
        assert count == len(frames), f'Expected {len(frames)} frames, got {count}'
    assert max_error <= tolerance, f'Max matrix error {max_error} > {tolerance}'
    return {'layout': expected['layout'], 'frames': count, 'sparse': allow_sparse,
            'axis_angle_and_matrix_max_abs_error': max_error, 'passed': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--allow-sparse', action='store_true', help='Verify sampled emulator outputs; not a complete sequence test')
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.reference, allow_sparse=args.allow_sparse), indent=2))
