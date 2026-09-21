"""Export the existing online SMPL pose branch; no training or mesh at runtime.

python -m mobileposer.mobile_export --output android/assets_generated
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = {
    'wrists_thighs_waist': ['lw', 'rw', 'lt', 'rt', 'waist'],
    'wrists_shanks_waist': ['lw', 'rw', 'ls', 'rs', 'waist'],
    'wrists_feet_waist': ['lw', 'rw', 'lf', 'rf', 'waist'],
    'upperarms_thighs_waist': ['lu', 'ru', 'lt', 'rt', 'waist'],
    'wrists_upperarms_waist': ['lw', 'rw', 'lu', 'ru', 'waist'],
    'legs_waist': ['lt', 'rt', 'ls', 'rs', 'waist'],
}
PARENTS = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
REDUCED = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
IGNORED = [7, 8, 10, 11, 20, 21, 22, 23]
WINDOW, OUTPUT_INDEX = 45, 40


class DenseRnn(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear1 = nn.Linear(inputs, 256)
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True)
        self.linear2 = nn.Linear(512, outputs)

    def forward(self, x):
        # Original RNN uses packed batch-first data but its LSTM is time-first.
        x = torch.relu(self.linear1(x)).transpose(0, 1)
        x, _ = self.rnn(x)
        return self.linear2(x.transpose(0, 1))


class MobilePose(nn.Module):
    def __init__(self):
        super().__init__()
        self.joints = DenseRnn(60, 72)
        self.pose = DenseRnn(132, 96)
        self.register_buffer('identity', torch.eye(3))

    def load_combined(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        for module, prefix in [(self.joints, 'joints.joints.'), (self.pose, 'pose.pose.')]:
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        return self.eval()

    def forward(self, imu):
        joints = self.joints(imu)
        six = self.pose(torch.cat([joints, imu], dim=-1))[0, OUTPUT_INDEX].reshape(16, 6)
        c0 = six[:, :3] / torch.linalg.vector_norm(six[:, :3], dim=-1, keepdim=True)
        c1 = six[:, 3:] - (c0 * six[:, 3:]).sum(-1, keepdim=True) * c0
        c1 = c1 / torch.linalg.vector_norm(c1, dim=-1, keepdim=True)
        rot = torch.stack([c0, c1, torch.cross(c0, c1, dim=-1)], dim=-1)
        rot = torch.where(torch.isnan(rot), torch.zeros_like(rot), rot)
        global_rot = [rot[REDUCED.index(j)] if j in REDUCED else self.identity for j in range(24)]
        local = [global_rot[0]]
        for j in range(1, 24):
            local.append(self.identity if j in IGNORED else global_rot[PARENTS[j]].T @ global_rot[j])
        return torch.stack(local)


def windows(frames):
    window = None
    for frame in frames:
        window = np.repeat(frame[None], WINDOW, axis=0) if window is None else np.concatenate([window[1:], frame[None]])
        yield window[None].astype(np.float32)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_model(path):
    # Use the unmodified repository implementation as an independent oracle.
    from mobileposer.config import model_config
    model_config.device = torch.device('cpu')
    from mobileposer.models import MobilePoserNet
    model = MobilePoserNet().cpu().eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    if model.bodymodel.parent[1:] != PARENTS[1:]:
        raise ValueError('SMPL parent order differs from export contract')
    return model


def make_fixture(data_path, layout, count):
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    if data.get('layout') != layout:
        raise ValueError(f'Incorrect layout in {data_path}')
    synthesis = data.get('synthesis', {})
    if synthesis.get('mode') != 'surface' or synthesis.get('fps') != 30:
        raise ValueError('Expected 30 Hz surface data')
    sequence = next(i for i, a in enumerate(data['acc']) if len(a) >= count)
    acc = data['acc'][sequence][:count, :5].float().numpy()
    ori = data['ori'][sequence][:count, :5].float().numpy()
    frames = [{'time_seconds': i / 30, 'acceleration': a.tolist(), 'orientation': r.tolist()}
              for i, (a, r) in enumerate(zip(acc, ori))]
    fixture = {'format_version': 1, 'layout': layout, 'fps': 30, 'sensor_labels': LAYOUTS[layout],
               'acceleration_units': 'm/s2', 'acceleration_space': 'world_linear',
               'orientation_space': 'bone_calibrated', 'frames': frames,
               'source': {'file': str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
                          'sequence': sequence, 'synthetic': True}}
    features = np.concatenate([acc.reshape(count, 15) / 30, ori.reshape(count, 45)], axis=1)
    return fixture, features


def export(args):
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'pytorch': torch.__version__, 'onnxruntime': ort.__version__, 'tolerance': 1e-4,
              'android_device': 'not tested', 'layouts': {}}
    models = []
    for layout in args.layouts:
        print(f'Exporting {layout}', flush=True)
        weights = args.checkpoints / layout / '1/base_model.pth'
        model = MobilePose().load_combined(weights)
        fixture, features = make_fixture(args.data / layout / 'dip_test.pt', layout, args.frames)
        target = args.output / f'{layout}.onnx'
        example = torch.from_numpy(next(windows(features)))
        with torch.no_grad():
            torch.onnx.export(model, example, str(target), input_names=['imu'], output_names=['smpl_local_rotations'],
                              opset_version=17, **({'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}))
        onnx.checker.check_model(str(target))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
        original = reference_model(weights)
        refs, wrapper_error, ort_error = [], 0.0, 0.0
        for i, window in enumerate(windows(features)):
            with torch.no_grad():
                tensor = torch.from_numpy(window)
                joints = original.joints(tensor, [WINDOW])
                pose6 = original.pose(torch.cat([joints, tensor], dim=-1), [WINDOW])
                expected = original._reduced_global_to_full(pose6)[OUTPUT_INDEX].numpy()
                actual = model(tensor).numpy()
            onnx_pose = session.run(None, {'imu': window})[0]
            if not np.isfinite(onnx_pose).all():
                raise ValueError('Non-finite ONNX output')
            wrapper_error = max(wrapper_error, float(np.abs(expected - actual).max()))
            ort_error = max(ort_error, float(np.abs(expected - onnx_pose).max()))
            refs.append({'frame_index': i, 'pose_time_seconds': max(0, i - 4) / 30,
                         'smpl_local_rotations': expected.tolist()})
        metrics = {'frames': len(refs), 'wrapper_max_abs_error': wrapper_error, 'onnx_max_abs_error': ort_error,
                   'bytes': target.stat().st_size, 'parameters': sum(p.numel() for p in model.parameters())}
        report['layouts'][layout] = metrics
        if max(wrapper_error, ort_error) > report['tolerance']:
            raise AssertionError(f'{layout}: {metrics}')
        (args.output / f'{layout}.input.json').write_text(json.dumps(fixture, separators=(',', ':')))
        (args.output / f'{layout}.reference.json').write_text(json.dumps({'layout': layout, 'frames': refs}, separators=(',', ':')))
        models.append({'layout': layout, 'sensor_labels': LAYOUTS[layout], 'model': target.name,
                       'sha256': sha256(target), 'checkpoint_sha256': sha256(weights), **metrics})
        print(metrics, flush=True)
    manifest = {'format_version': 1, 'input_shape': [1, WINDOW, 60], 'output_shape': [24, 3, 3],
                'fps': 30, 'output_index': OUTPUT_INDEX, 'ignored_joints': IGNORED, 'models': models}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'validation_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints/no_head_5imu_surface')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/no_head_5imu_surface_dip_test')
    parser.add_argument('--output', type=Path, default=ROOT / 'android/assets_generated')
    parser.add_argument('--layouts', nargs='+', choices=list(LAYOUTS), default=list(LAYOUTS))
    parser.add_argument('--frames', type=int, default=180)
    args = parser.parse_args()
    if args.frames < WINDOW:
        parser.error('--frames must be at least 45')
    export(args)
