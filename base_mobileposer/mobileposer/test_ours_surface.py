"""Strict physical-IMU inference on one motion package; no pose-derived inputs.

Run with the project's mobileposer Python environment. See the generated report
for missing layouts and sensor-only heading calibration diagnostics.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation, Slerp

from mobileposer.config import paths, model_config
from mobileposer.no_head_layouts import LAYOUTS
from mobileposer.surface_imu import file_hash, DEFAULT_CONFIG

FPS = 30
ROLES = {
    'lw': 'LEFT_LOWER_ARM', 'rw': 'RIGHT_LOWER_ARM',
    'lu': 'LEFT_UPPER_ARM', 'ru': 'RIGHT_UPPER_ARM',
    'lt': 'LEFT_UPPER_LEG', 'rt': 'RIGHT_UPPER_LEG',
    'ls': 'LEFT_LOWER_LEG', 'rs': 'RIGHT_LOWER_LEG',
    'lf': 'LEFT_FOOT', 'rf': 'RIGHT_FOOT', 'waist': 'WAIST',
}
# SolarXR -> VMC Unity (reflect z) -> SMPL (reflect x). Both are Y-up.
# AMASS's source Z-up conversion must NOT be applied to this Y-up stream.
SERVER_TO_SMPL = np.diag([-1., 1., -1.])


def quat(sample, field):
    q = np.array([sample[field][k] for k in ('x', 'y', 'z', 'w')])
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError(f'Invalid {field}')
    return Rotation.from_quat(q).as_matrix()


def yaw_matrix(angle):
    return Rotation.from_rotvec([0., angle, 0.]).as_matrix()


def fit_heading(raw, adjusted):
    """Solve adjusted[t] = Yaw @ raw[t] @ mounting, without any body GT.

    SlimeVR getReferenceAdjustedAccel only applies rawRot.sandwich(accel).
    A constant right mounting rotation cancels in inter-frame world deltas.
    All recorded frames are used: this is offline calibration, not streaming.
    """
    a = raw @ raw[0].T
    b = adjusted @ adjusted[0].T

    def objective(angle):
        h = yaw_matrix(angle)
        return float(np.mean((h @ a @ h.T - b) ** 2))

    grid = np.linspace(-np.pi, np.pi, 181)
    values = np.array([objective(x) for x in grid])
    best = grid[values.argmin()]
    fit = minimize_scalar(objective, bounds=(best - np.pi/90, best + np.pi/90), method='bounded')
    h = yaw_matrix(fit.x)
    residual = Rotation.from_matrix((h @ a @ h.T).transpose(0, 2, 1) @ b).magnitude()
    diagnostic = {
        'yaw_deg': float(np.rad2deg(fit.x)),
        'delta_residual_mean_deg': float(np.rad2deg(residual).mean()),
        'delta_residual_p95_deg': float(np.quantile(np.rad2deg(residual), .95)),
        'objective_contrast': float(np.ptp(values)),
        'method': 'offline sensor-only constant yaw, all frames; no FBX/pose labels',
    }
    if np.ptp(values) < 1e-5 or diagnostic['delta_residual_p95_deg'] > 10:
        raise ValueError(f'Unobservable or inconsistent acceleration heading: {diagnostic}')
    return h, diagnostic


def bone_orientation(adjusted, label):
    # UnityArmature.setGlobalRotationForBone: left arm +90deg Z, right -90deg Z.
    # This changes the down-reference arm frame into the T-pose bone frame.
    angle = np.pi/2 if label in ('lw', 'lu') else -np.pi/2 if label in ('rw', 'ru') else 0.
    offset = Rotation.from_rotvec([0., 0., angle]).as_matrix()
    return SERVER_TO_SMPL @ (adjusted @ offset) @ SERVER_TO_SMPL.T


def prepare(capture, output):
    manifest = json.loads((capture / 'manifest.json').read_text())
    raw_path = capture / manifest['rawImuJsonFile']
    digest = file_hash(raw_path)
    if digest != manifest['rawImuJsonSha256']:
        raise ValueError('Raw IMU manifest hash mismatch')
    package = json.loads(raw_path.read_text())
    frames = package['frames']
    times = np.array([f['timeSeconds'] for f in frames])
    if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Non-monotonic/invalid capture times')
    target = times[0] + np.arange(int(np.floor((times[-1]-times[0])*FPS)) + 1) / FPS
    signals, diagnostics = {}, {}
    for label, role in ROLES.items():
        samples = []
        for f in frames:
            matches = [s for s in f['sensors'] if s['trackerRole'] == role]
            if len(matches) > 1:
                raise ValueError(f'Duplicate role {role}')
            samples.append(matches[0] if matches else None)
        if any(s is None for s in samples):
            diagnostics[label] = {'role': role, 'status': 'missing_physical_sensor'}
            continue
        try:
            required = ('online', 'hasAcceleration', 'hasOrientation', 'hasRotationReferenceAdjusted')
            if any(not all(s.get(k, False) for k in required) for s in samples):
                raise ValueError('Missing/offline/invalid signal; refusing silent fill')
            if len({s['sensorId'] for s in samples}) != 1:
                raise ValueError('Sensor identity changed during capture')
            if any(s['accelerationKind'] != 'linear' for s in samples):
                raise ValueError('Expected world-space linear acceleration from SolarXR')
            raw = np.stack([quat(s, 'orientation') for s in samples])
            ref = np.stack([quat(s, 'rotationReferenceAdjusted') for s in samples])
            heading, diagnostic = fit_heading(raw, ref)
            acc = np.array([[s['accelerationG'][k] for k in ('x', 'y', 'z')] for s in samples]) * 9.80665
            if not np.isfinite(acc).all():
                raise ValueError('Nonfinite acceleration')
            acc = acc @ heading.T @ SERVER_TO_SMPL.T
            ori = bone_orientation(ref, label)
            acc = np.stack([np.interp(target, times, acc[:, k]) for k in range(3)], axis=-1)
            ori = Slerp(times, Rotation.from_matrix(ori))(target).as_matrix()
            signals[label] = (torch.tensor(acc, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32))
            diagnostics[label] = dict(diagnostic, role=role, status='ok', sensor_id=samples[0]['sensorId'],
                                      acc_rms_m_s2=float(np.sqrt(np.mean(acc**2))))
        except ValueError as e:
            diagnostics[label] = {'role': role, 'status': 'invalid', 'reason': str(e)}
    report = {
        'capture': str(capture.resolve()), 'raw_imu_sha256': digest,
        'source_frames': len(times), 'resampled_frames': len(target), 'fps': FPS,
        'source_dt_max_s': float(np.diff(times).max()),
        'coordinate_system': 'SMPL Y-up; Server -> reflect Z (VMC) -> reflect X (SMPL)',
        'reference_used_for_input': False, 'sensors': diagnostics, 'layouts': {},
    }
    if np.diff(times).max() > .2:
        raise ValueError('Capture has >200 ms recording gap')
    for layout, spec in LAYOUTS.items():
        missing = [x for x in spec['labels'] if x not in signals]
        if missing:
            report['layouts'][layout] = {'status': 'unavailable', 'missing_labels': missing}
            continue
        data = {
            'layout': layout, 'layout_labels': spec['labels'], 'layout_joints': spec['joints'],
            'acc': [torch.stack([signals[x][0] for x in spec['labels']], dim=1)],
            'ori': [torch.stack([signals[x][1] for x in spec['labels']], dim=1)],
            'timestamps': torch.tensor(target), 'fps': FPS,
            'metadata': [{'source': str(capture.resolve()), 'signal_source': 'physical_imu_only',
                          'raw_imu_sha256': digest, 'reference_used_for_input': False}],
        }
        dest = output / 'inputs' / layout / 'ours1.pt'
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, dest)
        report['layouts'][layout] = {'status': 'prepared', 'input': str(dest), 'labels': spec['labels']}
    return report


def load_fbx_reference(capture, output, blender):
    """Fresh export of THIS package's FBX; reference only, never model inputs."""
    from mobileposer.fit_ours_smpl import _export_fbx_joints, _FBX_BONES
    manifest = json.loads((capture / 'manifest.json').read_text())
    fbx = capture / manifest['fbxFile']
    if file_hash(fbx) != manifest['fbxSha256']:
        raise ValueError('FBX manifest hash mismatch')
    dest = output / 'reference' / '1.fbx_joints.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    _export_fbx_joints(blender, fbx.resolve(), dest.resolve(), True)
    cache = np.load(dest)
    meta = json.loads(str(cache['meta']))
    if not all(meta['present']) or meta['bones'] != _FBX_BONES:
        raise ValueError('Incomplete FBX skeleton')
    # Blender imported this MayaYUp FBX into Blender Z-up. FBX already reflects
    # Unity X. Undo Blender's Y-up -> Z-up transform, without another X flip.
    p = cache['positions'][..., [0, 2, 1]].copy()
    p[..., 2] *= -1
    ref = np.zeros((len(p), 24, 3), dtype=np.float32)
    ids = [0, 1, 2, 4, 5, 7, 8, 10, 11, 3, 9, 15, 16, 17, 18, 19, 20, 21]
    ref[:, ids] = p
    ref[:, 6] = (ref[:, 3] + ref[:, 9]) / 2
    ref[:, 12] = (ref[:, 9] + ref[:, 15]) / 2
    ref[:, 13] = (ref[:, 9] + ref[:, 16]) / 2
    ref[:, 14] = (ref[:, 9] + ref[:, 17]) / 2
    ref[:, 22:24] = ref[:, 20:22]
    np.save(output / 'reference' / 'joints_y_up.npy', ref)
    return ref, ids, {'fbx_sha256': file_hash(fbx), 'fps': meta['fps'], 'frames': len(ref),
                      'kind': 'SlimeVR-driven avatar; NOT independent motion-capture GT',
                      'interpolated_joints_for_display_only': [6, 12, 13, 14, 22, 23]}


def humanpose_reference(package):
    """Read actual Unity Humanoid playback joints, retaining source timestamps."""
    expected = ['pelvis', 'femur_l', 'femur_r', 'tibia_l', 'tibia_r', 'talus_l', 'talus_r',
                'toes_l', 'toes_r', 'lumbar_body', 'thorax', 'head', 'humerus_l', 'humerus_r',
                'ulna_l', 'ulna_r', 'hand_l', 'hand_r']
    if package['bones'] != expected or package['coordinateSystem'] != 'unity_world_xyz':
        raise ValueError('Unexpected HumanPose bone order or coordinate system')
    if any(len(f['present']) != 18 or not all(f['present']) for f in package['frames']):
        raise ValueError('HumanPose export has missing bones')
    times = np.array([f['time'] for f in package['frames']], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('HumanPose timestamps must be strictly increasing')
    p = np.array([[[b[k] for k in 'xyz'] for b in f['positions']] for f in package['frames']], dtype=np.float32)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite HumanPose positions')
    p[..., 0] *= -1  # Unity -> right-handed SMPL; both Y-up.
    ref = np.zeros((len(p), 24, 3), dtype=np.float32)
    ids = [0, 1, 2, 4, 5, 7, 8, 10, 11, 3, 9, 15, 16, 17, 18, 19, 20, 21]
    ref[:, ids] = p
    ref[:, 6] = (ref[:, 3] + ref[:, 9]) / 2
    ref[:, 12] = (ref[:, 9] + ref[:, 15]) / 2
    ref[:, 13] = (ref[:, 9] + ref[:, 16]) / 2
    ref[:, 14] = (ref[:, 9] + ref[:, 17]) / 2
    ref[:, 22:24] = ref[:, 20:22]
    return ref, ids, times


def load_reference(capture, output, unity_editor):
    manifest = json.loads((capture/'manifest.json').read_text())
    motion = capture/manifest['motionJsonFile']
    digest = file_hash(motion)
    if digest != manifest['motionJsonSha256']:
        raise ValueError('Motion JSON manifest hash mismatch')
    project = paths.root_dir/'IMUTrack-for-Spine'
    dest = output/'reference/ours1.humanpose_joints.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(unity_editor), '-batchmode', '-nographics', '-projectPath', str(project),
                    '-executeMethod', 'HumanPoseJointExporter.ExportFromCommandLine',
                    '-motionJson', str(motion.resolve()), '-outJson', str(dest.resolve()),
                    '-logFile', str((dest.parent/'unity_export.log').resolve())], check=True)
    package = json.loads(dest.read_text())
    if Path(package['sourceMotionJson']).resolve() != motion.resolve():
        raise ValueError('HumanPose export source mismatch')
    ref, ids, times = humanpose_reference(package)
    np.save(dest.parent/'joints_y_up.npy', ref)
    return ref, ids, {
        'motion_json_sha256': digest, 'humanpose_export_sha256': file_hash(dest),
        'exporter_sha256': file_hash(project/'Assets/Editor/HumanPoseJointExporter.cs'),
        'avatar_asset': package['avatarAssetPath'], 'frames': len(ref), 'fps': package['fps'],
        'timestamps': times.tolist(),
        'kind': 'motion.json -> Unity HumanPoseHandler.SetHumanPose; NOT independent mocap GT',
        'coordinate_conversion': 'Unity Y-up -> SMPL Y-up: (-x,y,z)',
        'interpolated_joints_for_display_only': [6, 12, 13, 14, 22, 23],
    }


def render(ref, pred, dest, title, legend='Avatar reference (blue) / IMU prediction (red)'):
    """Fast full-duration front/side video with fixed scale and root centering."""
    import cv2
    from mobileposer.evaluate_no_head_5imu import EDGES
    ref = ref - ref[:, :1]
    pred = pred - pred[:, :1]
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), 15, (1200, 700))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create {dest}')
    try:
        for i in range(0, len(pred), 2):
            frame = np.full((700, 1200, 3), (24, 20, 16), dtype=np.uint8)
            cv2.putText(frame, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1)
            cv2.putText(frame, f'{legend}   t={i/FPS:.2f}s',
                        (20, 62), cv2.FONT_HERSHEY_SIMPLEX, .6, (220, 220, 220), 1)
            for pane, axis in enumerate((0, 2)):
                origin = np.array([300 + 600*pane, 355])
                cv2.putText(frame, 'Front' if pane == 0 else 'Side', (pane*600+30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 1)
                for seq, color in ((ref, (240, 160, 60)), (pred, (80, 80, 240))):
                    xy = (seq[i][:, [axis, 1]] * [240, -240] + origin).astype(int)
                    for a, b in EDGES:
                        cv2.line(frame, tuple(xy[a]), tuple(xy[b]), color, 2, cv2.LINE_AA)
                    for point in xy:
                        cv2.circle(frame, tuple(point), 3, color, -1)
            writer.write(frame)
            if i == len(pred)//4*2:
                cv2.imwrite(str(dest.with_suffix('.png')), frame)
    finally:
        writer.release()


@torch.no_grad()
def infer(args, report):
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.utils.model_utils import load_model
    model_config.device = torch.device(args.device)
    body = ParametricModel(paths.smpl_file)
    ref, ids, ref_meta = load_reference(args.capture_dir, args.output, args.unity_editor)
    report['reference'] = ref_meta
    for layout, entry in report['layouts'].items():
        if entry['status'] != 'prepared':
            continue
        checkpoint = args.checkpoint_root / layout / '1/base_model.pth'
        synthesis = json.loads((args.checkpoint_root / layout / 'synthesis.json').read_text())
        if synthesis['mode'] != 'surface' or synthesis['attachment_sha256'] != file_hash(DEFAULT_CONFIG):
            raise ValueError(f'{layout}: checkpoint attachment mismatch')
        data = torch.load(entry['input'], map_location='cpu', weights_only=False)
        acc, ori = data['acc'][0], data['ori'][0]
        # Exactly mobileposer.data.PoseDataset: first 5 physical slots, acc / 30.
        imu = torch.cat(((acc/30).flatten(1), ori.flatten(1)), dim=1)
        if args.predictions_root is not None:
            previous = torch.load(args.predictions_root/layout/'prediction.pt', map_location='cpu', weights_only=False)
            old_input = torch.load(args.predictions_root/'inputs'/layout/'ours1.pt', map_location='cpu', weights_only=False)
            if (previous['checkpoint_sha256'] != file_hash(checkpoint)
                    or previous['input_metadata'] != data['metadata']
                    or not torch.equal(previous['timestamps'], data['timestamps'])
                    or not torch.equal(old_input['acc'][0], acc)
                    or not torch.equal(old_input['ori'][0], ori)):
                raise ValueError('Cannot reuse prediction: checkpoint/input mismatch')
            pose, network_joints, tran, contact = [previous[k] for k in ('pose', 'network_joints', 'tran', 'contact')]
            model = None
        else:
            model = load_model(str(checkpoint)).to(args.device).eval()
            model.reset()
            pose, network_joints, tran, contact = model.forward_offline(imu.to(args.device)[None], [len(imu)])
        pose, tran = pose.cpu(), tran.cpu()
        _, joint = body.forward_kinematics(pose, tran=tran)
        if not all(torch.isfinite(x).all() for x in (pose, tran, joint)):
            raise ValueError(f'{layout}: nonfinite prediction')
        dest = args.output / layout
        dest.mkdir(parents=True, exist_ok=True)
        torch.save({'pose': pose, 'tran': tran, 'joint': joint, 'contact': contact.cpu(),
                    'network_joints': network_joints.cpu(), 'fps': FPS, 'timestamps': data['timestamps'],
                    'layout': layout, 'checkpoint_sha256': file_hash(checkpoint),
                    'input_metadata': data['metadata']}, dest / 'prediction.pt')
        # Compare independent branches of the same package. This is agreement
        # with a SlimeVR avatar, NOT reconstruction accuracy against true GT.
        times = data['timestamps'].numpy()
        rt = np.array(ref_meta['timestamps'])
        if times[0] < rt[0] or times[-1] > rt[-1] + 1e-6:
            raise ValueError('HumanPose reference does not cover input timestamps')
        synced = np.stack([np.interp(times, rt, ref[:, j, k]) for j in range(24) for k in range(3)], -1).reshape(-1, 24, 3)
        pred = joint.numpy()
        error = np.linalg.norm((pred-pred[:, :1])[:, ids] - (synced-synced[:, :1])[:, ids], axis=-1)
        entry.update(status='inferred', frames=len(pose), checkpoint=str(checkpoint),
                     checkpoint_sha256=file_hash(checkpoint),
                     avatar_root_relative_joint_disagreement_cm=float(error.mean()*100),
                     reference_joint_ids=ids,
                     per_joint_disagreement_cm=dict(zip(map(str, ids), (error.mean(axis=0)*100).tolist())),
                     prediction=str(dest/'prediction.pt'))
        if not args.no_video:
            render(synced, pred, dest/'comparison.mp4', layout)
        print(layout, entry['avatar_root_relative_joint_disagreement_cm'], flush=True)
        del model
    return report


def write_summary(output, report):
    rows = []
    lines = [
        '# ours/1 实体 IMU → Surface MobilePoser 测试', '',
        f"唯一采集来源：`{report['capture']}`。原始 {report['source_frames']} 帧，按时间戳重采样为 "
        f"{report['resampled_frames']} 帧 / 30 FPS；不在记录时间范围之外外推。", '',
        '输入只来自 raw-imu.json 的实体传感器；没有使用 FBX、Avatar 姿态或虚拟 tracker 生成输入。', '',
        '|配置|输入顺序|状态|与 Avatar 的关节差异 (cm)|视频|',
        '|---|---|---|---:|---|',
    ]
    for layout, entry in report['layouts'].items():
        value = entry.get('avatar_root_relative_joint_disagreement_cm')
        rows.append({'layout': layout, 'status': entry['status'], 'avatar_disagreement_cm': value})
        score = f'{value:.2f}' if value is not None else '—'
        video = f'[对照视频]({layout}/comparison.mp4)' if (output/layout/'comparison.mp4').exists() else '—'
        status = entry['status']
        if status == 'unavailable':
            status += ': ' + ', '.join(entry['missing_labels'])
        lines.append(f"|{layout}|{', '.join(LAYOUTS[layout]['labels'])}|{status}|{score}|{video}|")
    lines += ['', '## 输入约定与位置映射', '',
              '- lu/ru：LEFT/RIGHT_UPPER_ARM；lw/rw：LEFT/RIGHT_LOWER_ARM（远端前臂，不是手部）。',
              '- lt/rt：LEFT/RIGHT_UPPER_LEG；ls/rs：LEFT/RIGHT_LOWER_LEG；waist：WAIST。',
              '- lf/rf：LEFT/RIGHT_FOOT，当前采集缺失，所以 wrists_feet_waist 未测试。',
              '- 根据 trackerRole 映射，不根据旧截图中的硬件 ID 映射；本次 ID 记录于 report.json。',
              '- 使用 rotationReferenceAdjusted，按 SlimeVR UnityArmature 的左右臂 ±90° Z 偏置转换到 T-pose 骨段坐标。',
              '- Server → VMC 反射 Z → SMPL 反射 X；保持 Y-up，不重复套用 AMASS 的 Z-up → Y-up 旋转。',
              '- accelerationG × 9.80665；对已是线性世界加速度的数据不再次旋转原始姿态、不重复减重力，也不假设开头静止而减均值。',
              '- SlimeVR 的线性加速度没有完整航向校正：利用同一传感器全序列 raw/reference 四元数的相对旋转拟合恒定 yaw，完全不读取身体参考姿态。',
              '- 这是离线航向估计；漂移和非恒定校准产生的残差见 report.json。不可观测或残差过大时拒绝该传感器。',
              '- 旋转用 SLERP、加速度用线性插值至 30 FPS。输入为 (T,5,3) / (T,5,3,3)，网络输入 (T,60)，仅加速度除以 30。',
              '- 检查 checkpoint 的 surface attachment 哈希。真实加速度保留实际安装点测量，不用 SMPL 合成信号替换；实际位置与训练皮肤点仍可能有偏差。', '',
              '## 如何解释结果', '',
              '视频蓝色是当前 motion.json 经 Unity HumanPose 播放导出的 Avatar，红色是预测 SMPL。分别逐帧减 pelvis，仅比较根节点相对姿态；不做旋转、尺度或时间偏移拟合。', '',
              '数值为 18 个实际导出对应关节（包含 pelvis）在全部帧上的平均欧氏距离。显示所需的额外脊柱/锁骨节点仅做插值，不参与计分。', '',
              '**Avatar 由同一套 IMU 经 SlimeVR 解算，不是独立动捕真值。** 该指标是两条算法链路的关节一致性，包含体型/骨长差异，不能当作真实重建精度。视频也不能评估全局平移精度。', '',
              '验证 motion.json 的 manifest SHA256，使用软件同款 mesh_edit1.vrm，通过 HumanPoseHandler.SetHumanPose 播放 muscles/bodyRotation，并固定 bodyPosition 到首帧，与软件回放约定一致。Unity → SMPL 为 (-x,y,z)。', '',
              '参考动作按 motion.json 实际时间戳插值对齐 IMU，不用帧序号对应。旧 FBX 参考与该播放路径不一致，旧视频及数值已被此版本替代。', '',
              '每组 prediction.pt 保存 pose (T,24,3,3)、tran (T,3)、joint、contact、timestamps 和 checkpoint 哈希；inputs/ 保存输入与来源。', '',
              '## 复现', '', '在仓库根目录执行：', '', '```bash',
              'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python -m mobileposer.test_ours_surface',
              '```', '',
              '适配回归测试：`python -m unittest discover -s tests -p test_ours_surface_adapter.py`。', '']
    (output/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['layout', 'status', 'avatar_disagreement_cm'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, default=paths.raw_ours/'1')
    parser.add_argument('--checkpoint-root', type=Path, default=paths.root_dir/'checkpoints/no_head_5imu_surface')
    parser.add_argument('--output', type=Path, default=paths.root_dir/'results/ours1_surface_motionjson')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--unity-editor', type=Path, default=Path('/home/duanyuhan/Unity/Hub/Editor/2022.3.13f1/Editor/Unity'))
    parser.add_argument('--predictions-root', type=Path, help='Reuse predictions only after exact input/checkpoint validation')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = prepare(args.capture_dir, args.output)
    report_path = args.output/'report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.prepare_only:
        report = infer(args, report)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_summary(args.output, report)
    print(report_path)


if __name__ == '__main__':
    main()
