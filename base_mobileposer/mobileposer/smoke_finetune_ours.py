"""Same-sequence overfit experiment, not a held-out evaluation.

Consumes only the verified ours/1 inputs and motion.json HumanPose export.
Original checkpoints are read-only. Train joints + poser end-to-end, leaving
velocity and contact weights frozen. Full-sequence training avoids window-edge
differences in this deliberately small overfit experiment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from mobileposer.config import paths, model_config, joint_set
from mobileposer.articulate import math
from mobileposer.articulate.model import ParametricModel
from mobileposer.surface_imu import file_hash
from mobileposer.test_ours_surface import humanpose_reference, render


def make_targets(source, output):
    from mobileposer.retarget_ours_humanpose import retarget_cache
    report = json.loads((source/'report.json').read_text())
    capture = Path(report['capture'])
    manifest = json.loads((capture/'manifest.json').read_text())
    cache = source/'reference/ours1.humanpose_joints.json'
    assert file_hash(cache) == report['reference']['humanpose_export_sha256']
    assert file_hash(capture/manifest['motionJsonFile']) == report['reference']['motion_json_sha256']
    package = json.loads(cache.read_text())
    avatar, ids, times = humanpose_reference(package)
    first = next(e for e in report['layouts'].values() if e['status'] == 'inferred')
    inputs = torch.load(first['input'], map_location='cpu', weights_only=False)
    target_times = inputs['timestamps'].numpy()
    if target_times[0] < times[0] or target_times[-1] > times[-1]:
        raise ValueError('IMU timestamps outside HumanPose recording')
    # Calibrate the avatar rest rig once, then transfer animation rotations.
    # Do not reuse any legacy ours_smpl.pt file, or the incorrect FBX branch.
    pose, _, _, _, calibration = retarget_cache(cache, 'calibrated', 300)
    pose = torch.tensor(np.stack([
        Slerp(times, Rotation.from_matrix(pose[:, j].numpy()))(target_times).as_matrix()
        for j in range(24)], axis=1), dtype=torch.float32)
    # Match the actual model decoder: these local rotations cannot be learned.
    ignored = [j for j in joint_set.ignored if j != 0]
    unprojected_pose = pose.clone()
    pose[:, ignored] = torch.eye(3)
    body = ParametricModel(paths.smpl_file)
    global_pose, joint = body.forward_kinematics(pose)
    ref = np.stack([np.interp(target_times, times, avatar[:, j, k])
                    for j in range(24) for k in range(3)], -1).reshape(-1, 24, 3)
    target = {'pose': pose, 'joint': joint, 'global_pose': global_pose,
              'unprojected_pose': unprojected_pose, 'avatar': torch.tensor(ref, dtype=torch.float32),
              'avatar_joint_ids': ids, 'timestamps': inputs['timestamps'], 'fps': 30,
              'metadata': {'source': str(capture), 'humanpose_sha256': file_hash(cache),
                           'type': 'HumanPose retargeted SMPL pseudo-labels; same-sequence smoke test',
                           'rest_calibration': calibration, 'shape': 'neutral, matching trained model',
                           'decoder_fixed_local_joints': ignored, 'translation_supervised': False}}
    output.mkdir(parents=True, exist_ok=True)
    torch.save(target, output/'targets.pt')
    render(ref, joint.numpy(), output/'pseudo_label_check.mp4', 'HumanPose reference / SMPL pseudo-label',
           legend='HumanPose (blue) / SMPL pseudo-label (red)')
    return target, report


def pose_forward(model, imu):
    lengths = [imu.shape[1]]
    joints = model.joints(imu, lengths)
    reduced = model.pose(torch.cat((joints, imu), dim=-1), lengths)
    pose = model._reduced_global_to_full(reduced)
    _, fk = model.bodymodel.forward_kinematics(pose)
    return joints, reduced, pose, fk


def metrics(pose, joint, target):
    ids = target['avatar_joint_ids']
    relative = joint - joint[:, :1]
    gt = target['joint'].to(joint)
    ref = target['avatar'].to(joint)
    diff = pose[:, joint_set.reduced].transpose(-1, -2) @ target['pose'].to(pose)[:, joint_set.reduced]
    angles = torch.acos(((diff.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1))
    return {'pseudo_smpl_mpjpe_cm': float((relative-(gt-gt[:, :1])).norm(dim=-1).mean()*100),
            'avatar_disagreement_cm': float((relative[:, ids]-(ref-ref[:, :1])[:, ids]).norm(dim=-1).mean()*100),
            'pseudo_local_rotation_deg': float(angles.mean()*180/np.pi)}


def run_layout(name, entry, target, args):
    from mobileposer.utils.model_utils import load_model
    dest = args.output/name
    dest.mkdir(parents=True, exist_ok=True)
    original = Path(entry['checkpoint'])
    original_hash = file_hash(original)
    if original_hash != entry['checkpoint_sha256']:
        raise ValueError('Baseline checkpoint changed')
    data = torch.load(entry['input'], map_location='cpu', weights_only=False)
    if not torch.equal(data['timestamps'], target['timestamps']):
        raise ValueError('Target/IMU timestamp mismatch')
    acc, ori = data['acc'][0], data['ori'][0]
    imu = torch.cat(((acc/30).flatten(1), ori.flatten(1)), -1)[None].to(args.device)
    model = load_model(str(original)).to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.joints, model.pose):
        for p in module.parameters():
            p.requires_grad_(True)
    # Disable external dropout, but keep cuDNN LSTMs in training mode so
    # backward has reserve buffers. These LSTMs have internal dropout=0.
    model.eval()
    model.joints.joints.rnn.train()
    model.pose.pose.rnn.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    target_joints = target['joint'].to(args.device)[None].flatten(2)
    target_6d = math.rotation_matrix_to_r6d(target['global_pose'][:, joint_set.reduced].to(args.device)).reshape(1, len(acc), -1)
    rows = []
    best_state, best_score, best_step = None, float('inf'), 0
    with torch.no_grad():
        _, _, before_pose, before_joint = pose_forward(model, imu)
        before = metrics(before_pose, before_joint, target)
        before_joint = before_joint.cpu()
    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            joints, reduced, _, fk = pose_forward(model, imu)
            joint_loss = (joints-target_joints).square().mean()
            rotation_loss = (reduced-target_6d).square().mean()
            fk_loss = (fk-target['joint'].to(args.device)).square().mean()
            loss = joint_loss + rotation_loss + fk_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                joints, reduced, pose, fk = pose_forward(model, imu)
                result = metrics(pose, fk, target)
                result['loss'] = float((joints-target_joints).square().mean()
                                       +(reduced-target_6d).square().mean()
                                       +(fk-target['joint'].to(args.device)).square().mean())
            rows.append(dict(step=step, **result))
            if result['pseudo_smpl_mpjpe_cm'] < best_score:
                best_score, best_step = result['pseudo_smpl_mpjpe_cm'], step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f'{name} step={step} loss={result["loss"]:.6f} pseudo={result["pseudo_smpl_mpjpe_cm"]:.2f}cm avatar={result["avatar_disagreement_cm"]:.2f}cm', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    # Verify frozen weights and save a directly loadable combined checkpoint.
    base = torch.load(original, map_location='cpu', weights_only=False)
    for key, value in best_state.items():
        if not key.startswith(('joints.', 'pose.')):
            if not torch.equal(value, base[key].cpu()):
                raise AssertionError(f'Frozen parameter changed: {key}')
    torch.save(best_state, dest/'model_finetuned.pth')
    with torch.no_grad():
        model.reset()
        pose, network_joints, tran, contact = model.forward_offline(imu, [len(acc)])
        _, joint = model.bodymodel.forward_kinematics(pose, tran=tran)
        after = metrics(pose, joint, target)
    torch.save({'pose': pose.cpu(), 'joint': joint.cpu(), 'tran': tran.cpu(),
                'network_joints': network_joints.cpu(), 'contact': contact.cpu(), 'fps': 30,
                'timestamps': target['timestamps'], 'experiment': 'same-sequence overfit'}, dest/'prediction.pt')
    render(target['avatar'].numpy(), joint.cpu().numpy(), dest/'after.mp4', name+' / AFTER same-sequence overfit')
    render(target['avatar'].numpy(), before_joint.numpy(), dest/'before.mp4', name+' / BEFORE')
    side_by_side(dest/'before.mp4', dest/'after.mp4', dest/'before_after.mp4')
    assert file_hash(original) == original_hash
    result = {'layout': name, 'before': before, 'after': after, 'best_step': best_step,
              'steps': args.steps, 'learning_rate': args.lr, 'seed': args.seed,
              'selection': 'lowest training-sequence pseudo-SMPL MPJPE; no held-out validation',
              'original_checkpoint': str(original), 'original_sha256': original_hash,
              'trainable_modules': ['joints', 'pose'], 'frozen_modules': ['velocity', 'foot_contact'],
              'train_and_test_same_sequence': True}
    (dest/'metrics.json').write_text(json.dumps(result, indent=2))
    with (dest/'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    del model
    torch.cuda.empty_cache()
    return result


def side_by_side(before, after, output):
    import cv2
    a, b = cv2.VideoCapture(str(before)), cv2.VideoCapture(str(after))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 15, (1800, 525))
    if not writer.isOpened():
        raise RuntimeError('Cannot create comparison video')
    try:
        while True:
            ok_a, fa = a.read(); ok_b, fb = b.read()
            if ok_a != ok_b:
                raise ValueError('Before/after video length mismatch')
            if not ok_a:
                break
            writer.write(np.concatenate([cv2.resize(x, (900, 525)) for x in (fa, fb)], axis=1))
    finally:
        a.release(); b.release(); writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=paths.root_dir/'results/ours1_surface_motionjson')
    parser.add_argument('--output', type=Path, default=paths.root_dir/'results/ours1_overfit_smoke')
    parser.add_argument('--layouts', nargs='+')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval-every', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.lr <= 0:
        parser.error('steps, eval-every and lr must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty; choose a new directory to preserve previous experiments')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.json').write_text(json.dumps({
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_report_sha256': file_hash(args.source/'report.json'),
        'script_sha256': file_hash(Path(__file__)),
        'torch_version': torch.__version__,
    }, indent=2))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model_config.device = torch.device(args.device)
    target, report = make_targets(args.source, args.output)
    results = []
    for name, entry in report['layouts'].items():
        if args.layouts and name not in args.layouts:
            continue
        if entry['status'] != 'inferred':
            print(f'Skip {name}: no physical IMU input', flush=True); continue
        torch.manual_seed(args.seed)
        results.append(run_layout(name, entry, target, args))
        (args.output/'summary.json').write_text(json.dumps(results, indent=2))
    if not results:
        raise ValueError('No requested layout has physical IMU input')
    lines = ['# ours/1 same-sequence overfit smoke test', '',
             '训练与测试为同一动作；无独立验证集，指标只检验过拟合链路。蓝色为 motion.json HumanPose，红色为预测。', '',
             '微调 joints + poser，冻结 velocity/contact。标签由当前 HumanPose rest-pose 校准后转为 SMPL，固定不可预测关节并采用中性体型。', '',
             '[伪标签检查视频](pseudo_label_check.mp4)：蓝色 HumanPose，红色 SMPL 伪标签。', '',
             '|layout|SMPL伪标签 MPJPE 前→后(cm)|Avatar差异 前→后(cm)|视频|', '|---|---:|---:|---|']
    for r in results:
        a,b = r['before'],r['after']; name=r['layout']
        lines.append(f'|{name}|{a["pseudo_smpl_mpjpe_cm"]:.2f} → {b["pseudo_smpl_mpjpe_cm"]:.2f}|{a["avatar_disagreement_cm"]:.2f} → {b["avatar_disagreement_cm"]:.2f}|[前后并排]({name}/before_after.mp4)|')
    lines += ['', '使用整段序列训练，关闭 dropout；以该训练序列的伪标签 MPJPE 选择 checkpoint。',
              '全局平移和接触没有监督：权重虽冻结，其预测仍可能随上游 joints 改变，当前视频和指标均根节点对齐。',
              '每组目录保存 model_finetuned.pth、prediction.pt、history.csv、metrics.json 和视频。原始 checkpoint 未修改。', '',
              '复现：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m mobileposer.smoke_finetune_ours --output results/ours1_overfit_smoke_new`', '']
    (args.output/'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
