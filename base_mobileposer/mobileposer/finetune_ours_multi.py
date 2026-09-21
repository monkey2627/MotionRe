"""Jointly overfit all discovered ours captures; render every usable capture/layout.

One checkpoint per layout, trained on all compatible recordings. No held-out
validation: both checkpoint selection and videos use the training recordings.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mobileposer.config import paths, model_config, joint_set
from mobileposer.articulate import math
from mobileposer.no_head_layouts import LAYOUTS
from mobileposer.surface_imu import DEFAULT_CONFIG, file_hash
from mobileposer.test_ours_surface import prepare, load_reference, render
from mobileposer.smoke_finetune_ours import make_targets, pose_forward, metrics, side_by_side


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def discover(root):
    """Nested motion packages are common (e.g. 4_臀桥/1/manifest.json)."""
    captures = []
    for manifest in sorted(root.rglob('manifest.json')):
        relative = manifest.parent.relative_to(root).as_posix()
        slug = relative.replace('/', '__')
        key = slug + '_' + hashlib.sha256(relative.encode()).hexdigest()[:8]
        captures.append((key, manifest.parent))
    if not captures:
        raise ValueError(f'No manifest.json under {root}')
    return captures


def prepare_all(args):
    catalog = {}
    for key, capture in discover(args.raw_root):
        dest = args.output/'captures'/key
        dest.mkdir(parents=True, exist_ok=True)
        print(f'Prepare {capture}', flush=True)
        # Fail visibly for malformed packages; never substitute identity/zero data.
        report = prepare(capture, dest)
        write_json(dest/'report.json', report)
        available = [name for name in args.layouts if report['layouts'][name]['status'] == 'prepared']
        catalog[key] = {'capture': str(capture), 'layouts': available,
                        'unavailable': {name: report['layouts'][name] for name in args.layouts if name not in available}}
        write_json(args.output/'catalog.json', catalog)
        if not available:
            print(f'Skip {key}: no requested layout has valid physical signals', flush=True)
            continue
        _, _, report['reference'] = load_reference(capture, dest, args.unity_editor)
        write_json(dest/'report.json', report)
        target, _ = make_targets(dest, dest)
        catalog[key]['target'] = str(dest/'targets.pt')
        catalog[key]['report'] = str(dest/'report.json')
        write_json(args.output/'catalog.json', catalog)
        del target
    return catalog


def load_sequences(catalog, name):
    sequences = []
    for key, item in catalog.items():
        if name not in item['layouts'] or 'target' not in item:
            continue
        report = json.loads(Path(item['report']).read_text())
        data = torch.load(report['layouts'][name]['input'], map_location='cpu', weights_only=False)
        target = torch.load(item['target'], map_location='cpu', weights_only=False)
        if data['layout_labels'] != LAYOUTS[name]['labels']:
            raise ValueError(f'{key}: wrong layout order')
        if not torch.equal(data['timestamps'], target['timestamps']):
            raise ValueError(f'{key}: input/label timestamps differ')
        acc, ori = data['acc'][0], data['ori'][0]
        imu = torch.cat(((acc/30).flatten(1), ori.flatten(1)), -1)[None]
        if not torch.isfinite(imu).all() or not torch.isfinite(target['pose']).all():
            raise ValueError(f'{key}: nonfinite input or labels')
        sequences.append((key, imu, target))
    return sequences


def frame_average(values):
    total = sum(n for n, _ in values)
    return {k: sum(n*m[k] for n, m in values)/total for k in values[0][1]}


@torch.no_grad()
def evaluate(model, sequences, device):
    model.eval()
    values = []
    for _, imu, target in sequences:
        _, _, pose, joint = pose_forward(model, imu.to(device))
        values.append((imu.shape[1], metrics(pose, joint, target)))
    return frame_average(values)


@torch.no_grad()
def export_predictions(model, sequences, dest, stage, device):
    model.eval()
    results = {}
    for key, imu, target in sequences:
        folder = dest/key
        folder.mkdir(parents=True, exist_ok=True)
        model.reset()
        pose, network_joints, tran, contact = model.forward_offline(imu.to(device), [imu.shape[1]])
        _, joint = model.bodymodel.forward_kinematics(pose, tran=tran)
        if not all(torch.isfinite(x).all() for x in (pose, joint, tran, contact)):
            raise ValueError(f'{key}: nonfinite {stage} prediction')
        result = metrics(pose, joint, target)
        torch.save({'pose': pose.cpu(), 'joint': joint.cpu(), 'tran': tran.cpu(),
                    'network_joints': network_joints.cpu(), 'contact': contact.cpu(),
                    'timestamps': target['timestamps'], 'fps': 30,
                    'metadata': target['metadata'], 'stage': stage}, folder/f'{stage}.pt')
        render(target['avatar'].numpy(), joint.cpu().numpy(), folder/f'{stage}.mp4', f'{dest.name} / {stage.upper()}')
        if stage == 'after':
            side_by_side(folder/'before.mp4', folder/'after.mp4', folder/'before_after.mp4')
        results[key] = result
    return results


def train_layout(args, name, sequences):
    from mobileposer.utils.model_utils import load_model
    dest = args.output/'layouts'/name
    dest.mkdir(parents=True, exist_ok=True)
    # Optional warm-start supports the previous smoke-test combined checkpoints.
    initial = (args.init_root/name/'model_finetuned.pth' if args.init_root else
               args.checkpoint_root/name/'1/base_model.pth')
    base_path = args.checkpoint_root/name/'1/base_model.pth'
    synthesis = json.loads((args.checkpoint_root/name/'synthesis.json').read_text())
    if synthesis['mode'] != 'surface' or synthesis['attachment_sha256'] != file_hash(DEFAULT_CONFIG):
        raise ValueError(f'{name}: incompatible surface checkpoint')
    initial_hash = file_hash(initial)
    if args.init_root:
        evidence = args.init_root/name/'metrics.json'
        if not evidence.exists():
            evidence = args.init_root/'layouts'/name/'metrics.json'
        info = json.loads(evidence.read_text())
        origin = info.get('base_checkpoint_sha256', info.get('original_sha256'))
        if origin != file_hash(base_path):
            raise ValueError(f'{name}: warm-start provenance does not match base checkpoint')
    model = load_model(str(initial)).to(args.device)
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
              if not k.startswith(('joints.', 'pose.'))}
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.joints, model.pose):
        for p in module.parameters():
            p.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    before = export_predictions(model, sequences, dest, 'before', args.device)
    best_score, best_epoch, history = float('inf'), 0, []
    # Each epoch visits every capture once, in a seeded shuffled order.
    for epoch in range(args.epochs+1):
        losses = []
        if epoch:
            model.eval()  # external dropout disabled for this smoke test
            model.joints.joints.rnn.train()  # cuDNN backward reserve buffers
            model.pose.pose.rnn.train()
            for index in torch.randperm(len(sequences)).tolist():
                _, imu, target = sequences[index]
                optimizer.zero_grad(set_to_none=True)
                joints, reduced, _, fk = pose_forward(model, imu.to(args.device))
                tj = target['joint'].to(args.device)
                tr = math.rotation_matrix_to_r6d(target['global_pose'][:, joint_set.reduced].to(args.device)).reshape(1, imu.shape[1], -1)
                loss = ((joints-tj[None].flatten(2)).square().mean()
                        +(reduced-tr).square().mean()+(fk-tj).square().mean())
                if not torch.isfinite(loss):
                    raise ValueError(f'{name}: nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            score = evaluate(model, sequences, args.device)
            history.append(dict(epoch=epoch, mean_update_loss=float(np.mean(losses)) if losses else None, **score))
            if score['pseudo_smpl_mpjpe_cm'] < best_score:
                best_score, best_epoch = score['pseudo_smpl_mpjpe_cm'], epoch
                torch.save({k: v.detach().cpu() for k,v in model.state_dict().items()}, dest/'model_finetuned.pth')
            write_json(dest/'history.json', history)
            print(f'{name} epoch={epoch}/{args.epochs} pseudo={score["pseudo_smpl_mpjpe_cm"]:.3f}cm avatar={score["avatar_disagreement_cm"]:.3f}cm', flush=True)
    state = torch.load(dest/'model_finetuned.pth', map_location='cpu', weights_only=False)
    for k, value in frozen.items():
        if not torch.equal(value, state[k]):
            raise ValueError(f'Frozen weight changed: {k}')
    model.load_state_dict(state)
    after = export_predictions(model, sequences, dest, 'after', args.device)
    if file_hash(initial) != initial_hash:
        raise ValueError('Initial checkpoint was modified')
    result = {'layout': name, 'initial_checkpoint': str(initial), 'initial_sha256': initial_hash,
              'base_checkpoint_sha256': file_hash(base_path), 'best_epoch': best_epoch,
              'selection': 'frame-weighted MPJPE on all training captures; no held-out set',
              'before': before, 'after': after}
    write_json(dest/'metrics.json', result)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize(output, results, catalog):
    lines = ['# 多动作联合微调 smoke test', '',
             '每个 layout 一个联合模型；训练和测试为相同动作集合。蓝色 HumanPose，红色预测。',
             '只微调 joints/poser，冻结 velocity/contact；视频根节点对齐，不评估全局位移。', '',
             '|布局|采集|伪标签 MPJPE 前→后(cm)|Avatar差异 前→后(cm)|全时长视频|', '|---|---|---:|---:|---|']
    rows = []
    for name, result in results.items():
        for key, before in result['before'].items():
            after = result['after'][key]
            video = f'layouts/{name}/{key}/before_after.mp4'
            lines.append(f'|{name}|{key}|{before["pseudo_smpl_mpjpe_cm"]:.2f} → {after["pseudo_smpl_mpjpe_cm"]:.2f}|{before["avatar_disagreement_cm"]:.2f} → {after["avatar_disagreement_cm"]:.2f}|[前后对比]({video})|')
            rows.append({'layout': name, 'capture': key, 'video': video,
                         **{'before_'+k:v for k,v in before.items()}, **{'after_'+k:v for k,v in after.items()}})
    lines += ['', '## 不可用布局', '', '缺少实体 IMU 或校准不可靠的组合不会用虚拟信号补齐，详见 catalog.json 及 captures/*/report.json。', '']
    for key, item in catalog.items():
        if item['unavailable']:
            lines.append(f'- {key}: {", ".join(item["unavailable"])}')
    (output/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    if rows:
        with (output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=paths.raw_ours)
    p.add_argument('--output', type=Path, default=paths.root_dir/'results/ours_multi_finetune')
    p.add_argument('--checkpoint-root', type=Path, default=paths.root_dir/'checkpoints/no_head_5imu_surface')
    p.add_argument('--init-root', type=Path, help='Optional directory containing <layout>/model_finetuned.pth')
    p.add_argument('--unity-editor', type=Path, default=Path('/home/duanyuhan/Unity/Hub/Editor/2022.3.13f1/Editor/Unity'))
    p.add_argument('--layouts', nargs='+', choices=list(LAYOUTS), default=list(LAYOUTS))
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--list-only', action='store_true', help='List recursive capture discovery without modifying files')
    args = p.parse_args()
    args.raw_root = args.raw_root.resolve(); args.output = args.output.resolve()
    captures = discover(args.raw_root)
    if args.list_only:
        for key, capture in captures: print(key, capture)
        return
    if args.epochs < 1 or args.eval_every < 1 or args.lr <= 0:
        p.error('epochs, eval-every and lr must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output must be empty: choose a new --output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'config.json', {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model_config.device = torch.device(args.device)
    catalog = prepare_all(args)
    results = {}
    for name in args.layouts:
        sequences = load_sequences(catalog, name)
        if not sequences:
            print(f'Skip {name}: no compatible captures', flush=True); continue
        torch.manual_seed(args.seed)
        results[name] = train_layout(args, name, sequences)
        write_json(args.output/'summary.json', results)
        summarize(args.output, results, catalog)
    if not results:
        raise ValueError('No usable layout: inspect catalog.json and per-capture sensor diagnostics')


if __name__ == '__main__':
    main()
