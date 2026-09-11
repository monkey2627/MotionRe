"""
MobilePoser – DIP-IMU evaluation.

DIP-IMU ori = real sensor measurements (with drift/noise).
Uses the same combo masking as AMASS evaluation but on real IMU data.

Run from code/base_mobileposer/:
    python -m mobileposer.evaluate_dip --model checkpoints/weights.pth
    python -m mobileposer.evaluate_dip --model checkpoints/weights.pth --min_frames 150
"""

import sys, os, argparse
from pathlib import Path

import numpy as np
import torch
import tqdm

_DIR  = Path(__file__).resolve().parent           # mobileposer/
_BASE = _DIR.parent                                # base_mobileposer/
_CODE = _BASE.parent                               # code/
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_CODE))

from mobileposer.config import amass, datasets, paths
from mobileposer.utils.model_utils import load_model
import mobileposer.articulate as art
from drift_eval_common import angle_between_rotmats, LUMBAR_JOINTS, SENSOR_TO_JOINT

FPS = int(datasets.fps)

SEGS = {
    'Lumbar(j3)':       [3],
    'Thoracic(j6,9)':   [6, 9],
    'Hip(j1,2)':        [1, 2],
    'Knee(j4,5)':       [4, 5],
    'UpperArm(j16,17)': [16, 17],
    'Forearm(j18,19)':  [18, 19],
}
LUMBAR_JOINTS = [1, 2, 3, 6, 9]
# Native MobilePoser uses all five optional wearable slots.  The pelvis is a
# reference/root signal in the processed data, not one of the model inputs.
FULL_COMBOS = {
    # Physical count includes the pelvis reference (slot 5), which is not a
    # learned input: 4/5/6 physical IMUs map to 3/4/5 model slots.
    'full_4s': [0, 1, 2],
    'full_5s': [0, 1, 2, 3],
    'full_6s': [0, 1, 2, 3, 4],
}
PHYSICAL_SENSOR_COUNTS = {'full_4s': 4, 'full_5s': 5, 'full_6s': 6}
PHYSICAL_SENSOR_SLOTS = {
    'full_4s': [0, 1, 2, 5],
    'full_5s': [0, 1, 2, 3, 5],
    'full_6s': [0, 1, 2, 3, 4, 5],
}

DEFAULT_OUT = str(_CODE.parent / 'r' / 'dip_results')


# ── data ──────────────────────────────────────────────────────────────────────

def load_dip(min_frames: int) -> list:
    p = paths.processed_datasets / 'eval' / 'dip_test.pt'
    data = torch.load(str(p), map_location='cpu')
    seqs = []
    for index, (acc, ori, pose, tran) in enumerate(zip(data['acc'], data['ori'], data['pose'], data['tran'])):
        if pose.shape[0] >= min_frames:
            seqs.append({'acc': acc.float(), 'ori': ori.float(),
                         'pose': pose.float(), 'tran': tran.float(),
                         'source': 'dip_test[{}]'.format(index), 'action': 'dip'})
    print(f'DIP-IMU test: {len(seqs)}/{len(data["pose"])} seqs >= {min_frames} frames')
    return seqs


# ── inference ─────────────────────────────────────────────────────────────────

def prepare_imu(acc, ori, combo_indices):
    """60-D input for MobilePoser (5 sensor slots, no root)."""
    acc5 = acc[:, :5] / amass.acc_scale
    ori5 = ori[:, :5]
    ca = torch.zeros_like(acc5); co = torch.zeros_like(ori5)
    ca[:, combo_indices] = acc5[:, combo_indices]
    co[:, combo_indices] = ori5[:, combo_indices]
    return torch.cat([ca.flatten(1), co.flatten(1)], dim=1)


@torch.no_grad()
def run_one(model, imu, gt_pose, gt_tran, device):
    model.reset()
    imu_d = imu.to(device).unsqueeze(0)
    pose_p, _, tran_p, _ = model.forward_offline(imu_d, [imu_d.shape[1]])
    pose_p = pose_p.cpu(); tran_p = tran_p.cpu()
    T = gt_pose.shape[0]
    rot_err  = angle_between_rotmats(pose_p[:T], gt_pose)
    tran_err = (tran_p[:T] - (gt_tran - gt_tran[:1])).norm(dim=-1)
    return rot_err.numpy(), tran_err.numpy()


# ── evaluation loop ───────────────────────────────────────────────────────────

def _evaluate_combo(cname, cidx, sequences, model, device, max_frames, out_dir):
    """Evaluate one combo with per-sequence checkpointing for resume support."""
    _CKPT = os.path.join(out_dir, f'.eval_ckpt_{cname}.npz')

    rot_sum  = np.zeros((max_frames, 24))
    tran_sum = np.zeros(max_frames)
    count    = np.zeros(max_frames)
    start_idx = 0
    records = []

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rot_sum   = ck['rot_sum']
        tran_sum  = ck['tran_sum']
        count     = ck['count']
        start_idx = int(ck['seqs_done'])
        print(f'  [Resume] {cname}: {start_idx}/{len(sequences)} seqs done')

    for idx, seq in enumerate(tqdm.tqdm(sequences, desc=f'  {cname}', leave=False)):
        if idx < start_idx:
            continue
        T = min(seq['pose'].shape[0], max_frames)
        imu = prepare_imu(seq['acc'][:T], seq['ori'][:T], cidx)
        try:
            err_r, err_t = run_one(model, imu, seq['pose'][:T], seq['tran'][:T], device)
            rot_sum[:T]  += err_r
            tran_sum[:T] += err_t
            count[:T]    += 1
        except Exception as e:
            print(f'    skip ({cname}): {e}')
            from benchmarks.detailed_results import write_sequence_result
            records.append(write_sequence_result(
                Path(out_dir), idx, seq['source'], seq['action'], None, None, FPS,
                failure_reason=str(e), configuration=cname))
        else:
            from benchmarks.detailed_results import write_sequence_result
            records.append(write_sequence_result(
                Path(out_dir), idx, seq['source'], seq['action'], err_r, err_t, FPS,
                configuration=cname))
        np.savez(_CKPT, rot_sum=rot_sum, tran_sum=tran_sum, count=count, seqs_done=idx + 1)

    if os.path.exists(_CKPT):
        os.remove(_CKPT)

    c = np.maximum(count, 1)
    return {
        'rot':    np.where(count[:, None] > 0, rot_sum  / c[:, None], 0.0),
        'tran':   np.where(count > 0,           tran_sum / c,           np.nan),
        'count':  count,
        'n':      PHYSICAL_SENSOR_COUNTS[cname],
        'n_seqs': int(count[0]),
        'records': records,
    }


def evaluate(sequences, model, device, combos, max_frames, out_dir=None) -> dict:
    if out_dir is None:
        out_dir = DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    return {
        cname: _evaluate_combo(cname, cidx, sequences, model, device, max_frames, out_dir)
        for cname, cidx in combos.items()
    }


# ── output ────────────────────────────────────────────────────────────────────

def print_summary(results, max_frames):
    cps = [cp for cp in [10, 30, 60] if cp <= max_frames // FPS]
    seg_names = list(SEGS.keys())
    for cp in cps:
        fr = min(cp * FPS - 1, max_frames - 1)
        rows = []
        for cname, res in results.items():
            vals = [float(res['rot'][fr, j].mean()) for j in SEGS.values()]
            l5   = float(res['rot'][fr, LUMBAR_JOINTS].mean())
            tr   = float(res['tran'][fr]) if not np.isnan(res['tran'][fr]) else float('nan')
            rows.append((l5, cname, res['n'], vals, tr))
        print(f'\n─── @{cp}s ─── (MobilePoser / DIP-IMU real IMU)')
        hdr = f"{'Combo':<12} {'N':>2}  " + ''.join(f"{s.split('(')[0]:>10}" for s in seg_names) + f"  {'Lumbar5':>8}  {'Tran(m)':>8}"
        print(hdr); print('─' * len(hdr))
        for l5, cname, n, vals, tr in sorted(rows):
            line = f"{cname:<12} {n:>2}  " + ''.join(f"{v:>10.2f}" for v in vals)
            tr_s = f"{tr:>8.3f}" if not np.isnan(tr) else "     N/A"
            print(f"{line}  {l5:>8.2f}  {tr_s}")


def save_npz(results, out_dir: Path, max_frames):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = {'combos': list(results.keys()), 'fps': FPS, 'max_frames': max_frames}
    for cname, res in results.items():
        d[f'{cname}_rot']  = res['rot']
        d[f'{cname}_tran'] = res['tran']
    np.savez(str(out_dir / 'mobileposer_dip.npz'), **d)
    print(f'\nSaved: {out_dir / "mobileposer_dip.npz"}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',        required=True, help='Path to weights.pth')
    p.add_argument('--min_frames',   type=int, default=300)
    p.add_argument('--max_seconds',  type=int, default=60)
    p.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',      default=DEFAULT_OUT)
    p.add_argument('--combos', nargs='+', default=['full_6s'],
                   help='all or a subset of: {}'.format(', '.join(FULL_COMBOS)))
    p.add_argument('--no_video',     action='store_true')
    p.add_argument('--video_seconds', type=int, default=30)
    p.add_argument('--video_fps',    type=int, default=10)
    args = p.parse_args()

    max_frames = args.max_seconds * FPS
    sequences  = load_dip(args.min_frames)
    if not sequences:
        print('No sequences found.'); return

    print(f'Loading MobilePoser from {args.model} ...')
    model = load_model(args.model).to(args.device)
    model.eval()

    print(f'Running {len(sequences)} DIP-IMU sequences ...')
    if args.combos == ['all']:
        selected = FULL_COMBOS
    else:
        invalid = [combo for combo in args.combos if combo not in FULL_COMBOS]
        if invalid:
            p.error('Unknown combo(s): {}'.format(', '.join(invalid)))
        selected = {combo: FULL_COMBOS[combo] for combo in args.combos}
    results = evaluate(sequences, model, args.device, selected, max_frames, out_dir=args.out_dir)
    print_summary(results, max_frames)
    save_npz(results, Path(args.out_dir), max_frames)
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from benchmarks.standard_results import write_standard_result
    primary_combo = 'full_6s' if 'full_6s' in results else next(iter(results))
    result = results[primary_combo]
    write_standard_result(
        _Path(args.out_dir), 'mobileposer', 'dip', result['rot'], result['count'],
        FPS, PHYSICAL_SENSOR_COUNTS[primary_combo],
        PHYSICAL_SENSOR_SLOTS[primary_combo], result['tran'],
    )
    from benchmarks.detailed_results import write_detailed_index
    detailed_records = [record for res in results.values() for record in res['records']]
    write_detailed_index(_Path(args.out_dir), 'mobileposer', 'dip', FPS, detailed_records)

    if not args.no_video:
        from benchmarks.video import render_comparison_video
        bodymodel = art.model.ParametricModel(str(paths.smpl_file))
        for index, seq in enumerate(sequences):
            length = min(len(seq['pose']), args.video_seconds * FPS)
            imu = prepare_imu(seq['acc'][:length], seq['ori'][:length], FULL_COMBOS['full_5s'])
            try:
                with torch.no_grad():
                    model.reset()
                    imu_d = imu.to(args.device).unsqueeze(0)
                    pose_p, _, tran_p, _ = model.forward_offline(imu_d, [imu_d.shape[1]])
                pose_pred = pose_p.cpu()[:length]
                tran_pred = tran_p.cpu()[:length]
            except Exception as exc:
                print(f'\n  skip video seq {index}: {exc}')
                continue
            pose_fk = torch.eye(3).view(1, 1, 3, 3).expand(length, 24, 3, 3).clone()
            root_ori = seq['ori'][:length, 5]
            for s, j in enumerate(SENSOR_TO_JOINT):
                pose_fk[:, j] = root_ori.transpose(-1, -2) @ seq['ori'][:length, s]
            tran_pred = tran_pred - tran_pred[:1] + seq['tran'][:1]
            with torch.no_grad():
                _, gt_joints   = bodymodel.forward_kinematics(seq['pose'][:length], tran=seq['tran'][:length])
                _, pred_joints = bodymodel.forward_kinematics(pose_pred, tran=tran_pred)
                _, fk_joints   = bodymodel.forward_kinematics(pose_fk,   tran=seq['tran'][:length])
            errors    = angle_between_rotmats(pose_pred, seq['pose'][:length])[:, LUMBAR_JOINTS].mean(-1).numpy()
            fk_errors = angle_between_rotmats(pose_fk,   seq['pose'][:length])[:, LUMBAR_JOINTS].mean(-1).numpy()
            render_comparison_video(
                gt_joints=gt_joints.numpy(), method_joints=pred_joints.numpy(),
                fk_joints=fk_joints.numpy(), method='MobilePoser', combo='full_5s',
                sequence=seq, fps=FPS, out_dir=_Path(args.out_dir),
                seq_idx=index, max_seconds=args.video_seconds,
                render_fps=args.video_fps, method_errors=errors, fk_errors=fk_errors,
            )


if __name__ == '__main__':
    main()
