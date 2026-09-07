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
NO_HEAD_COMBOS = {k: v for k, v in amass.combos.items() if 4 not in v}

DEFAULT_OUT = str(_CODE.parent / 'r' / 'dip_results')


# ── data ──────────────────────────────────────────────────────────────────────

def load_dip(min_frames: int) -> list:
    p = paths.processed_datasets / 'eval' / 'dip_test.pt'
    data = torch.load(str(p), map_location='cpu')
    seqs = []
    for acc, ori, pose, tran in zip(data['acc'], data['ori'], data['pose'], data['tran']):
        if pose.shape[0] >= min_frames:
            seqs.append({'acc': acc.float(), 'ori': ori.float(),
                         'pose': pose.float(), 'tran': tran.float()})
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

def evaluate(sequences, model, device, combos, max_frames) -> dict:
    results = {}
    for cname, cidx in combos.items():
        rot_sum  = np.zeros((max_frames, 24))
        tran_sum = np.zeros(max_frames)
        count    = np.zeros(max_frames)
        for seq in tqdm.tqdm(sequences, desc=f'  {cname}', leave=False):
            T = min(seq['pose'].shape[0], max_frames)
            imu = prepare_imu(seq['acc'][:T], seq['ori'][:T], cidx)
            try:
                err_r, err_t = run_one(model, imu, seq['pose'][:T], seq['tran'][:T], device)
                rot_sum[:T]  += err_r
                tran_sum[:T] += err_t
                count[:T]    += 1
            except Exception as e:
                print(f'    skip ({cname}): {e}')
        c = np.maximum(count, 1)
        results[cname] = {
            'rot':    np.where(count[:, None] > 0, rot_sum  / c[:, None], 0.0),
            'tran':   np.where(count > 0,           tran_sum / c,           np.nan),
            'n':      len(cidx),
            'n_seqs': int(count[0]),
        }
    return results


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
    p.add_argument('--model',       required=True, help='Path to weights.pth')
    p.add_argument('--min_frames',  type=int, default=300)
    p.add_argument('--max_seconds', type=int, default=60)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',     default=DEFAULT_OUT)
    args = p.parse_args()

    max_frames = args.max_seconds * FPS
    sequences  = load_dip(args.min_frames)
    if not sequences:
        print('No sequences found.'); return

    print(f'Loading MobilePoser from {args.model} ...')
    model = load_model(args.model, device=args.device)
    model.eval()

    print(f'Running {len(sequences)} DIP-IMU sequences ...')
    results = evaluate(sequences, model, args.device, NO_HEAD_COMBOS, max_frames)
    print_summary(results, max_frames)
    save_npz(results, Path(args.out_dir), max_frames)


if __name__ == '__main__':
    main()
