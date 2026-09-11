"""
PNP – DIP-IMU evaluation.

DIP-IMU ori = real sensor measurements (with drift/noise).
Unlike the AMASS evaluation, NO synthetic noise is added here —
the data already contains real IMU noise.

PNP is fixed 6-sensor (lw, rw, lp, rp, hd, root). All 6 sensors
are available in DIP-IMU, so no masking is needed.

Run from code/PNP/:
    python evaluate_dip.py
    python evaluate_dip.py --min_frames 150 --max_seconds 30
"""

import sys, os, argparse
from pathlib import Path

_DIR  = Path(__file__).resolve().parent   # code/PNP/
_CODE = _DIR.parent                        # code/

# PNP's __init__ loads weights via relative path — must cd here first
os.chdir(str(_DIR))
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_CODE))

import numpy as np
import torch
import tqdm

from net import PNP
import articulate as art

sys.path.insert(0, str(_CODE / 'base_mobileposer'))
from mobileposer.config import datasets, paths

from drift_eval_common import angle_between_rotmats, LUMBAR_JOINTS

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

_GRAVITY = torch.tensor([0.0, -9.8, 0.0])   # world-frame gravity (SMPL Y-up)
_SMPL    = _DIR.parent / 'base_mobileposer' / 'mobileposer' / 'smpl' / 'basicmodel_m.pkl'

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


# ── angular velocity ──────────────────────────────────────────────────────────

def angular_velocity(ori: torch.Tensor) -> torch.Tensor:
    """Finite-difference angular velocity from ori [T, 6, 3, 3] → [T, 6, 3]."""
    T = ori.shape[0]
    w = torch.zeros_like(ori[..., 0])
    for t in range(T):
        if   t == 0:     dR = (ori[1]  - ori[0])  * FPS
        elif t == T - 1: dR = (ori[-1] - ori[-2]) * FPS
        else:            dR = (ori[t+1] - ori[t-1]) * (FPS / 2.0)
        Om = ori[t].transpose(-1, -2) @ dR
        w[t, :, 0] = Om[:, 2, 1]
        w[t, :, 1] = Om[:, 0, 2]
        w[t, :, 2] = Om[:, 1, 0]
    return w


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_one(model, acc, ori, gt_pose, gt_tran, device):
    T = acc.shape[0]
    # DIP-IMU acc was preprocessed with gravity removed (matching AMASS convention).
    # Add gravity back so PNP receives signal in real-IMU convention.
    acc_g = (acc + _GRAVITY).to(device)
    ori_d = ori.to(device)
    w     = angular_velocity(ori).to(device)

    model.rnn_initialize()
    pose_list, tran_list = [], []
    for t in range(T):
        p, tr = model.forward_frame(acc_g[t], w[t], ori_d[t])
        pose_list.append(p.detach().cpu())
        tran_list.append(tr.detach().cpu())

    pose_p = torch.stack(pose_list)   # [T, 24, 3, 3]
    tran_p = torch.stack(tran_list)   # [T, 3]

    rot_err  = angle_between_rotmats(pose_p, gt_pose).numpy()
    tran_err = ((tran_p - tran_p[:1]) - (gt_tran - gt_tran[:1])).norm(dim=-1).numpy()
    return rot_err, tran_err


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(sequences, model, device, max_frames, out_dir=DEFAULT_OUT) -> dict:
    _CKPT = os.path.join(out_dir, '.eval_ckpt_pnp_dip.npz')
    os.makedirs(out_dir, exist_ok=True)

    rot_sum  = np.zeros((max_frames, 24))
    tran_sum = np.zeros(max_frames)
    count    = np.zeros(max_frames)
    start_idx = 0

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rot_sum   = ck['rot_sum']
        tran_sum  = ck['tran_sum']
        count     = ck['count']
        start_idx = int(ck['seqs_done'])
        print(f'  [Resume] PNP/DIP: {start_idx}/{len(sequences)} seqs done')

    for idx, seq in enumerate(tqdm.tqdm(sequences, desc='  PNP/6s')):
        if idx < start_idx:
            continue
        T = min(seq['pose'].shape[0], max_frames)
        try:
            err_r, err_t = run_one(
                model, seq['acc'][:T], seq['ori'][:T],
                seq['pose'][:T], seq['tran'][:T], device)
            rot_sum[:T]  += err_r
            tran_sum[:T] += err_t
            count[:T]    += 1
        except Exception as e:
            print(f'    skip: {e}')
        np.savez(_CKPT, rot_sum=rot_sum, tran_sum=tran_sum, count=count, seqs_done=idx + 1)

    if os.path.exists(_CKPT):
        os.remove(_CKPT)

    c = np.maximum(count, 1)
    return {'6s': {
        'rot':    np.where(count[:, None] > 0, rot_sum  / c[:, None], 0.0),
        'tran':   np.where(count > 0,           tran_sum / c,           np.nan),
        'count':  count,
        'n':      6,
        'n_seqs': int(count[0]),
    }}


# ── output ────────────────────────────────────────────────────────────────────

def print_summary(results, max_frames):
    cps = [cp for cp in [10, 30, 60] if cp <= max_frames // FPS]
    seg_names = list(SEGS.keys())
    for cp in cps:
        fr = min(cp * FPS - 1, max_frames - 1)
        print(f'\n─── @{cp}s ─── (PNP / DIP-IMU real IMU)')
        hdr = f"{'Combo':<8} {'N':>2}  " + ''.join(f"{s.split('(')[0]:>10}" for s in seg_names) + f"  {'Lumbar5':>8}  {'Tran(m)':>8}"
        print(hdr); print('─' * len(hdr))
        for cname, res in results.items():
            vals = [float(res['rot'][fr, j].mean()) for j in SEGS.values()]
            l5   = float(res['rot'][fr, LUMBAR_JOINTS].mean())
            tr   = float(res['tran'][fr]) if not np.isnan(res['tran'][fr]) else float('nan')
            line = f"{cname:<8} {res['n']:>2}  " + ''.join(f"{v:>10.2f}" for v in vals)
            tr_s = f"{tr:>8.3f}" if not np.isnan(tr) else "     N/A"
            print(f"{line}  {l5:>8.2f}  {tr_s}")


def save_npz(results, out_dir: Path, max_frames):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = {'combos': list(results.keys()), 'fps': FPS, 'max_frames': max_frames}
    for cname, res in results.items():
        d[f'{cname}_rot']  = res['rot']
        d[f'{cname}_tran'] = res['tran']
    np.savez(str(out_dir / 'pnp_dip.npz'), **d)
    print(f'\nSaved: {out_dir / "pnp_dip.npz"}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_frames',    type=int, default=300)
    p.add_argument('--max_seconds',   type=int, default=60)
    p.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',       default=DEFAULT_OUT)
    p.add_argument('--no_video',      action='store_true')
    p.add_argument('--video_seconds', type=int, default=30)
    p.add_argument('--video_fps',     type=int, default=10)
    args = p.parse_args()

    max_frames = args.max_seconds * FPS
    sequences  = load_dip(args.min_frames)
    if not sequences:
        print('No sequences found.'); return

    print('Loading PNP (weights auto-loaded from data/weights/PNP/weights.pt) ...')
    weights_path = _DIR / 'data' / 'weights' / 'PNP' / 'weights.pt'
    if not weights_path.exists():
        print(f'ERROR: weights not found at {weights_path}')
        return
    model = PNP().eval().to(args.device)
    print(f'  Loaded: {weights_path}')

    print(f'Running {len(sequences)} DIP-IMU sequences ...')
    results = evaluate(sequences, model, args.device, max_frames, out_dir=args.out_dir)
    print_summary(results, max_frames)
    save_npz(results, Path(args.out_dir), max_frames)
    import sys as _sys
    _sys.path.insert(0, str(_DIR.parent))
    from benchmarks.standard_results import write_standard_result
    result = results['6s']
    write_standard_result(
        Path(args.out_dir), 'pnp', 'dip', result['rot'], result['count'],
        FPS, 6, [0, 1, 2, 3, 4, 5], result['tran'],
    )

    if not args.no_video:
        from benchmarks.video import render_comparison_video
        from drift_eval_common import SENSOR_TO_JOINT
        bodymodel = art.model.ParametricModel(str(_SMPL))
        for index, seq in enumerate(sequences):
            length = min(seq['pose'].shape[0], args.video_seconds * FPS)
            acc_g = (seq['acc'][:length] + _GRAVITY).to(args.device)
            ori_d = seq['ori'][:length].to(args.device)
            w     = angular_velocity(seq['ori'][:length]).to(args.device)
            try:
                model.rnn_initialize()
                pose_list, tran_list = [], []
                with torch.no_grad():
                    for t in range(length):
                        p_t, tr_t = model.forward_frame(acc_g[t], w[t], ori_d[t])
                        pose_list.append(p_t.detach().cpu())
                        tran_list.append(tr_t.detach().cpu())
                pose_pred = torch.stack(pose_list)
                tran_pred = torch.stack(tran_list)
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
                fk_joints=fk_joints.numpy(), method='PNP', combo='6s',
                sequence=seq, fps=FPS, out_dir=Path(args.out_dir),
                seq_idx=index, max_seconds=args.video_seconds,
                render_fps=args.video_fps, method_errors=errors, fk_errors=fk_errors,
            )


if __name__ == '__main__':
    main()
