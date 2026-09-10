"""
SlimeVR (FK parent propagation) – DIP-IMU evaluation.

DIP-IMU ori = real sensor measurements (with drift/noise).
This is the fair evaluation: SlimeVR can no longer exploit near-perfect
synthesized orientations as it does on AMASS.

Run from code/slimevr/:
    python evaluate_dip.py
    python evaluate_dip.py --min_frames 150    # include seqs >= 5s
    python evaluate_dip.py --out_dir /custom/path
"""

import sys, os, argparse
from pathlib import Path

import numpy as np
import torch
import tqdm

_DIR   = Path(__file__).resolve().parent          # code/slimevr/
_CODE  = _DIR.parent                               # code/
_BASE  = _CODE / 'base_mobileposer'
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_CODE))

from mobileposer.config import amass, datasets, paths
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
# The pelvis is supplied implicitly by ``fk_predict``; five listed slots plus
# pelvis are SlimeVR's native full six-sensor input.
FULL_COMBOS = {'full_6s': [0, 1, 2, 3, 4]}

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


# ── FK inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def fk_predict(ori: torch.Tensor, combo_indices: list, bodymodel) -> torch.Tensor:
    T = ori.shape[0]
    parent = bodymodel.parent
    sensor_joints = {SENSOR_TO_JOINT[5]}
    for s in combo_indices:
        if s != 4:
            sensor_joints.add(SENSOR_TO_JOINT[s])
    R = torch.eye(3).view(1, 1, 3, 3).expand(T, 24, -1, -1).clone()
    R[:, SENSOR_TO_JOINT[5]] = ori[:, 5]
    for s in combo_indices:
        if s != 4:
            R[:, SENSOR_TO_JOINT[s]] = ori[:, s]
    for j in range(1, 24):
        if j not in sensor_joints:
            R[:, j] = R[:, parent[j]]
    return bodymodel.inverse_kinematics_R(R.view(T, -1)).view(T, 24, 3, 3)


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(sequences, bodymodel, combos, max_frames) -> dict:
    results = {}
    for cname, cidx in combos.items():
        rot_sum = np.zeros((max_frames, 24))
        count   = np.zeros(max_frames)
        for seq in tqdm.tqdm(sequences, desc=f'  {cname}', leave=False):
            T = min(seq['pose'].shape[0], max_frames)
            pred = fk_predict(seq['ori'][:T], cidx, bodymodel)
            err  = angle_between_rotmats(pred, seq['pose'][:T]).numpy()
            rot_sum[:T] += err
            count[:T]   += 1
        c = np.maximum(count, 1)
        rot_avg = np.where(count[:, None] > 0, rot_sum / c[:, None], 0.0)
        results[cname] = {'rot': rot_avg, 'count': count, 'n': 6, 'n_seqs': int(count[0])}
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
            rows.append((l5, cname, res['n'], vals))
        print(f'\n─── @{cp}s ─── (SlimeVR / DIP-IMU real IMU)')
        hdr = f"{'Combo':<12} {'N':>2}  " + ''.join(f"{s.split('(')[0]:>10}" for s in seg_names) + f"  {'Lumbar5':>8}"
        print(hdr); print('─' * len(hdr))
        for l5, cname, n, vals in sorted(rows):
            line = f"{cname:<12} {n:>2}  " + ''.join(f"{v:>10.2f}" for v in vals)
            print(f"{line}  {l5:>8.2f}")


def save_npz(results, out_dir: Path, max_frames):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = {'combos': list(results.keys()), 'fps': FPS, 'max_frames': max_frames}
    for cname, res in results.items():
        tran = np.full(max_frames, np.nan)
        d[f'{cname}_rot']  = res['rot']
        d[f'{cname}_tran'] = tran
    np.savez(str(out_dir / 'slimevr_dip.npz'), **d)
    print(f'\nSaved: {out_dir / "slimevr_dip.npz"}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_frames', type=int, default=300, help='Min seq length (frames)')
    p.add_argument('--max_seconds', type=int, default=60,  help='Evaluation window (s)')
    p.add_argument('--out_dir', default=DEFAULT_OUT)
    args = p.parse_args()

    max_frames = args.max_seconds * FPS
    sequences  = load_dip(args.min_frames)
    if not sequences:
        print('No sequences found.'); return

    bodymodel = art.model.ParametricModel(str(paths.smpl_file))
    print(f'\nRunning SlimeVR FK on {len(sequences)} DIP-IMU sequences ...')
    results = evaluate(sequences, bodymodel, FULL_COMBOS, max_frames)
    print_summary(results, max_frames)
    save_npz(results, Path(args.out_dir), max_frames)
    from benchmarks.standard_results import write_standard_result
    result = results['full_6s']
    write_standard_result(
        Path(args.out_dir), 'slimevr', 'dip', result['rot'], result['count'], FPS,
        6, [0, 1, 2, 3, 4, 5],
    )


if __name__ == '__main__':
    main()
