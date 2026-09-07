"""
IMUCoCo – DIP-IMU evaluation.

DIP-IMU ori = real sensor measurements (with drift/noise).

Run from code/IMUCoCo/:
    python evaluate_dip.py
    python evaluate_dip.py --min_frames 150 --max_seconds 30
"""

import sys, os, argparse
from pathlib import Path

_DIR  = Path(__file__).resolve().parent   # code/IMUCoCo/
_CODE = _DIR.parent                        # code/

# Must be in IMUCoCo's own directory for its relative-path imports
os.chdir(str(_DIR))
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_CODE))

import numpy as np
import torch
import tqdm

import articulate as art
from models.imucoco import IMUCoCo
from models.dtp import Poser
from utils import imu_config
import path_config

# resolve DIP test data path via base_mobileposer config
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

# Sensor vertex IDs on SMPL mesh: [lw, rw, lp, rp, hd, root]
SENSOR_VERTEX_IDS = [1962, 5431, 947, 4433, 412, 3021]

COMBOS = {
    'lp':     [2],
    'rp':     [3],
    'lw_lp':  [0, 2],
    'lw_rp':  [0, 3],
    'rw_lp':  [1, 2],
    'rw_rp':  [1, 3],
    '6s':     [0, 1, 2, 3, 4, 5],
}

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


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(device):
    vc_cat = torch.tensor(imu_config.vertex_coordinates_with_category).float().to(device)
    jc_cat = torch.tensor(imu_config.joint_coordinates_with_category).float().to(device)
    c_max  = torch.max(vc_cat[:, 1:], dim=0).values
    c_min  = torch.min(vc_cat[:, 1:], dim=0).values
    vc     = torch.tensor(imu_config.vertex_coordinates).float().to(device)

    imucoco = IMUCoCo(
        coordinate_origins=jc_cat, coordinate_max=c_max, coordinate_min=c_min,
        smpl_mesh_coordinates=vc_cat, n_hidden=128, n_kr_hidden=32,
        n_mfe_layers=2, n_jnm_layers=3, n_sce_freq=4, n_sce_emb=40,
        online_mode=False,
        joint_node_allocation_map=path_config.saved_imucoco_loss_map_path,
        joint_node_max_err_tolerance=-1,
    ).to(device)
    imucoco.load_state_dict(
        torch.load(path_config.saved_imucoco_checkpoint_path, map_location=device),
        strict=False)
    imucoco.freeze(); imucoco.eval()

    poser = Poser(joint_feature_dim=128, n_hidden=300, n_glb=40,
                  num_layer=3, n_total_devices=24, load_tran_module=True).to(device)
    poser.load_state_dict(
        torch.load(path_config.saved_hpe_checkpoint_path, map_location=device),
        strict=False)
    poser.eval()

    body_model = art.ParametricModel('smpl/SMPL_MALE.pkl', device='cpu')
    print(f'  IMUCoCo: {path_config.saved_imucoco_checkpoint_path}')
    print(f'  Poser  : {path_config.saved_hpe_checkpoint_path}')
    return imucoco, poser, body_model, vc


# ── inference ─────────────────────────────────────────────────────────────────

def _r6d(R):
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


@torch.no_grad()
def run_one(imucoco, poser, body_model, vc, acc, ori, gt_pose, gt_tran, cidx, device):
    T = acc.shape[0]
    vids   = [SENSOR_VERTEX_IDS[s] for s in cidx]
    coords = vc[vids]
    imucoco.set_current_device_coordinates(coords)
    imucoco.buffer_placement_codes_with_current_devices(parallel=False)

    ori_sel = ori[:, cidx]                              # [T, D, 3, 3]
    acc_sel = acc[:, cidx]                              # [T, D, 3]
    imu9    = torch.cat([_r6d(ori_sel), acc_sel], -1)   # [T, D, 9]
    imu9    = imu9.unsqueeze(0).to(device)

    feat_m  = imucoco.inference_time_forward_mesh(imu9)

    first_local = gt_pose[0:1].cpu()
    glb_0, _    = body_model.forward_kinematics(first_local, calc_mesh=False)
    glb_init    = _r6d(glb_0).to(device)

    _, pose_p, tran_p = poser.forward(
        x=feat_m,
        v_init=torch.zeros(1, 24, 3, device=device),
        glb_init=glb_init,
        seq_len=torch.tensor([T]),
        compute_tran='transpose',
    )
    pred_local = pose_p[0].cpu()
    rot_err    = angle_between_rotmats(pred_local, gt_pose).numpy()

    tp     = tran_p.cpu()
    gt_t   = gt_tran.cpu()
    tran_err = (tp - tp[:1] - (gt_t - gt_t[:1])).norm(dim=-1).numpy()
    return rot_err, tran_err


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(sequences, imucoco, poser, body_model, vc, combos, max_frames, device) -> dict:
    results = {}
    for cname, cidx in combos.items():
        rot_sum  = np.zeros((max_frames, 24))
        tran_sum = np.zeros(max_frames)
        count    = np.zeros(max_frames)
        for seq in tqdm.tqdm(sequences, desc=f'  {cname}', leave=False):
            T = min(seq['pose'].shape[0], max_frames)
            try:
                err_r, err_t = run_one(
                    imucoco, poser, body_model, vc,
                    seq['acc'][:T], seq['ori'][:T],
                    seq['pose'][:T], seq['tran'][:T],
                    cidx, device)
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
        print(f'\n─── @{cp}s ─── (IMUCoCo / DIP-IMU real IMU)')
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
    np.savez(str(out_dir / 'imucoco_dip.npz'), **d)
    print(f'\nSaved: {out_dir / "imucoco_dip.npz"}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_frames',  type=int, default=300)
    p.add_argument('--max_seconds', type=int, default=60)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',     default=DEFAULT_OUT)
    args = p.parse_args()

    max_frames = args.max_seconds * FPS
    sequences  = load_dip(args.min_frames)
    if not sequences:
        print('No sequences found.'); return

    print('Loading IMUCoCo models ...')
    imucoco, poser, body_model, vc = load_models(args.device)

    print(f'Running {len(sequences)} DIP-IMU sequences ...')
    results = evaluate(sequences, imucoco, poser, body_model, vc, COMBOS, max_frames, args.device)
    print_summary(results, max_frames)
    save_npz(results, Path(args.out_dir), max_frames)


if __name__ == '__main__':
    main()
