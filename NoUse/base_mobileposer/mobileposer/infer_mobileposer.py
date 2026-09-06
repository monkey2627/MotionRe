#!/usr/bin/env python3
"""
MobilePoser inference on AMASS synthetic IMU data for per-body-part error evaluation.

For each sensor combo, runs the pre-trained model on AMASS sequences and saves:
  {pred: List[Tensor[T,24,3,3]], gt: List[Tensor[T,24,3,3]], combo: List[int]}

Usage:
  cd code/base_mobileposer
  conda activate mobileposer

  # All evaluation combos (slow):
  python -m mobileposer.infer_mobileposer --model checkpoints/weights.pth --combos all

  # Specific combos:
  python -m mobileposer.infer_mobileposer --model checkpoints/weights.pth \
      --combos lw_rw_h lp_rp_h all

  # Quick smoke test (50 sequences):
  python -m mobileposer.infer_mobileposer --model checkpoints/weights.pth \
      --combos all lw_rw_h --max-seq 50
"""
import os
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from mobileposer.config import paths, amass as amass_cfg, eval_combos as eval_combos_cfg
from mobileposer.utils.model_utils import load_model


# Whitelist: only load these AMASS subsets (skips DIP, TotalCapture, IMUPoser files)
_AMASS_NAMES = set(n.lower() for n in [
    'ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU',
    'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D',
    'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU',
    'SSM_synced', 'TCD_handMocap', 'Transitions_mocap',
])


def load_amass_sequences(data_dir: Path, max_seq: int = None):
    """
    Yields (acc [T,5,3], ori [T,5,3,3], pose_gt [T,24,3,3]) for each AMASS sequence.
    acc is already divided by acc_scale=30; ori is raw rotation matrices.
    pose_gt is local rotation matrices from the raw .pt file.
    """
    sequences = []
    pt_files = sorted(data_dir.glob('*.pt'))
    for pt_file in pt_files:
        if pt_file.stem.lower() not in _AMASS_NAMES:
            continue
        try:
            data = torch.load(pt_file, map_location='cpu')
        except Exception as e:
            print(f"  Warning: could not load {pt_file.name}: {e}")
            continue
        accs  = data['acc']   # list of [T, 6, 3]
        oris  = data['ori']   # list of [T, 6, 3, 3]
        poses = data['pose']  # list of [T, 24, 3, 3] — local rotation matrices

        for acc, ori, pose in zip(accs, oris, poses):
            sequences.append((
                acc[:, :5].float() / amass_cfg.acc_scale,  # [T, 5, 3]
                ori[:, :5].float(),                         # [T, 5, 3, 3]
                pose.float(),                               # [T, 24, 3, 3]
            ))
            if max_seq is not None and len(sequences) >= max_seq:
                return sequences
    return sequences


def apply_combo(acc: torch.Tensor, ori: torch.Tensor, combo_ids: list) -> torch.Tensor:
    """Zero-mask all slots except combo_ids, return [T, 60] model input."""
    masked_acc = torch.zeros_like(acc)  # [T, 5, 3]
    masked_ori = torch.zeros_like(ori)  # [T, 5, 3, 3]
    masked_acc[:, combo_ids] = acc[:, combo_ids]
    masked_ori[:, combo_ids] = ori[:, combo_ids]
    # Flatten: [T,5,3] → [T,15]; [T,5,3,3] → [T,45]; cat → [T,60]
    return torch.cat([masked_acc.flatten(1), masked_ori.flatten(1)], dim=1)


@torch.no_grad()
def run_combo(model, sequences, combo_ids: list, device: torch.device):
    """Run model on all sequences for one combo. Returns (pred_list, gt_list)."""
    model.eval()
    preds, gts = [], []
    for acc, ori, pose_gt in tqdm(sequences, desc='  sequences', leave=False):
        model.reset()
        x = apply_combo(acc, ori, combo_ids).to(device)  # [T, 60]
        pose_pred, _, _, _ = model.forward_offline(x.unsqueeze(0), [x.shape[0]])
        preds.append(pose_pred.cpu())  # [T, 24, 3, 3]
        gts.append(pose_gt.cpu())      # [T, 24, 3, 3]
    return preds, gts


def main():
    parser = argparse.ArgumentParser(description='MobilePoser per-combo AMASS inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to weights.pth (e.g. checkpoints/weights.pth)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Dir with processed AMASS .pt files (default: data/processed_datasets/)')
    parser.add_argument('--output-dir', type=str, default='results/mobileposer',
                        help='Output directory for per-combo .pt files')
    parser.add_argument('--combos', nargs='+', default=['all'],
                        help=('Combo names to evaluate, or "all" for every combo in eval_combos. '
                              'E.g.: --combos lw_rw_h lp_rp_h all'))
    parser.add_argument('--n-sensors', type=int, default=None,
                        help='Filter: only combos with exactly N sensors (1-5)')
    parser.add_argument('--max-seq', type=int, default=None,
                        help='Max AMASS sequences to load (useful for quick tests)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else paths.processed_datasets
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which combos to run
    all_combos = eval_combos_cfg.all_combos
    if args.combos == ['all']:
        selected = all_combos
    else:
        selected = {}
        for name in args.combos:
            if name == 'all':
                selected.update(all_combos)
            elif name in all_combos:
                selected[name] = all_combos[name]
            else:
                parser.error(f"Unknown combo: {name}. Available: {list(all_combos.keys())}")

    if args.n_sensors is not None:
        selected = {k: v for k, v in selected.items() if len(v) == args.n_sensors}

    print(f"Combos to evaluate ({len(selected)}): {list(selected.keys())}")

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model} (device={device})...")
    model = load_model(args.model)
    model = model.to(device)

    # Load AMASS sequences
    print(f"Loading AMASS sequences from {data_dir} ...")
    sequences = load_amass_sequences(data_dir, args.max_seq)
    print(f"Loaded {len(sequences)} sequences")
    if len(sequences) == 0:
        raise RuntimeError(f"No AMASS .pt files found in {data_dir}. "
                           "Run: python -m mobileposer.process --dataset amass")

    # Run inference per combo
    for combo_name, combo_ids in selected.items():
        out_path = out_dir / f'mobileposer_{combo_name}.pt'
        if out_path.exists():
            print(f"[skip] {combo_name} — result already exists at {out_path}")
            continue

        print(f"\n[{combo_name}] {len(combo_ids)} sensors, slots={combo_ids}")
        preds, gts = run_combo(model, sequences, combo_ids, device)

        torch.save({'pred': preds, 'gt': gts, 'combo': combo_ids,
                    'combo_name': combo_name, 'n_sequences': len(sequences)}, out_path)
        print(f"  Saved → {out_path}")

    print('\nAll done.')


if __name__ == '__main__':
    main()
