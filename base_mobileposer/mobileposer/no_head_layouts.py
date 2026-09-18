"""Generate joint-baseline or fixed-surface synthetic IMUs for no-head layouts."""

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mobileposer.articulate import math
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import datasets, paths
from mobileposer.surface_imu import DEFAULT_CONFIG, load_attachments, file_hash, surface_kinematics


TARGET_FPS = 30

# SMPL joint indices used by this codebase:
# 0 pelvis, 1/2 L/R hip, 3 spine1, 4/5 L/R knee, 7/8 L/R ankle,
# 16/17 L/R shoulder, 18/19 L/R elbow.
LAYOUTS = {
    "wrists_thighs_waist": {
        "labels": ["lw", "rw", "lt", "rt", "waist"],
        "joints": [18, 19, 1, 2, 3],
    },
    "wrists_shanks_waist": {
        "labels": ["lw", "rw", "ls", "rs", "waist"],
        "joints": [18, 19, 4, 5, 3],
    },
    "wrists_feet_waist": {
        "labels": ["lw", "rw", "lf", "rf", "waist"],
        "joints": [18, 19, 7, 8, 3],
    },
    "upperarms_thighs_waist": {
        "labels": ["lu", "ru", "lt", "rt", "waist"],
        "joints": [16, 17, 1, 2, 3],
    },
    "wrists_upperarms_waist": {
        "labels": ["lw", "rw", "lu", "ru", "waist"],
        "joints": [18, 19, 16, 17, 3],
    },
    "legs_waist": {
        "labels": ["lt", "rt", "ls", "rs", "waist"],
        "joints": [1, 2, 4, 5, 3],
    },
}


def _save_dataset(data, out_path):
    # An interrupted generation must not leave a file that looks complete.
    temporary = out_path.with_suffix('.pt.tmp')
    torch.save(data, temporary)
    os.replace(temporary, out_path)


def synthesis_metadata(mode, attachment_config):
    if mode not in {'joint', 'surface'}:
        raise ValueError(f'Unknown synthesis mode: {mode}')
    meta = {'mode': mode, 'fps': TARGET_FPS, 'acceleration': 'world_linear_m_s2',
            'orientation': 'bone_calibrated', 'smpl_sha256': file_hash(paths.smpl_file),
            'pose_blendshape': False, 'smooth_n': 4, 'root_slot': 'joint_0'}
    if mode == 'surface':
        meta['attachments'] = load_attachments(attachment_config)
        meta['attachment_sha256'] = file_hash(attachment_config)
    return meta


def validate_cached_dataset(out_path, expected, overwrite=False):
    if not out_path.exists() or overwrite:
        return False
    cached = torch.load(out_path, map_location='cpu', weights_only=False)
    actual = cached.get('synthesis')
    if actual != expected and not (actual is None and expected['mode'] == 'joint'):
        raise ValueError(f'{out_path}: incompatible synthesis cache; use a separate directory or --overwrite')
    print(f'[skip] compatible dataset: {out_path}')
    return True


def _synthesize(model, p, shape, tran, layout_name, synthesis, mesh_chunk_size):
    layout = LAYOUTS[layout_name]
    if synthesis['mode'] == 'surface':
        ori, joint, points = surface_kinematics(model, p, shape, tran, layout['labels'], synthesis['attachments'], mesh_chunk_size)
    else:
        r, joint = model.forward_kinematics(p, shape, tran)
        ids = layout['joints'] + [0]
        ori, points = r[:, ids], joint[:, ids]
    return ori, joint, _syn_acc(points)


def _syn_acc(points: torch.Tensor, smooth_n: int = 4) -> torch.Tensor:
    mid = smooth_n // 2
    scale_factor = TARGET_FPS ** 2
    acc = torch.stack(
        [(points[i] + points[i + 2] - 2 * points[i + 1]) * scale_factor for i in range(points.shape[0] - 2)]
    )
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [
                (points[i] + points[i + smooth_n * 2] - 2 * points[i + smooth_n])
                * scale_factor
                / smooth_n**2
                for i in range(points.shape[0] - smooth_n * 2)
            ]
        )
    return acc


def _foot_ground_probs(joint: torch.Tensor) -> torch.Tensor:
    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
    lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), (dist_lfeet < 0.008).int()))
    rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), (dist_rfeet < 0.008).int()))
    return torch.stack((lfoot_contact, rfoot_contact), dim=1)


def _load_amass_dataset(ds_name: str):
    data_pose, data_trans, data_beta, length = [], [], [], []
    pattern = os.path.join(paths.raw_amass, ds_name, "*/*_poses.npz")
    for npz_fname in tqdm(sorted(glob.glob(pattern)), desc=f"read {ds_name}", leave=False):
        try:
            cdata = np.load(npz_fname)
        except Exception:
            continue

        framerate = int(cdata["mocap_framerate"])
        if framerate not in [120, 60, 59]:
            continue

        step = max(1, round(framerate / TARGET_FPS))
        poses = cdata["poses"][::step].astype(np.float32)
        data_pose.extend(poses)
        data_trans.extend(cdata["trans"][::step].astype(np.float32))
        data_beta.append(cdata["betas"][:10])
        length.append(poses.shape[0])

    if not data_pose:
        return None

    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]
    pose = pose[:, :24].clone()

    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0]))
    )
    return pose, shape, tran, length


def generate_layout_dataset(layout_name: str, output_dir: Path, overwrite: bool = False,
                            mode="joint", attachment_config=DEFAULT_CONFIG, mesh_chunk_size=128,
                            dataset_names=None, max_sequences=None) -> None:
    if layout_name not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout_name}. Available: {sorted(LAYOUTS)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    body_model = ParametricModel(paths.smpl_file)

    synthesis = synthesis_metadata(mode, attachment_config)
    if max_sequences is not None:
        if max_sequences < 1:
            raise ValueError("max_sequences must be positive")
        synthesis["max_sequences"] = max_sequences
    for ds_name in (dataset_names or datasets.amass_datasets):
        out_path = output_dir / f"{ds_name}.pt"
        if validate_cached_dataset(out_path, synthesis, overwrite):
            continue

        loaded = _load_amass_dataset(ds_name)
        if loaded is None:
            print(f"[skip] {layout_name}/{ds_name}: no supported AMASS files")
            continue

        pose, shape, tran, length = loaded
        out_pose, out_shape, out_tran, out_joint, out_ori, out_acc, out_contact = [], [], [], [], [], [], []
        b = 0
        for i, seq_len in tqdm(list(enumerate(length)), desc=f"synth {layout_name}/{ds_name}", leave=False):
            if max_sequences is not None and len(out_pose) >= max_sequences:
                break
            seq_len = int(seq_len)
            if seq_len <= 12:
                b += seq_len
                continue

            p = math.axis_angle_to_rotation_matrix(pose[b : b + seq_len]).view(-1, 24, 3, 3)
            ori, joint, acc = _synthesize(body_model, p, shape[i], tran[b : b + seq_len], layout_name, synthesis, mesh_chunk_size)

            out_pose.append(p.clone())
            out_shape.append(shape[i].clone())
            out_tran.append(tran[b : b + seq_len].clone())
            out_joint.append(joint[:, :24].contiguous().clone())
            out_acc.append(acc)
            out_ori.append(ori)
            out_contact.append(_foot_ground_probs(joint).clone())
            b += seq_len

        if not out_pose:
            raise RuntimeError(f"No usable sequences: {ds_name}")
        _save_dataset(
            {
                "synthesis": synthesis,
                "layout": layout_name,
                "layout_labels": LAYOUTS[layout_name]["labels"],
                "layout_joints": LAYOUTS[layout_name]["joints"],
                "joint": out_joint,
                "pose": out_pose,
                "shape": out_shape,
                "tran": out_tran,
                "acc": out_acc,
                "ori": out_ori,
                "contact": out_contact,
            },
            out_path,
        )
        print(f"[done] {layout_name}/{ds_name}: {out_path}")


def generate_dip_layout_dataset(
    layout_name: str,
    output_dir: Path,
    split: str = "test",
    overwrite: bool = False,
    mode="joint", attachment_config=DEFAULT_CONFIG, mesh_chunk_size=128, max_sequences=None,
) -> None:
    """Generate no-head 5-IMU layout data from DIP-IMU GT poses.

    DIP-IMU has real IMUs only at the original MobilePoser sensor locations.
    The no-head layouts use different body locations, so this mirrors the AMASS
    no-head preprocessing: use DIP GT SMPL poses and synthesize orientation and
    acceleration at the configured joints or fixed surface attachments.
    """
    if layout_name not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout_name}. Available: {sorted(LAYOUTS)}")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    out_path = output_dir / f"dip_{split}.pt"
    synthesis = synthesis_metadata(mode, attachment_config)
    if max_sequences is not None:
        if max_sequences < 1:
            raise ValueError("max_sequences must be positive")
        synthesis["max_sequences"] = max_sequences
    if validate_cached_dataset(out_path, synthesis, overwrite):
        return

    subjects = (
        ["s_01", "s_02", "s_03", "s_04", "s_05", "s_06", "s_07", "s_08"]
        if split == "train"
        else ["s_09", "s_10"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    body_model = ParametricModel(paths.smpl_file)
    step = max(1, round(60 / TARGET_FPS))

    out_pose, out_shape, out_tran, out_joint, out_ori, out_acc, out_contact, metadata = [], [], [], [], [], [], [], []
    for subject_name in subjects:
        subject_dir = paths.raw_dip / subject_name
        if not subject_dir.exists():
            print(f"[skip] missing DIP subject: {subject_dir}")
            continue

        for motion_path in sorted(subject_dir.iterdir()):
            if max_sequences is not None and len(out_pose) >= max_sequences:
                break
            if motion_path.name.startswith("."):
                continue
            try:
                with motion_path.open("rb") as f:
                    data = pickle.load(f, encoding="latin1")
                pose = torch.from_numpy(data["gt"]).float()[6:-6:step].contiguous()
                if torch.isnan(pose).any():
                    print(f"[skip] DIP pose has NaN: {subject_name}/{motion_path.name}")
                    continue

                if len(pose) <= 12:
                    continue
                shape = torch.ones(10)
                tran = torch.zeros(pose.shape[0], 3)
                p = math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
                ori, joint, acc = _synthesize(body_model, p, shape, tran, layout_name, synthesis, mesh_chunk_size)

                out_pose.append(p.clone())
                out_shape.append(shape.clone())
                out_tran.append(tran.clone())
                out_joint.append(joint[:, :24].contiguous().clone())
                out_acc.append(acc)
                out_ori.append(ori)
                out_contact.append(_foot_ground_probs(joint).clone())
                metadata.append({"source": f"DIP_IMU/{subject_name}/{motion_path.name}"})
            except Exception as e:
                print(f"[skip] error processing DIP {subject_name}/{motion_path.name}: {e}")

    if not out_pose:
        raise RuntimeError(f"No usable DIP {split} sequences found under {paths.raw_dip}")

    _save_dataset(
        {
            "layout": layout_name,
            "synthesis": synthesis,
            "global_translation_available": False,
            "layout_labels": LAYOUTS[layout_name]["labels"],
            "layout_joints": LAYOUTS[layout_name]["joints"],
            "joint": out_joint,
            "pose": out_pose,
            "shape": out_shape,
            "tran": out_tran,
            "acc": out_acc,
            "ori": out_ori,
            "contact": out_contact,
            "metadata": metadata,
        },
        out_path,
    )
    print(f"[done] {layout_name}/dip_{split}: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate no-head 5-IMU AMASS datasets.")
    parser.add_argument("--layout", nargs="+", default=["all"], help=f"Layout names or all: {sorted(LAYOUTS)}")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--source", choices=["amass", "dip-test", "dip-train"], default="amass")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--imu-mode", choices=["joint", "surface"], default="joint")
    parser.add_argument("--attachment-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mesh-chunk-size", type=int, default=128)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()

    if args.output_root is None:
        suffix = "_surface" if args.imu_mode == "surface" else ""
        source_suffix = "processed" if args.source == "amass" else "dip_test"
        args.output_root = paths.root_dir / f"data/no_head_5imu{suffix}_{source_suffix}"
    layout_names = list(LAYOUTS) if args.layout == ["all"] else args.layout
    for layout_name in layout_names:
        output_dir = Path(args.output_root) / layout_name
        if args.source == "amass":
            generate_layout_dataset(layout_name, output_dir, overwrite=args.overwrite, mode=args.imu_mode, attachment_config=args.attachment_config, mesh_chunk_size=args.mesh_chunk_size, dataset_names=args.datasets, max_sequences=args.max_sequences)
        else:
            split = "test" if args.source == "dip-test" else "train"
            if split == "train":
                output_dir = output_dir / "train"
            generate_dip_layout_dataset(layout_name, output_dir, split=split, overwrite=args.overwrite, mode=args.imu_mode, attachment_config=args.attachment_config, mesh_chunk_size=args.mesh_chunk_size, max_sequences=args.max_sequences)


if __name__ == "__main__":
    main()
