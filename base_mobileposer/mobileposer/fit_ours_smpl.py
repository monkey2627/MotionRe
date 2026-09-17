"""Fit SMPL poses to locally exported FBX skeleton animations.

This is a practical retargeting bridge for ``data/raw/ours`` captures:

1. Blender samples each FBX armature into per-frame world-space bone positions.
2. PyTorch optimizes SMPL pose, translation, and shape against those joints.
3. The result is saved in the same processed-dataset format as MobilePoser.

The fitted result is an estimated GT. Inspect it before using it for metrics or
finetuning.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from mobileposer.articulate import math
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths


_DEFAULT_BLENDER = Path("/home/duanyuhan/blender-4.2.0-linux-x64/blender")

_FBX_BONES = [
    "pelvis",
    "femur_l",
    "femur_r",
    "tibia_l",
    "tibia_r",
    "talus_l",
    "talus_r",
    "toes_l",
    "toes_r",
    "lumbar_body",
    "thorax",
    "head",
    "humerus_l",
    "humerus_r",
    "ulna_l",
    "ulna_r",
    "hand_l",
    "hand_r",
]

# SMPL joint indices in this codebase:
# 0 pelvis, 1/2 L/R hip, 3 spine1, 4/5 L/R knee, 7/8 L/R ankle,
# 9 spine3, 10/11 L/R foot, 15 head, 16/17 L/R shoulder,
# 18/19 L/R elbow, 20/21 L/R wrist.
_JOINT_MAP = [
    (0, "pelvis", 5.0),
    (1, "femur_l", 2.0),
    (2, "femur_r", 2.0),
    (3, "lumbar_body", 1.5),
    (4, "tibia_l", 3.0),
    (5, "tibia_r", 3.0),
    (7, "talus_l", 2.5),
    (8, "talus_r", 2.5),
    (10, "toes_l", 1.5),
    (11, "toes_r", 1.5),
    (9, "thorax", 1.5),
    (15, "head", 1.2),
    (16, "humerus_l", 2.0),
    (17, "humerus_r", 2.0),
    (18, "ulna_l", 2.0),
    (19, "ulna_r", 2.0),
    (20, "hand_l", 2.0),
    (21, "hand_r", 2.0),
    (22, "hand_l", 0.5),
    (23, "hand_r", 0.5),
]

_BONE_DIR_MAP = [
    (1, 4, "femur_l", "tibia_l", 2.0),
    (4, 7, "tibia_l", "talus_l", 3.0),
    (7, 10, "talus_l", "toes_l", 1.5),
    (2, 5, "femur_r", "tibia_r", 2.0),
    (5, 8, "tibia_r", "talus_r", 3.0),
    (8, 11, "talus_r", "toes_r", 1.5),
    (3, 9, "lumbar_body", "thorax", 1.5),
    (9, 15, "thorax", "head", 1.0),
    (16, 18, "humerus_l", "ulna_l", 2.5),
    (18, 20, "ulna_l", "hand_l", 3.0),
    (17, 19, "humerus_r", "ulna_r", 2.5),
    (19, 21, "ulna_r", "hand_r", 3.0),
]


def _find_blender(user_path: str | None) -> Path:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    candidates.append(_DEFAULT_BLENDER)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Blender executable not found. Pass --blender /path/to/blender."
    )


def _capture_dirs(raw_dir: Path) -> List[Path]:
    if (raw_dir / "manifest.json").exists():
        return [raw_dir]
    return sorted(p for p in raw_dir.iterdir() if (p / "manifest.json").exists())


def _capture_name(capture_dir: Path) -> str:
    return capture_dir.name if capture_dir.name else "capture"


def _manifest_path(capture_dir: Path, key: str) -> Path:
    manifest_path = capture_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return capture_dir / manifest[key]


def _write_blender_export_script(script_path: Path) -> None:
    script_path.write_text(
        r'''
import argparse
import json
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bones", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    wanted = args.bones.split(",")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.fbx(filepath=args.fbx)

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found in FBX")
    arm = armatures[0]
    if not (arm.animation_data and arm.animation_data.action):
        raise RuntimeError("No armature action found in FBX")

    action = arm.animation_data.action
    start = int(round(action.frame_range[0]))
    end = int(round(action.frame_range[1]))
    frames = list(range(start, end + 1))

    positions = np.full((len(frames), len(wanted), 3), np.nan, dtype=np.float32)
    rotations = np.full((len(frames), len(wanted), 4), np.nan, dtype=np.float32)
    present = []
    for bone_name in wanted:
        present.append(bone_name in arm.pose.bones)

    for frame_i, frame in enumerate(frames):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for bone_i, bone_name in enumerate(wanted):
            if bone_name not in arm.pose.bones:
                continue
            pose_bone = arm.pose.bones[bone_name]
            world_matrix = arm.matrix_world @ pose_bone.matrix
            positions[frame_i, bone_i] = tuple(world_matrix.to_translation())
            quat = world_matrix.to_quaternion()
            rotations[frame_i, bone_i] = (quat.w, quat.x, quat.y, quat.z)

    meta = {
        "armature": arm.name,
        "action": action.name,
        "frame_start": start,
        "frame_end": end,
        "fps": bpy.context.scene.render.fps,
        "bones": wanted,
        "present": present,
    }
    np.savez(args.out, positions=positions, rotations=rotations, meta=json.dumps(meta))


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )


def _export_fbx_joints(blender: Path, fbx_path: Path, out_npz: Path, overwrite: bool) -> None:
    if out_npz.exists() and not overwrite:
        return
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ours_fbx_export_") as tmpdir:
        script_path = Path(tmpdir) / "export_fbx_joints.py"
        _write_blender_export_script(script_path)
        cmd = [
            str(blender),
            "--background",
            "--python",
            str(script_path),
            "--",
            "--fbx",
            str(fbx_path),
            "--out",
            str(out_npz),
            "--bones",
            ",".join(_FBX_BONES),
        ]
        subprocess.run(cmd, check=True)


def _load_fbx_positions(npz_path: Path) -> Tuple[torch.Tensor, Dict]:
    data = np.load(npz_path, allow_pickle=False)
    positions = torch.from_numpy(data["positions"]).float()
    meta = json.loads(str(data["meta"]))

    # Blender imports this FBX as Z-up. MobilePoser/SMPL uses Y-up.
    # Keep left-right X, map vertical Z -> Y, and use -Y as depth.
    positions = torch.stack(
        (positions[..., 0], positions[..., 2], -positions[..., 1]), dim=-1
    )
    return positions, meta


def _target_from_fbx(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bone_to_idx = {name: idx for idx, name in enumerate(_FBX_BONES)}
    smpl_ids, targets, weights = [], [], []
    for smpl_id, bone_name, weight in _JOINT_MAP:
        smpl_ids.append(smpl_id)
        targets.append(positions[:, bone_to_idx[bone_name]])
        weights.append(weight)

    target = torch.stack(targets, dim=1)
    valid = torch.isfinite(target).all(dim=-1)
    target = torch.nan_to_num(target, nan=0.0)
    weight = torch.tensor(weights, dtype=torch.float32).view(1, -1) * valid.float()
    return torch.tensor(smpl_ids, dtype=torch.long), target, weight


def _bone_target_from_fbx(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bone_to_idx = {name: idx for idx, name in enumerate(_FBX_BONES)}
    parents, children, dirs, weights = [], [], [], []
    for parent, child, src_parent, src_child, weight in _BONE_DIR_MAP:
        vec = positions[:, bone_to_idx[src_child]] - positions[:, bone_to_idx[src_parent]]
        valid = torch.isfinite(vec).all(dim=-1) & (vec.norm(dim=-1) > 1e-6)
        dirs.append(torch.nn.functional.normalize(torch.nan_to_num(vec, nan=0.0), dim=-1))
        parents.append(parent)
        children.append(child)
        weights.append(weight)
    target_dirs = torch.stack(dirs, dim=1)
    weight = torch.tensor(weights, dtype=torch.float32).view(1, -1)
    valid = torch.isfinite(target_dirs).all(dim=-1)
    return (
        torch.tensor(parents, dtype=torch.long),
        torch.tensor(children, dtype=torch.long),
        target_dirs,
        weight * valid.float(),
    )


def _foot_ground_probs(joint: torch.Tensor) -> torch.Tensor:
    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
    lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), (dist_lfeet < 0.008).int()))
    rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), (dist_rfeet < 0.008).int()))
    return torch.stack((lfoot_contact, rfoot_contact), dim=1)


def _fit_sequence(
    target: torch.Tensor,
    smpl_ids: torch.Tensor,
    weights: torch.Tensor,
    bone_parents: torch.Tensor,
    bone_children: torch.Tensor,
    bone_dirs: torch.Tensor,
    bone_weights: torch.Tensor,
    *,
    device: torch.device,
    iterations: int,
    lr: float,
    optimize_betas: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    body_model = ParametricModel(paths.smpl_file, device=device)
    target = target.to(device)
    smpl_ids = smpl_ids.to(device)
    weights = weights.to(device)
    bone_parents = bone_parents.to(device)
    bone_children = bone_children.to(device)
    bone_dirs = bone_dirs.to(device)
    bone_weights = bone_weights.to(device)

    n_frames = target.shape[0]
    identity_r6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
    pose_r6d = identity_r6d.view(1, 1, 6).repeat(n_frames, 24, 1)
    pose_r6d = pose_r6d.clone().detach().requires_grad_(True)
    tran = target[:, 0].detach().clone().requires_grad_(True)
    betas = torch.zeros(10, device=device, requires_grad=optimize_betas)

    params = [pose_r6d, tran]
    if optimize_betas:
        params.append(betas)
    optimizer = torch.optim.Adam(params, lr=lr)

    prev_loss = None
    for _ in tqdm(range(iterations), desc="Fitting SMPL", leave=False):
        optimizer.zero_grad()
        pose = math.r6d_to_rotation_matrix(pose_r6d).view(n_frames, 24, 3, 3)
        _, joint = body_model.forward_kinematics(pose, betas, tran)
        pred = joint[:, smpl_ids]
        joint_loss = (((pred - target) ** 2).sum(dim=-1) * weights).sum() / weights.sum().clamp_min(1.0)

        pred_bones = joint[:, bone_children] - joint[:, bone_parents]
        pred_dirs = torch.nn.functional.normalize(pred_bones, dim=-1)
        bone_dir_loss = ((1.0 - (pred_dirs * bone_dirs).sum(dim=-1)) * bone_weights).sum() / bone_weights.sum().clamp_min(1.0)

        eye = torch.eye(3, device=device).view(1, 1, 3, 3)
        pose_prior = ((pose[:, 1:] - eye) ** 2).mean()
        root_prior = ((pose[:, :1] - eye) ** 2).mean() * 0.1
        smooth_pose = ((pose_r6d[1:] - pose_r6d[:-1]) ** 2).mean()
        smooth_tran = ((tran[1:] - tran[:-1]) ** 2).mean()
        beta_prior = (betas ** 2).mean()
        loss = (
            joint_loss
            + 0.05 * bone_dir_loss
            + 0.005 * pose_prior
            + 0.001 * root_prior
            + 0.05 * smooth_pose
            + 0.02 * smooth_tran
            + 0.001 * beta_prior
        )
        loss.backward()
        optimizer.step()
        prev_loss = loss.detach()

    with torch.no_grad():
        pose = math.r6d_to_rotation_matrix(pose_r6d).view(n_frames, 24, 3, 3)
        _, joint = body_model.forward_kinematics(pose, betas, tran)
        pred = joint[:, smpl_ids]
        err = torch.norm(pred - target, dim=-1)
        valid = weights > 0
        mean_err = err[valid].mean().item()
        max_err = err[valid].max().item()

    metrics = {
        "loss": float(prev_loss.item()) if prev_loss is not None else 0.0,
        "mean_joint_error_m": mean_err,
        "max_joint_error_m": max_err,
    }
    return (
        pose.detach().cpu(),
        tran.detach().cpu(),
        betas.detach().cpu(),
        joint.detach().cpu(),
        metrics,
    )


def _load_processed_ours(processed_ours: Path) -> Dict:
    if not processed_ours.exists():
        raise FileNotFoundError(
            f"{processed_ours} not found. Run: python -m mobileposer.process --dataset ours"
        )
    return torch.load(processed_ours, map_location="cpu")


def fit_ours(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.output)
    blender = _find_blender(args.blender)
    device = torch.device(args.device)

    processed = None if args.no_processed_source else _load_processed_ours(Path(args.processed_ours))
    captures = _capture_dirs(raw_dir)
    if processed is not None and len(captures) != len(processed["acc"]):
        print(
            f"Warning: {len(captures)} captures but processed ours has {len(processed['acc'])} sequences. "
            "They will be matched by sorted order.",
            file=sys.stderr,
        )

    poses, trans, shapes, joints, contacts, fit_meta = [], [], [], [], [], []
    for seq_idx, capture_dir in enumerate(captures):
        manifest_fbx = _manifest_path(capture_dir, "fbxFile")
        cache_npz = cache_dir / f"{_capture_name(capture_dir)}.fbx_joints.npz"
        print(f"Exporting FBX joints: {manifest_fbx}")
        _export_fbx_joints(blender, manifest_fbx, cache_npz, args.overwrite_cache)

        positions, fbx_meta = _load_fbx_positions(cache_npz)
        smpl_ids, target, weights = _target_from_fbx(positions)
        bone_parents, bone_children, bone_dirs, bone_weights = _bone_target_from_fbx(positions)
        imu_len = processed["acc"][seq_idx].shape[0] if processed is not None else target.shape[0]
        fit_len = min(imu_len, target.shape[0])
        target = target[:fit_len]
        weights = weights[:, :].expand(fit_len, -1).clone()
        bone_dirs = bone_dirs[:fit_len]
        bone_weights = bone_weights.expand(fit_len, -1).clone()

        print(f"Fitting sequence {seq_idx}: {fit_len} frames")
        pose, tran, shape, joint, metrics = _fit_sequence(
            target,
            smpl_ids,
            weights,
            bone_parents,
            bone_children,
            bone_dirs,
            bone_weights,
            device=device,
            iterations=args.iterations,
            lr=args.lr,
            optimize_betas=not args.fixed_shape,
        )

        poses.append(pose[:fit_len])
        trans.append(tran[:fit_len])
        shapes.append(shape)
        joints.append(joint[:fit_len])
        contacts.append(_foot_ground_probs(joint[:fit_len]))
        fit_meta.append(
            {
                "capture_dir": str(capture_dir),
                "fbx": str(manifest_fbx),
                "fbx_meta": fbx_meta,
                "fit": metrics,
                "joint_map": _JOINT_MAP,
                "coordinate_map": "smpl_xyz = (fbx_x, fbx_z, -fbx_y)",
            }
        )

    if processed is None:
        seq_count = len(poses)
        out = {}
        out["acc"] = [torch.zeros(p.shape[0], 6, 3) for p in poses]
        out["ori"] = [torch.eye(3).repeat(p.shape[0], 6, 1, 1) for p in poses]
        out["metadata"] = [{} for _ in poses]
    else:
        out = dict(processed)
        seq_count = min(len(poses), len(out["acc"]))
        for key in ("acc", "ori"):
            out[key] = [out[key][i][: poses[i].shape[0]] for i in range(seq_count)]
    out["pose"] = poses[:seq_count]
    out["tran"] = trans[:seq_count]
    out["shape"] = shapes[:seq_count]
    out["joint"] = joints[:seq_count]
    out["contact"] = contacts[:seq_count]
    out["fit_metadata"] = fit_meta[:seq_count]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved fitted ours dataset to: {out_path}")
    for i, meta in enumerate(fit_meta[:seq_count]):
        print(f"seq {i} fit metrics: {meta['fit']}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit SMPL GT to data/raw/ours FBX captures.")
    parser.add_argument("--raw-dir", default=str(paths.raw_ours))
    parser.add_argument("--processed-ours", default=str(paths.eval_dir / "ours.pt"))
    parser.add_argument("--output", default=str(paths.eval_dir / "ours_smpl.pt"))
    parser.add_argument("--cache-dir", default=str(paths.root_dir / "data/processed_datasets/ours_fbx_cache"))
    parser.add_argument("--blender", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--fixed-shape", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--no-processed-source", action="store_true",
                        help="Fit only from FBX and write placeholder acc/ori. Useful when raw IMU is unavailable.")
    return parser


if __name__ == "__main__":
    fit_ours(build_argparser().parse_args())
