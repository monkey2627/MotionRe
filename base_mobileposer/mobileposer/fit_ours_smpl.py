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

_HUMANPOSE_BONES = _FBX_BONES

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

# Unity's virtual trackers are world-space rigid transforms.  These are the
# only roles that can be tied to an actual SMPL joint for the 2_side_bridge
# recording; do not infer a signal for roles absent from tracker.json.
_TRACKER_TO_SMPL_JOINT = {
    "CHEST": 9,
    "WAIST": 3,
    "LEFT_ELBOW": 18,
    "RIGHT_ELBOW": 19,
    "LEFT_KNEE": 4,
    "RIGHT_KNEE": 5,
    "LEFT_FOOT": 7,
    "RIGHT_FOOT": 8,
}
_UNITY_TO_SMPL_BASIS = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))


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

    # RuntimeMotionPackageExporter writes Unity local positions to FBX with:
    #     fbx = (-unity.x, unity.y, unity.z)
    # and declares MayaYUp. Blender exposes the imported animation in the same
    # Y-up numeric basis for this exported skeleton, so invert only the x flip.
    positions = torch.stack(
        (-positions[..., 0], positions[..., 1], positions[..., 2]), dim=-1
    )
    return positions, meta


def _load_humanpose_positions(json_path: Path) -> Tuple[torch.Tensor, Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        package = json.load(f)

    bones = package.get("bones") or []
    frames = package.get("frames") or []
    if not bones or not frames:
        raise ValueError(f"HumanPose joint cache has no bones or frames: {json_path}")

    name_to_idx = {name: idx for idx, name in enumerate(bones)}
    missing = [name for name in _HUMANPOSE_BONES if name not in name_to_idx]
    if missing:
        raise ValueError(f"HumanPose joint cache is missing bones {missing}: {json_path}")

    positions = np.full((len(frames), len(_HUMANPOSE_BONES), 3), np.nan, dtype=np.float32)
    present = np.zeros((len(frames), len(_HUMANPOSE_BONES)), dtype=bool)
    for frame_idx, frame in enumerate(frames):
        frame_positions = frame.get("positions") or []
        frame_present = frame.get("present") or []
        for bone_idx, bone_name in enumerate(_HUMANPOSE_BONES):
            src_idx = name_to_idx[bone_name]
            if src_idx >= len(frame_positions):
                continue
            is_present = bool(frame_present[src_idx]) if src_idx < len(frame_present) else True
            if not is_present:
                continue
            value = frame_positions[src_idx]
            # Unity HumanPose uses the mirrored x handedness of the SMPL/FBX
            # convention used by this project (the SMPL left hip is +x).
            positions[frame_idx, bone_idx] = (-value["x"], value["y"], value["z"])
            present[frame_idx, bone_idx] = True

    meta = {
        "source": str(json_path),
        "source_type": "unity_humanpose",
        "fps": package.get("fps"),
        "frameCount": package.get("frameCount", len(frames)),
        "bones": _HUMANPOSE_BONES,
        "humanBones": package.get("humanBones"),
        "coordinateSystem": package.get("coordinateSystem", "unity_world_xyz"),
        "playback": package.get("playback"),
        "present": present.all(axis=0).tolist(),
    }
    return torch.from_numpy(positions).float(), meta


def _load_target_positions(cache_dir: Path, capture_dir: Path) -> Tuple[torch.Tensor, Dict]:
    humanpose_cache = cache_dir / f"{_capture_name(capture_dir)}.humanpose_joints.json"
    if humanpose_cache.exists():
        positions, meta = _load_humanpose_positions(humanpose_cache)
        meta["cache_path"] = str(humanpose_cache)
        return positions, meta

    fbx_cache = cache_dir / f"{_capture_name(capture_dir)}.fbx_joints.npz"
    positions, meta = _load_fbx_positions(fbx_cache)
    meta["source_type"] = "fbx_regular_bones"
    meta["cache_path"] = str(fbx_cache)
    return positions, meta


def _target_from_fbx(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bone_to_idx = {name: idx for idx, name in enumerate(_HUMANPOSE_BONES)}
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
    bone_to_idx = {name: idx for idx, name in enumerate(_HUMANPOSE_BONES)}
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


def _unity_quat_to_matrix(value: Dict) -> torch.Tensor:
    """Convert a Unity xyzw quaternion to the project's wxyz convention."""
    quat = torch.tensor([value["w"], value["x"], value["y"], value["z"]], dtype=torch.float32)
    rotation = math.quaternion_to_rotation_matrix(quat.view(1, 4))[0]
    basis = _UNITY_TO_SMPL_BASIS.to(rotation)
    return basis @ rotation @ basis


def _load_virtual_tracker_rotations(tracker_path: Path, length: int,
                                    requested_roles: List[str] | None = None) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Load only real Unity virtual trackers with an SMPL joint correspondence."""
    package = json.loads(tracker_path.read_text(encoding="utf-8"))
    frames = package.get("frames") or []
    usable_roles = []
    for frame in frames[:length]:
        for tracker in frame.get("trackers", []):
            role = tracker.get("name", "").split("//")[-1]
            if (role in _TRACKER_TO_SMPL_JOINT
                    and (requested_roles is None or role in requested_roles)
                    and role not in usable_roles):
                usable_roles.append(role)
    if not usable_roles:
        raise ValueError(f"No mappable virtual trackers found in {tracker_path}")

    rotations = torch.eye(3).view(1, 1, 3, 3).repeat(length, len(usable_roles), 1, 1)
    present = torch.zeros(length, len(usable_roles), dtype=torch.bool)
    role_index = {role: index for index, role in enumerate(usable_roles)}
    for frame_index, frame in enumerate(frames[:length]):
        for tracker in frame.get("trackers", []):
            role = tracker.get("name", "").split("//")[-1]
            if role not in role_index or "rotation" not in tracker:
                continue
            rotations[frame_index, role_index[role]] = _unity_quat_to_matrix(tracker["rotation"])
            present[frame_index, role_index[role]] = True
    # A rotation constraint is meaningful only for a stream present throughout
    # the fitted interval.  This prevents silently filling holes with identity.
    keep = present.all(dim=0)
    rotations = rotations[:, keep]
    roles = [role for role, valid in zip(usable_roles, keep.tolist()) if valid]
    if not roles:
        raise ValueError(f"No continuously present mappable trackers found in {tracker_path}")
    return rotations, roles, present[:, keep]


def _project_rotation(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix)
    result = u @ vh
    if torch.linalg.det(result) < 0:
        u = u.clone()
        u[:, -1] *= -1
        result = u @ vh
    return result


def _tracker_to_bone_offsets(tracker_rotation: torch.Tensor, global_rotation: torch.Tensor,
                             calibration_frames: int) -> torch.Tensor:
    """Estimate fixed B in R_tracker(t) @ B = R_smpl_bone(t)."""
    count = min(calibration_frames, tracker_rotation.shape[0])
    if count < 1:
        raise ValueError("calibration_frames must be positive")
    return torch.stack([
        _project_rotation((tracker_rotation[:count, i].transpose(-1, -2) @ global_rotation[:count, i]).sum(0))
        for i in range(tracker_rotation.shape[1])
    ])


def _orientation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared geodesic surrogate; stable at the pi discontinuity."""
    relative = predicted.transpose(-1, -2) @ target
    return (3.0 - relative.diagonal(dim1=-2, dim2=-1).sum(-1)).mean()


def _fit_sequence_orientation_constrained(
    target: torch.Tensor, smpl_ids: torch.Tensor, weights: torch.Tensor,
    bone_parents: torch.Tensor, bone_children: torch.Tensor, bone_dirs: torch.Tensor,
    bone_weights: torch.Tensor, tracker_rotation: torch.Tensor, tracker_joints: torch.Tensor,
    initial_pose: torch.Tensor, initial_tran: torch.Tensor, initial_shape: torch.Tensor,
    *, device: torch.device, iterations: int, lr: float, calibration_frames: int,
    orientation_weight: float, offset_update_interval: int, smooth_weight: float,
    optimize_shape: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Fit SMPL while preserving HumanPose geometry and tracker orientation motion.

    Tracker axes are not assumed to be SMPL axes.  A per-role fixed offset is
    estimated on the initial calibration segment and re-estimated periodically
    from the current solution, which is the alternating minimization step.
    """
    body = ParametricModel(paths.smpl_file, device=device)
    target, smpl_ids, weights = target.to(device), smpl_ids.to(device), weights.to(device)
    bone_parents, bone_children = bone_parents.to(device), bone_children.to(device)
    bone_dirs, bone_weights = bone_dirs.to(device), bone_weights.to(device)
    tracker_rotation, tracker_joints = tracker_rotation.to(device), tracker_joints.to(device)
    n_frames = target.shape[0]

    pose_r6d = math.rotation_matrix_to_r6d(initial_pose.to(device)).view(n_frames, 24, 6)
    pose_r6d = pose_r6d.detach().clone().requires_grad_(True)
    tran = initial_tran.to(device).detach().clone().requires_grad_(True)
    betas = initial_shape.to(device).detach().clone().requires_grad_(optimize_shape)
    params = [pose_r6d, tran] + ([betas] if optimize_shape else [])
    optimizer = torch.optim.Adam(params, lr=lr)

    with torch.no_grad():
        initial_global = body.forward_kinematics_R(math.r6d_to_rotation_matrix(pose_r6d).view(n_frames, 24, 3, 3))
        offsets = _tracker_to_bone_offsets(tracker_rotation, initial_global[:, tracker_joints], calibration_frames)

    previous_loss = None
    for step in tqdm(range(iterations), desc="Fitting SMPL + tracker orientation", leave=False):
        if step and step % offset_update_interval == 0:
            with torch.no_grad():
                current_pose = math.r6d_to_rotation_matrix(pose_r6d).view(n_frames, 24, 3, 3)
                current_global = body.forward_kinematics_R(current_pose)
                offsets = _tracker_to_bone_offsets(
                    tracker_rotation, current_global[:, tracker_joints], calibration_frames
                )
        optimizer.zero_grad()
        pose = math.r6d_to_rotation_matrix(pose_r6d).view(n_frames, 24, 3, 3)
        global_rotation, joint = body.forward_kinematics(pose, betas, tran)
        joint_loss = (((joint[:, smpl_ids] - target).square().sum(-1) * weights).sum()
                      / weights.sum().clamp_min(1.0))
        pred_bones = torch.nn.functional.normalize(joint[:, bone_children] - joint[:, bone_parents], dim=-1)
        bone_loss = (((1.0 - (pred_bones * bone_dirs).sum(-1)) * bone_weights).sum()
                     / bone_weights.sum().clamp_min(1.0))
        orientation_target = tracker_rotation @ offsets.unsqueeze(0)
        orientation_loss = _orientation_loss(global_rotation[:, tracker_joints], orientation_target)
        eye = torch.eye(3, device=device).view(1, 1, 3, 3)
        pose_prior = (pose[:, 1:] - eye).square().mean()
        smooth_pose = (pose_r6d[1:] - pose_r6d[:-1]).square().mean()
        smooth_tran = (tran[1:] - tran[:-1]).square().mean()
        loss = (joint_loss + 0.05 * bone_loss + orientation_weight * orientation_loss
                + 0.002 * pose_prior + smooth_weight * smooth_pose + 0.02 * smooth_tran
                + 0.001 * betas.square().mean())
        loss.backward()
        optimizer.step()
        previous_loss = loss.detach()

    with torch.no_grad():
        pose = math.r6d_to_rotation_matrix(pose_r6d).view(n_frames, 24, 3, 3)
        global_rotation, joint = body.forward_kinematics(pose, betas, tran)
        offsets = _tracker_to_bone_offsets(tracker_rotation, global_rotation[:, tracker_joints], calibration_frames)
        target_rotation = tracker_rotation @ offsets.unsqueeze(0)
        angular = torch.acos(((global_rotation[:, tracker_joints].transpose(-1, -2) @ target_rotation)
                              .diagonal(dim1=-2, dim2=-1).sum(-1).sub(1.0).div(2.0)).clamp(-1, 1))
        error = torch.norm(joint[:, smpl_ids] - target, dim=-1)
        valid = weights > 0
    return (pose.cpu(), tran.cpu(), betas.cpu(), joint.cpu(), {
        "loss": float(previous_loss.item()) if previous_loss is not None else 0.0,
        "mean_joint_error_m": float(error[valid].mean().item()),
        "max_joint_error_m": float(error[valid].max().item()),
        "mean_tracker_rotation_error_deg": float(torch.rad2deg(angular).mean().item()),
        "tracker_roles": list(_TRACKER_TO_SMPL_JOINT.keys()),
        "calibration_frames": calibration_frames,
        "orientation_weight": orientation_weight,
        "smooth_weight": smooth_weight,
    })


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
    blender = None
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
        humanpose_cache = cache_dir / f"{_capture_name(capture_dir)}.humanpose_joints.json"
        if humanpose_cache.exists():
            print(f"Using Unity HumanPose joints: {humanpose_cache}")
        else:
            cache_npz = cache_dir / f"{_capture_name(capture_dir)}.fbx_joints.npz"
            print(f"HumanPose cache missing; exporting FBX joints: {manifest_fbx}")
            if blender is None:
                blender = _find_blender(args.blender)
            _export_fbx_joints(blender, manifest_fbx, cache_npz, args.overwrite_cache)

        positions, source_meta = _load_target_positions(cache_dir, capture_dir)
        smpl_ids, target, weights = _target_from_fbx(positions)
        bone_parents, bone_children, bone_dirs, bone_weights = _bone_target_from_fbx(positions)
        imu_len = processed["acc"][seq_idx].shape[0] if processed is not None else target.shape[0]
        fit_len = min(imu_len, target.shape[0])
        tracker_path = _manifest_path(capture_dir, "trackerJsonFile")
        tracker_rotation = tracker_roles = None
        if args.orientation_constrained:
            tracker_frame_count = len(json.loads(tracker_path.read_text(encoding="utf-8")).get("frames") or [])
            fit_len = min(fit_len, tracker_frame_count)
            tracker_rotation, tracker_roles, _ = _load_virtual_tracker_rotations(
                tracker_path, fit_len, args.tracker_roles
            )
            fit_len = min(fit_len, tracker_rotation.shape[0])
        target = target[:fit_len]
        weights = weights[:fit_len].clone()
        bone_dirs = bone_dirs[:fit_len]
        bone_weights = bone_weights[:fit_len].clone()

        print(f"Fitting sequence {seq_idx}: {fit_len} frames")
        if args.orientation_constrained:
            # Lazy import avoids the retarget module's intentional dependency
            # on this file's HumanPose position helpers.
            from mobileposer.retarget_ours_humanpose import retarget_cache

            initial_pose, initial_tran, initial_shape, _, initial_calibration = retarget_cache(
                humanpose_cache, "calibrated", args.retarget_calibration_iterations
            )
            tracker_joints = torch.tensor([_TRACKER_TO_SMPL_JOINT[role] for role in tracker_roles])
            pose, tran, shape, joint, metrics = _fit_sequence_orientation_constrained(
                target, smpl_ids, weights, bone_parents, bone_children, bone_dirs, bone_weights,
                tracker_rotation[:fit_len], tracker_joints,
                initial_pose[:fit_len], initial_tran[:fit_len], initial_shape,
                device=device, iterations=args.iterations, lr=args.lr,
                calibration_frames=args.calibration_frames,
                orientation_weight=args.orientation_weight,
                offset_update_interval=args.offset_update_interval,
                smooth_weight=args.smooth_weight,
                optimize_shape=not args.fixed_shape,
            )
            metrics["tracker_roles"] = tracker_roles
            metrics["initial_retarget_calibration"] = initial_calibration
        else:
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
                "source_meta": source_meta,
                "fit": metrics,
                "joint_map": _JOINT_MAP,
                "coordinate_map": source_meta.get("coordinateSystem", "unity_world_xyz"),
                "tracker_json": str(tracker_path),
                "orientation_constrained": bool(args.orientation_constrained),
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
    parser = argparse.ArgumentParser(description="Fit SMPL GT to data/raw/ours HumanPose or FBX captures.")
    parser.add_argument("--raw-dir", default=str(paths.raw_ours))
    parser.add_argument("--processed-ours", default=str(paths.eval_dir / "ours.pt"))
    parser.add_argument("--output", default=str(paths.eval_dir / "ours_smpl.pt"))
    parser.add_argument("--cache-dir", default=str(paths.root_dir / "data/processed_datasets/ours_humanpose_cache"))
    parser.add_argument("--blender", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--fixed-shape", action="store_true")
    parser.add_argument("--orientation-constrained", action="store_true",
                        help="Jointly fit HumanPose geometry and actual Unity virtual-tracker orientations.")
    parser.add_argument("--calibration-frames", type=int, default=60,
                        help="Initial static frames for fixed tracker-to-SMPL rotation offsets.")
    parser.add_argument("--orientation-weight", type=float, default=0.1)
    parser.add_argument("--smooth-weight", type=float, default=0.02)
    parser.add_argument("--offset-update-interval", type=int, default=50,
                        help="Alternating offset-estimation interval in optimizer iterations.")
    parser.add_argument("--retarget-calibration-iterations", type=int, default=300,
                        help="Static HumanPose-to-SMPL initialization iterations.")
    parser.add_argument("--tracker-roles", nargs="+", default=None,
                        choices=sorted(_TRACKER_TO_SMPL_JOINT),
                        help="Only constrain these actual Unity tracker roles during orientation fitting.")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--no-processed-source", action="store_true",
                        help="Fit only from joint cache and write placeholder acc/ori. Useful when raw IMU is unavailable.")
    return parser


if __name__ == "__main__":
    fit_ours(build_argparser().parse_args())
