"""Retarget Unity HumanPose rotations to SMPL without per-frame IK fitting.

The source cache must be exported by ``HumanPoseJointExporter`` with a
``restPose``.  It contains the VRM bone axes needed to turn Unity world
rotations into animation deltas.  This is a rig retargeting operation, not a
claim that the original capture contained native SMPL parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mobileposer.articulate import math
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.fit_ours_smpl import (
    _HUMANPOSE_BONES,
    _bone_target_from_fbx,
    _capture_dirs,
    _capture_name,
    _fit_sequence,
    _foot_ground_probs,
    _target_from_fbx,
)


# Each SMPL joint receives a Unity bone's global animation delta.  The VRM
# Humanoid schema has no explicit spine2, neck, clavicles, or palm-end bones;
# those inherit the nearest articulated parent so their local SMPL rotation is
# neutral instead of counter-rotating the chain.
_SMPL_TO_HUMANPOSE = (
    "pelvis", "femur_l", "femur_r", "lumbar_body", "tibia_l", "tibia_r",
    "lumbar_body", "talus_l", "talus_r", "thorax", "toes_l", "toes_r",
    "thorax", "thorax", "thorax", "head", "humerus_l", "humerus_r",
    "ulna_l", "ulna_r", "hand_l", "hand_r", "hand_l", "hand_r",
)

_HUMANPOSE_PARENT = (None, 0, 0, 1, 2, 3, 4, 5, 6, 0, 9, 10, 10, 10, 12, 13, 14, 15)
_SMPL_DIRECT_SOURCE = (0, 1, 2, 9, 3, 4, None, 5, 6, 10, 7, 8, None, None, None, 11, 12, 13, 14, 15, 16, 17, None, None)


def _quat_to_matrix(value: dict) -> torch.Tensor:
    # Unity serializes Quaternion as xyzw; articulate expects wxyz.
    quat = torch.tensor([value["w"], value["x"], value["y"], value["z"]], dtype=torch.float32)
    rotation = math.quaternion_to_rotation_matrix(quat.view(1, 4))[0]
    basis = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    return basis @ rotation @ basis


def _vec3(value: dict) -> torch.Tensor:
    return torch.tensor([value["x"], value["y"], value["z"]], dtype=torch.float32)


def _load_humanpose(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with cache_path.open(encoding="utf-8") as f:
        package = json.load(f)

    bones = package.get("bones") or []
    rest = package.get("restPose")
    frames = package.get("frames") or []
    if not rest or not frames:
        raise ValueError(f"{cache_path} lacks restPose or frames; re-export it with HumanPoseJointExporter.")
    index = {name: i for i, name in enumerate(bones)}
    missing = sorted(set(_SMPL_TO_HUMANPOSE) - set(index))
    if missing:
        raise ValueError(f"{cache_path} is missing HumanPose bones: {missing}")

    source_indices = [index[name] for name in _SMPL_TO_HUMANPOSE]
    rest_rotations = rest.get("rotations") or []
    rest_positions = rest.get("positions") or []
    if len(rest_rotations) != len(bones) or len(rest_positions) != len(bones):
        raise ValueError(f"{cache_path} has an incomplete restPose.")

    rest_global = torch.stack([_quat_to_matrix(rest_rotations[i]) for i in source_indices])
    pelvis_index = index["pelvis"]
    frame_global, pelvis_positions = [], []
    for frame_index, frame in enumerate(frames):
        rotations = frame.get("rotations") or []
        positions = frame.get("positions") or []
        present = frame.get("present") or []
        if len(rotations) != len(bones) or len(positions) != len(bones):
            raise ValueError(f"{cache_path}: incomplete frame {frame_index}.")
        if any(not bool(present[i]) for i in source_indices):
            raise ValueError(f"{cache_path}: a required bone is absent in frame {frame_index}.")
        frame_global.append(torch.stack([_quat_to_matrix(rotations[i]) for i in source_indices]))
        pelvis = _vec3(positions[pelvis_index])
        pelvis[0] = -pelvis[0]
        pelvis_positions.append(pelvis)

    rest_positions_by_name = torch.stack([
        torch.stack((-_vec3(rest_positions[index[name]])[0], _vec3(rest_positions[index[name]])[1],
                     _vec3(rest_positions[index[name]])[2]))
        for name in _HUMANPOSE_BONES
    ])
    return torch.stack(frame_global), rest_global, torch.stack(pelvis_positions), rest_positions_by_name


def _calibrate_rest_pose(rest_positions: torch.Tensor, iterations: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Fit the avatar-to-SMPL rest-pose offset once, never per motion frame."""
    smpl_ids, target, weights = _target_from_fbx(rest_positions.unsqueeze(0))
    parents, children, directions, direction_weights = _bone_target_from_fbx(rest_positions.unsqueeze(0))
    # _fit_sequence includes temporal terms, so duplicate the static frame.
    pose, tran, shape, _, metrics = _fit_sequence(
        target.repeat(2, 1, 1), smpl_ids, weights.expand(2, -1).clone(),
        parents, children, directions.repeat(2, 1, 1), direction_weights.expand(2, -1).clone(),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        iterations=iterations, lr=0.03, optimize_betas=True,
    )
    return pose[:1], tran[:1], shape, metrics


def retarget_cache(cache_path: Path, mode: str = "calibrated", calibration_iterations: int = 500) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
    """Return SMPL local pose, root translation, shape, joints, and calibration metadata."""
    source_global, rest_global, pelvis_positions, rest_positions = _load_humanpose(cache_path)

    body = ParametricModel(paths.smpl_file)
    if mode == "calibrated":
        base_pose, base_tran, shape, calibration = _calibrate_rest_pose(rest_positions, calibration_iterations)
        base_global, _ = body.forward_kinematics(base_pose, shape)
        animation_delta = source_global @ rest_global.transpose(-1, -2).unsqueeze(0)
        target_global = animation_delta @ base_global
        pose = body.inverse_kinematics_R(target_global).view(-1, 24, 3, 3)
        tran = pelvis_positions + (base_tran[0] - rest_positions[0])
    elif mode == "global":
        # R(t) R(rest)^T is the world-space animation delta. Missing joints
        # inherit their source parent, making their target local rotation I.
        target_global = source_global @ rest_global.transpose(-1, -2).unsqueeze(0)
        pose = body.inverse_kinematics_R(target_global).view(-1, 24, 3, 3)
        shape = torch.zeros(10, dtype=torch.float32)
        tran = pelvis_positions
        calibration = None
    else:
        source_local = source_global.clone()
        rest_local = rest_global.clone()
        for child, parent in enumerate(_HUMANPOSE_PARENT):
            if parent is not None:
                source_local[:, child] = source_global[:, parent].transpose(-1, -2) @ source_global[:, child]
                rest_local[child] = rest_global[parent].transpose(-1, -2) @ rest_global[child]
        pose = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(source_global.shape[0], 24, 1, 1)
        for smpl_joint, source_joint in enumerate(_SMPL_DIRECT_SOURCE):
            if source_joint is None:
                continue
            if mode == "local-left":
                pose[:, smpl_joint] = source_local[:, source_joint] @ rest_local[source_joint].transpose(-1, -2)
            elif mode == "local-right":
                pose[:, smpl_joint] = rest_local[source_joint].transpose(-1, -2) @ source_local[:, source_joint]
            else:
                raise ValueError(f"Unknown retarget mode: {mode}")
        shape = torch.zeros(10, dtype=torch.float32)
        tran = pelvis_positions
        calibration = None
    _, joints = body.forward_kinematics(pose, shape, tran)
    return pose, tran, shape, joints, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Directly retarget cached Unity HumanPose rotations to SMPL.")
    parser.add_argument("--cache-dir", type=Path, default=paths.processed_datasets / "ours_humanpose_cache")
    parser.add_argument("--raw-dir", type=Path, default=paths.raw_ours)
    parser.add_argument("--output", type=Path, default=paths.eval_dir / "ours_smpl_humanpose_retarget.pt")
    parser.add_argument("--mode", choices=("calibrated", "global", "local-left", "local-right"), default="calibrated")
    parser.add_argument("--calibration-iterations", type=int, default=500)
    args = parser.parse_args()

    poses, trans, shapes, joints, contacts, metadata = [], [], [], [], [], []
    for capture_dir in _capture_dirs(args.raw_dir):
        cache_path = args.cache_dir / f"{_capture_name(capture_dir)}.humanpose_joints.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing HumanPose cache for {capture_dir.name}: {cache_path}")
        pose, tran, shape, joint, calibration = retarget_cache(
            cache_path, args.mode, args.calibration_iterations
        )
        poses.append(pose)
        trans.append(tran)
        shapes.append(shape)
        joints.append(joint)
        contacts.append(_foot_ground_probs(joint))
        metadata.append({
            "capture_dir": str(capture_dir),
            "source_type": "unity_humanpose_rotation_retarget",
            "retarget_mode": args.mode,
            "rest_calibration": calibration,
            "cache_path": str(cache_path),
            "shape": "SMPL mean shape; not recoverable from HumanPose",
        })
        print(f"Retargeted {capture_dir.name}: {pose.shape[0]} frames")

    # IMU data is intentionally not fabricated.  Merge pose/tran into a
    # separately validated and time-aligned IMU dataset before training.
    out = {
        "pose": poses,
        "tran": trans,
        "shape": shapes,
        "joint": joints,
        "contact": contacts,
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved rotation-retargeted SMPL poses to: {args.output}")


if __name__ == "__main__":
    main()
