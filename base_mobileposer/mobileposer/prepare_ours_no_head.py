"""Convert a local ours capture into the six no-head 5-IMU layout format."""

import argparse
import json
from pathlib import Path

import torch

from mobileposer.articulate import math
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.no_head_layouts import LAYOUTS


TARGET_FPS = 30
GRAVITY = 9.80665
# The no-head checkpoints were trained after this AMASS-to-runtime basis
# conversion in no_head_layouts._load_amass_dataset.
TRAINING_WORLD_ROT = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# HumanPose/Unity tracker coordinates are mirrored across x relative to the
# SMPL convention used by the canonical ours GT.  Apply the same basis change
# to positions and rotations before the AMASS-to-training-world transform.
UNITY_TO_SMPL_BASIS = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))

# Each layout is ordered exactly as its five-IMU model expects.
RAW_IMU_LAYOUTS = {
    "wrists_thighs_waist": ["LEFT_HAND", "RIGHT_HAND", "LEFT_UPPER_LEG", "RIGHT_UPPER_LEG", "WAIST"],
    "wrists_shanks_waist": ["LEFT_HAND", "RIGHT_HAND", "LEFT_LOWER_LEG", "RIGHT_LOWER_LEG", "WAIST"],
    "upperarms_thighs_waist": ["LEFT_UPPER_ARM", "RIGHT_UPPER_ARM", "LEFT_UPPER_LEG", "RIGHT_UPPER_LEG", "WAIST"],
    "wrists_upperarms_waist": ["LEFT_HAND", "RIGHT_HAND", "LEFT_UPPER_ARM", "RIGHT_UPPER_ARM", "WAIST"],
    "legs_waist": ["LEFT_UPPER_LEG", "RIGHT_UPPER_LEG", "LEFT_LOWER_LEG", "RIGHT_LOWER_LEG", "WAIST"],
}

# Feet have no direct IMU in the supplied raw-imu JSON. Their signal is
# synthesized from the FBX-fitted SMPL foot trajectory below.
TRACKER_FEET_LAYOUT = {
    "wrists_feet_waist": ["LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT", "WAIST"],
}

_ROLE_ALIASES = {"WAIST": ("WAIST", "HIP")}
_ROLE_TO_SMPL_JOINT = {
    "LEFT_HAND": 18, "RIGHT_HAND": 19,
    "LEFT_UPPER_ARM": 16, "RIGHT_UPPER_ARM": 17,
    "LEFT_UPPER_LEG": 1, "RIGHT_UPPER_LEG": 2,
    "LEFT_LOWER_LEG": 4, "RIGHT_LOWER_LEG": 5,
    "WAIST": 3, "LEFT_FOOT": 7, "RIGHT_FOOT": 8,
}
_VIRTUAL_TRACKER_TO_ROLE = {
    "LEFT_ELBOW": "LEFT_HAND", "RIGHT_ELBOW": "RIGHT_HAND",
    "LEFT_KNEE": "LEFT_LOWER_LEG", "RIGHT_KNEE": "RIGHT_LOWER_LEG",
    "LEFT_FOOT": "LEFT_FOOT", "RIGHT_FOOT": "RIGHT_FOOT",
    "WAIST": "WAIST", "CHEST": "CHEST",
}


def _vec3(value):
    return torch.tensor([float(value.get(a, 0.0)) for a in ("x", "y", "z")])


def _rotation_from_sensor(sensor):
    orientation = sensor.get("orientation") or {}
    if all(key in orientation for key in ("x", "y", "z", "w")):
        quat = torch.tensor([orientation["w"], orientation["x"], orientation["y"], orientation["z"]])
        if torch.isfinite(quat).all() and quat.norm() > 1e-6:
            return math.quaternion_to_rotation_matrix(quat.view(1, 4))[0]

    # Matches SlimeVrVirtualTrackerReplay.GetQuaternion.  The stored Euler
    # values are roll/pitch/yaw, not scipy's intrinsic XYZ convention.
    roll, pitch, yaw = torch.deg2rad(_vec3(sensor.get("eulerAnglesDeg", {}))) * 0.5
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    quat = torch.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))
    return math.quaternion_to_rotation_matrix(quat.view(1, 4))[0]


def _load_raw_imu(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    rate = float(raw.get("sampleRate", TARGET_FPS))
    step = max(1, round(rate / TARGET_FPS))
    frames = raw["frames"][::step]
    acc = {}
    ori = {}
    online = {}
    for role in set(sum(RAW_IMU_LAYOUTS.values(), [])):
        acc[role] = torch.zeros(len(frames), 3)
        ori[role] = torch.eye(3).repeat(len(frames), 1, 1)
        online[role] = torch.zeros(len(frames), dtype=torch.bool)
    for i, frame in enumerate(frames):
        by_role = {s.get("trackerRole"): s for s in frame.get("sensors", [])}
        for role in acc:
            sensor = next((by_role.get(alias) for alias in _ROLE_ALIASES.get(role, (role,)) if alias in by_role), None)
            if sensor is None:
                continue
            rot = _rotation_from_sensor(sensor)
            ori[role][i] = rot
            acc[role][i] = rot.matmul(_vec3(sensor.get("accelerationG", {})) * GRAVITY)
            online[role][i] = bool(sensor.get("online", False))
    return acc, ori, online


def _project_to_rotation(matrix):
    u, _, vh = torch.linalg.svd(matrix)
    result = u @ vh
    if torch.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def _calibrate_raw_imu(acc, ori, online, pose, frames):
    """Align each physical tracker frame to the corresponding SMPL bone frame."""
    n = min(len(pose), len(next(iter(ori.values()))))
    global_pose, _ = ParametricModel(paths.smpl_file).forward_kinematics(pose[:n])
    calibration = {}
    for role, joint in _ROLE_TO_SMPL_JOINT.items():
        if role not in ori:
            continue
        mask = online[role][:n].clone()
        mask[frames:n] = False
        if not mask.any():
            continue
        # R_device(t) @ R_device_to_bone = R_bone(t).
        device_to_bone = _project_to_rotation(
            (ori[role][:n][mask].transpose(-1, -2) @ global_pose[:n, joint][mask]).sum(dim=0)
        )
        ori[role] = ori[role] @ device_to_bone
        # Raw recordings provide linear acceleration. Remove its static bias
        # after it has been expressed in the shared world frame.
        acc[role] -= acc[role][:n][mask].mean(dim=0, keepdim=True)
        calibration[role] = int(mask.sum())
    return acc, ori, calibration


def _to_training_coordinates(pose, tran, shape):
    """Express Unity-derived GT in the world basis used by the checkpoints."""
    rotation = TRAINING_WORLD_ROT.to(pose)
    pose = pose.clone()
    pose[:, 0] = rotation @ pose[:, 0]
    tran = (rotation @ tran.unsqueeze(-1)).squeeze(-1)
    _, joint = ParametricModel(paths.smpl_file).forward_kinematics(pose, shape, tran)
    return pose, tran, joint


def _syn_acc(points):
    result = torch.zeros_like(points)
    if len(points) > 2:
        result[1:-1] = (points[:-2] + points[2:] - 2 * points[1:-1]) * TARGET_FPS**2
    return result


def _load_tracker_feet(path, length):
    raw = json.loads(path.read_text(encoding="utf-8"))
    frames = raw["frames"][:length]
    positions = {name: torch.zeros(length, 3) for name in ("LEFT_FOOT", "RIGHT_FOOT")}
    rotations = {name: torch.eye(3).repeat(length, 1, 1) for name in positions}
    present = {name: torch.zeros(length, dtype=torch.bool) for name in positions}
    for i, frame in enumerate(frames):
        for tracker in frame.get("trackers", []):
            name = tracker.get("name", "").split("/")[-1]
            if name not in positions:
                continue
            positions[name][i] = _vec3(tracker["position"])
            q = tracker["rotation"]
            # tracker JSON uses x,y,z,w; project math expects w,x,y,z.
            quat = torch.tensor([q["w"], q["x"], q["y"], q["z"]]).view(1, 4)
            rotations[name][i] = math.quaternion_to_rotation_matrix(quat)[0]
            present[name][i] = True
    return positions, rotations, present


def _load_virtual_trackers(path, length):
    """Load the Unity tracker stream used to drive the recorded HumanPose."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    roles = set(sum(RAW_IMU_LAYOUTS.values(), [])) | {"LEFT_FOOT", "RIGHT_FOOT"}
    positions = {role: torch.zeros(length, 3) for role in roles}
    rotations = {role: torch.eye(3).repeat(length, 1, 1) for role in roles}
    present = {role: torch.zeros(length, dtype=torch.bool) for role in roles}
    for frame_index, frame in enumerate(raw["frames"][:length]):
        for tracker in frame.get("trackers", []):
            name = tracker.get("name", "").split("//")[-1]
            role = _VIRTUAL_TRACKER_TO_ROLE.get(name)
            if role not in roles:
                continue
            positions[role][frame_index] = UNITY_TO_SMPL_BASIS @ _vec3(tracker["position"])
            q = tracker["rotation"]
            quat = torch.tensor([q["w"], q["x"], q["y"], q["z"]]).view(1, 4)
            unity_rotation = math.quaternion_to_rotation_matrix(quat)[0]
            rotations[role][frame_index] = (
                UNITY_TO_SMPL_BASIS @ unity_rotation @ UNITY_TO_SMPL_BASIS
            )
            present[role][frame_index] = True
    return {role: _syn_acc(position) for role, position in positions.items()}, rotations, present


def _process_capture(capture_dir, source, output_root, capture_idx, calibration_frames, input_source):
    manifest = json.loads((capture_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_imu_path = capture_dir / manifest["rawImuJsonFile"]
    tracker_path = capture_dir / manifest["trackerJsonFile"]
    processed = torch.load(source, map_location="cpu")
    source_pose = processed["pose"][capture_idx]
    source_tran = processed["tran"][capture_idx]
    source_shape = processed["shape"][capture_idx]
    pose, tran, joint = _to_training_coordinates(source_pose, source_tran, source_shape)
    if input_source == "raw-imu":
        imu_acc, imu_ori, imu_online = _load_raw_imu(raw_imu_path)
        n = min(len(pose), len(next(iter(imu_acc.values()))))
        positions, rotations, foot_present = _load_tracker_feet(tracker_path, n)
        for role in ("LEFT_FOOT", "RIGHT_FOOT"):
            imu_acc[role] = _syn_acc(positions[role])
            imu_ori[role] = rotations[role]
            imu_online[role] = foot_present[role]
    else:
        n = min(len(pose), len(json.loads(tracker_path.read_text(encoding="utf-8"))["frames"]))
        imu_acc, imu_ori, imu_online = _load_virtual_trackers(tracker_path, n)
    rotation = TRAINING_WORLD_ROT.to(pose)
    for role in imu_ori:
        imu_ori[role] = rotation @ imu_ori[role]
        imu_acc[role] = (rotation @ imu_acc[role].unsqueeze(-1)).squeeze(-1)
    imu_acc, imu_ori, calibration = _calibrate_raw_imu(
        imu_acc, imu_ori, imu_online, pose, calibration_frames
    )

    def roles_available(labels):
        return all(role in imu_online and bool(imu_online[role][:n].all()) for role in labels)

    for layout_name, labels in RAW_IMU_LAYOUTS.items():
        if not roles_available(labels):
            missing = [role for role in labels if not bool(imu_online.get(role, torch.zeros(n, dtype=torch.bool))[:n].all())]
            print(f"[skip] {layout_name}: unavailable tracker(s): {missing}")
            continue
        data = {
            "layout": layout_name,
            "layout_labels": LAYOUTS[layout_name]["labels"],
            "layout_joints": LAYOUTS[layout_name]["joints"],
            "acc": [torch.stack([imu_acc[x][:n] for x in labels], dim=1)],
            "ori": [torch.stack([imu_ori[x][:n] for x in labels], dim=1)],
            "pose": [pose[:n]],
            "tran": [tran[:n]],
            "shape": [source_shape],
            "joint": [joint[:n]],
            "contact": [processed["contact"][capture_idx][:n]],
            "metadata": [{
                "source": str(capture_dir), "signal_source": f"{input_source}-calibrated",
                "calibration_frames": calibration,
            }],
        }
        out = output_root / layout_name / "ours.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out)

    labels = TRACKER_FEET_LAYOUT["wrists_feet_waist"]
    if not roles_available(labels):
        missing = [role for role in labels if not bool(imu_online.get(role, torch.zeros(n, dtype=torch.bool))[:n].all())]
        print(f"[skip] wrists_feet_waist: unavailable tracker(s): {missing}")
    else:
        data = {
            "layout": "wrists_feet_waist",
            "layout_labels": LAYOUTS["wrists_feet_waist"]["labels"],
            "layout_joints": LAYOUTS["wrists_feet_waist"]["joints"],
            "acc": [torch.stack([imu_acc[x][:n] for x in labels], dim=1)],
            "ori": [torch.stack([imu_ori[x][:n] for x in labels], dim=1)],
            "pose": [pose[:n]],
            "tran": [tran[:n]],
            "shape": [source_shape],
            "joint": [joint[:n]],
            "contact": [processed["contact"][capture_idx][:n]],
            "metadata": [{
                "source": str(capture_dir), "signal_source": f"{input_source}-calibrated",
                "calibration_frames": calibration,
            }],
        }
        out = output_root / "wrists_feet_waist" / "ours.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=paths.raw_ours / "1")
    parser.add_argument("--capture-index", type=int, default=0,
                        help="Sequence index in --processed-ours corresponding to --capture-dir.")
    parser.add_argument("--calibration-frames", type=int, default=60,
                        help="Initial online frames used to estimate each tracker-to-bone rotation.")
    parser.add_argument("--input-source", choices=("raw-imu", "tracker-json"), default="raw-imu",
                        help="Use physical raw IMUs or Unity's recorded virtual tracker stream.")
    parser.add_argument("--processed-ours", type=Path, default=paths.eval_dir / "ours_smpl.pt")
    parser.add_argument("--output-root", type=Path, default=paths.root_dir / "data/ours_no_head_5imu")
    args = parser.parse_args()
    _process_capture(args.capture_dir, args.processed_ours, args.output_root,
                     args.capture_index, args.calibration_frames, args.input_source)
    print(f"Saved six layout datasets under {args.output_root}")


if __name__ == "__main__":
    main()
