"""Convert a local ours capture into the six no-head 5-IMU layout format."""

import argparse
import json
from pathlib import Path

import torch

from mobileposer.articulate import math
from mobileposer.config import paths
from mobileposer.no_head_layouts import LAYOUTS


TARGET_FPS = 30
GRAVITY = 9.80665

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


def _vec3(value):
    return torch.tensor([float(value.get(a, 0.0)) for a in ("x", "y", "z")])


def _load_raw_imu(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    rate = float(raw.get("sampleRate", TARGET_FPS))
    step = max(1, round(rate / TARGET_FPS))
    frames = raw["frames"][::step]
    acc = {}
    ori = {}
    for role in set(sum(RAW_IMU_LAYOUTS.values(), [])):
        acc[role] = torch.zeros(len(frames), 3)
        ori[role] = torch.eye(3).repeat(len(frames), 1, 1)
    for i, frame in enumerate(frames):
        by_role = {s.get("trackerRole"): s for s in frame.get("sensors", [])}
        for role, sensor in by_role.items():
            if role not in acc:
                continue
            rot = math.euler_angle_to_rotation_matrix(
                torch.deg2rad(_vec3(sensor.get("eulerAnglesDeg", {}))).view(1, 3),
                seq="XYZ",
            )[0]
            ori[role][i] = rot
            acc[role][i] = rot.matmul(_vec3(sensor.get("accelerationG", {})) * GRAVITY)
    return acc, ori


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
    return positions, rotations


def _process_capture(capture_dir, source, output_root, capture_idx):
    manifest = json.loads((capture_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_imu_path = capture_dir / manifest["rawImuJsonFile"]
    tracker_path = capture_dir / manifest["trackerJsonFile"]
    processed = torch.load(source, map_location="cpu")
    imu_acc, imu_ori = _load_raw_imu(raw_imu_path)

    for layout_name, labels in RAW_IMU_LAYOUTS.items():
        n = min(len(processed["pose"][capture_idx]), len(next(iter(imu_acc.values()))))
        data = {
            "layout": layout_name,
            "layout_labels": LAYOUTS[layout_name]["labels"],
            "layout_joints": LAYOUTS[layout_name]["joints"],
            "acc": [torch.stack([imu_acc[x][:n] for x in labels], dim=1)],
            "ori": [torch.stack([imu_ori[x][:n] for x in labels], dim=1)],
            "pose": [processed["pose"][capture_idx][:n]],
            "tran": [processed["tran"][capture_idx][:n]],
            "shape": [processed["shape"][capture_idx]],
            "joint": [processed["joint"][capture_idx][:n]],
            "contact": [processed["contact"][capture_idx][:n]],
            "metadata": [{"source": str(capture_dir), "signal_source": "raw-imu"}],
        }
        out = output_root / layout_name / "ours.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out)

    positions, rotations = _load_tracker_feet(tracker_path, len(processed["pose"][capture_idx]))
    labels = TRACKER_FEET_LAYOUT["wrists_feet_waist"]
    n = min(len(processed["pose"][capture_idx]), len(positions["LEFT_FOOT"]))
    feet_acc = {x: _syn_acc(positions[x][:n]) for x in ("LEFT_FOOT", "RIGHT_FOOT")}
    for x in ("LEFT_HAND", "RIGHT_HAND", "WAIST"):
        feet_acc[x] = imu_acc[x][:n]
    feet_ori = {x: rotations[x][:n] for x in ("LEFT_FOOT", "RIGHT_FOOT")}
    for x in ("LEFT_HAND", "RIGHT_HAND", "WAIST"):
        feet_ori[x] = imu_ori[x][:n]
    data = {
        "layout": "wrists_feet_waist",
        "layout_labels": LAYOUTS["wrists_feet_waist"]["labels"],
        "layout_joints": LAYOUTS["wrists_feet_waist"]["joints"],
        "acc": [torch.stack([feet_acc[x] for x in labels], dim=1)],
        "ori": [torch.stack([feet_ori[x] for x in labels], dim=1)],
        "pose": [processed["pose"][capture_idx][:n]],
        "tran": [processed["tran"][capture_idx][:n]],
        "shape": [processed["shape"][capture_idx]],
        "joint": [processed["joint"][capture_idx][:n]],
        "contact": [processed["contact"][capture_idx][:n]],
        "metadata": [{"source": str(capture_dir), "signal_source": "raw-imu+tracker-feet"}],
    }
    out = output_root / "wrists_feet_waist" / "ours.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=paths.raw_ours / "1")
    parser.add_argument("--processed-ours", type=Path, default=paths.eval_dir / "ours_smpl.pt")
    parser.add_argument("--output-root", type=Path, default=paths.root_dir / "data/ours_no_head_5imu")
    args = parser.parse_args()
    _process_capture(args.capture_dir, args.processed_ours, args.output_root, 0)
    print(f"Saved six layout datasets under {args.output_root}")


if __name__ == "__main__":
    main()
