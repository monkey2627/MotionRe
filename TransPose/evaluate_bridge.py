"""Run TransPose with its native six-IMU sensor semantics on shared raw data."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


_DIR = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    for candidate in (_DIR, *_DIR.parents):
        code = candidate / "code"
        if (candidate / "benchmarks").is_dir() and (
            (candidate / "base_mobileposer").is_dir()
            or (code / "base_mobileposer").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not locate MotionRe repository root")


_ROOT = _find_repo_root()
_CODE = _ROOT / "code" if (_ROOT / "code" / "base_mobileposer").is_dir() else _ROOT
_BASE = _CODE / "base_mobileposer"
os.chdir(str(_DIR))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_BASE))

from benchmarks.standard_results import write_standard_result
from benchmarks.detailed_results import write_sequence_result, write_detailed_index
from benchmarks.video import render_comparison_video
from drift_eval_common import angle_between_rotmats
import config as transpose_config
import articulate as transpose_art


FPS = 60
AMASS_ROT = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]])
DIP_IMU_MASK = [7, 8, 11, 12, 0, 2]
AMASS_VERTEX_MASK = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
AMASS_JOINT_MASK = torch.tensor([18, 19, 4, 5, 15, 0])


def _field(sequence, name):
    return sequence[name] if isinstance(sequence, dict) else getattr(sequence, name)


def _load_model(weights: Path, device: torch.device):
    import config

    config.paths.weights_file = str(weights)
    from net import TransPoseNet

    return TransPoseNet().to(device).eval()


def _syn_acc(vertices: torch.Tensor, smooth_n: int = 4) -> torch.Tensor:
    mid = smooth_n // 2
    acc = torch.stack([
        (vertices[i] + vertices[i + 2] - 2 * vertices[i + 1]) * (FPS ** 2)
        for i in range(vertices.shape[0] - 2)
    ])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack([
            (vertices[i] + vertices[i + smooth_n * 2] - 2 * vertices[i + smooth_n])
            * (FPS ** 2) / (smooth_n ** 2)
            for i in range(vertices.shape[0] - smooth_n * 2)
        ])
    return acc


def _load_dip_raw(raw_dip_dir: Path, min_frames: int, max_frames: int):
    sequences = []
    for subject_name in ("s_09", "s_10"):
        subject_dir = raw_dip_dir / subject_name
        if not subject_dir.is_dir():
            continue
        for motion_path in sorted(subject_dir.iterdir()):
            with motion_path.open("rb") as handle:
                data = pickle.load(handle, encoding="latin1")
            acc = torch.from_numpy(data["imu_acc"][:, DIP_IMU_MASK]).float()
            ori = torch.from_numpy(data["imu_ori"][:, DIP_IMU_MASK]).float()
            pose = torch.from_numpy(data["gt"]).float()
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])
            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            length = min(len(pose), len(acc), len(ori), max_frames)
            if length < min_frames:
                continue
            if torch.isnan(acc[:length]).any() or torch.isnan(ori[:length]).any() or torch.isnan(pose[:length]).any():
                print("  skip raw DIP {}: contains NaN after fill".format(motion_path))
                continue
            sequences.append({
                "acc": acc[:length].contiguous(),
                "ori": ori[:length].contiguous(),
                "pose": transpose_art.math.axis_angle_to_rotation_matrix(
                    pose[:length]
                ).view(-1, 24, 3, 3).contiguous(),
                "tran": torch.zeros(length, 3),
                "source": "{}/{}".format(subject_name, motion_path.name),
                "action": "dip",
            })
    return sequences


def _action_selection(manifest_path: Path, raw_amass: Path, max_per_action: int) -> dict:
    selected = {}
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        grouped = {}
        for row in csv.DictReader(handle):
            category = (row.get("category") or "other").strip()
            source = (row.get("raw_motion") or "").strip()
            if not source:
                continue
            try:
                relative = Path(source).resolve().relative_to(raw_amass.resolve())
            except ValueError:
                normalized = source.replace("\\", "/")
                marker = "/AMASS/"
                if marker not in normalized:
                    continue
                relative = Path(normalized.split(marker, 1)[1])
            grouped.setdefault(category, []).append((relative.parts[0], relative.as_posix()))
    for category, entries in grouped.items():
        for dataset, source in sorted(entries)[:max_per_action]:
            selected.setdefault(dataset, set()).add((category, source))
    return selected


def _iter_amass_files(raw_amass_dir: Path):
    allowed_datasets = set(transpose_config.amass_data)
    for dataset_dir in sorted(path for path in raw_amass_dir.iterdir() if path.is_dir()):
        if dataset_dir.name not in allowed_datasets:
            continue
        pattern = str(dataset_dir / "*" / "*_poses.npz")
        for npz_name in sorted(glob.glob(pattern)):
            yield dataset_dir.name, Path(npz_name)


def _load_amass_raw(
    raw_amass_dir: Path,
    min_frames: int,
    max_frames: int,
    max_seqs: int,
    action_manifest: Path = None,
    max_per_action: int = 100,
):
    body_model = transpose_art.ParametricModel(transpose_config.paths.smpl_file)
    action_map = _action_selection(action_manifest, raw_amass_dir, max_per_action) if action_manifest else None
    sequences = []
    unlimited = max_seqs <= 0
    for dataset, npz_path in _iter_amass_files(raw_amass_dir):
        allowed = action_map.get(dataset, set()) if action_map is not None else None
        raw_rel = npz_path.relative_to(raw_amass_dir).as_posix()
        action = "all"
        if allowed is not None:
            actions = [category for category, source in allowed if source == raw_rel]
            if not actions:
                continue
            action = actions[0]
        try:
            cdata = np.load(str(npz_path))
        except Exception as exc:
            print("  skip raw AMASS {}: {}".format(npz_path, exc))
            continue
        framerate = int(cdata["mocap_framerate"])
        if framerate == 120:
            step = 2
        elif framerate in (59, 60):
            step = 1
        else:
            continue
        raw_pose = torch.tensor(cdata["poses"][::step].astype(np.float32)).view(-1, 52, 3)
        raw_tran = torch.tensor(cdata["trans"][::step].astype(np.float32))
        if len(raw_pose) <= 12 or len(raw_pose) < min_frames:
            continue
        pose = raw_pose.clone()
        pose[:, 23] = pose[:, 37]
        pose = pose[:, :24].clone()
        tran = AMASS_ROT.matmul(raw_tran.unsqueeze(-1)).view(len(raw_tran), 3)
        pose[:, 0] = transpose_art.math.rotation_matrix_to_axis_angle(
            AMASS_ROT.matmul(transpose_art.math.axis_angle_to_rotation_matrix(pose[:, 0]))
        )
        shape = torch.tensor(cdata["betas"][:10].astype(np.float32))
        pose_mat = transpose_art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        grot, _, vert = body_model.forward_kinematics(pose_mat, shape, tran, calc_mesh=True)
        length = min(len(pose_mat), max_frames)
        sequences.append({
            "acc": _syn_acc(vert[:, AMASS_VERTEX_MASK])[:length].contiguous(),
            "ori": grot[:, AMASS_JOINT_MASK][:length].contiguous(),
            "pose": pose_mat[:length].contiguous(),
            "tran": tran[:length].contiguous(),
            "source": raw_rel,
            "action": action,
        })
        if not unlimited and len(sequences) >= max_seqs:
            break
    return sequences


@torch.no_grad()
def evaluate(sequences, model, device: torch.device, max_frames: int, out_dir: Path = None):
    from utils import normalize_and_concat

    _CKPT = str(out_dir / ".eval_ckpt_transpose.npz") if out_dir else None

    rotation_sum = np.zeros((max_frames, 24), dtype=np.float64)
    translation_sum = np.zeros(max_frames, dtype=np.float64)
    count = np.zeros(max_frames, dtype=np.int64)
    start_idx = 0
    records = []

    if _CKPT and os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rotation_sum    = ck["rotation_sum"]
        translation_sum = ck["translation_sum"]
        count           = ck["count"]
        start_idx       = int(ck["seqs_done"])
        print(f"  [Resume] TransPose: {start_idx}/{len(sequences)} seqs done")

    for idx, sequence in enumerate(tqdm(sequences, desc="TransPose")):
        if idx < start_idx:
            continue
        pose = _field(sequence, "pose")
        acc = _field(sequence, "acc")
        ori = _field(sequence, "ori")
        tran = _field(sequence, "tran")
        length = min(len(pose), max_frames)
        source = _field(sequence, "source") if not isinstance(sequence, dict) else sequence.get("source", "unknown")
        try:
            model.reset()
            imu = normalize_and_concat(acc[:length], ori[:length]).to(device)
            pose_pred, tran_pred = model.forward_offline(imu)
            pose_pred, tran_pred = pose_pred.cpu(), tran_pred.cpu()
            rotation_error = angle_between_rotmats(pose_pred, pose[:length]).numpy()
            translation_error = (tran_pred - (tran[:length] - tran[:1])).norm(dim=-1).numpy()
            rotation_sum[:length] += rotation_error
            translation_sum[:length] += translation_error
            count[:length] += 1
            records.append(write_sequence_result(
                out_dir, idx, source, sequence.get("action", "other") if isinstance(sequence, dict) else "other",
                rotation_error, translation_error, FPS, configuration="full_6s"))
        except Exception as exc:
            print("\n  skip {}: {}: {}".format(source, type(exc).__name__, exc))
            records.append(write_sequence_result(
                out_dir, idx, source, sequence.get("action", "other") if isinstance(sequence, dict) else "other",
                None, None, FPS, failure_reason="{}: {}".format(type(exc).__name__, exc),
                configuration="full_6s"))
        if _CKPT:
            np.savez(_CKPT, rotation_sum=rotation_sum, translation_sum=translation_sum,
                     count=count, seqs_done=idx + 1)

    if _CKPT and os.path.exists(_CKPT):
        os.remove(_CKPT)

    valid = count > 0
    rotation = np.full_like(rotation_sum, np.nan)
    translation = np.full(max_frames, np.nan)
    rotation[valid] = rotation_sum[valid] / count[valid, None]
    translation[valid] = translation_sum[valid] / count[valid]
    return rotation, translation, count, records


def main() -> int:
    parser = argparse.ArgumentParser(description="TransPose bridge evaluation on shared raw data")
    parser.add_argument("--suite", choices=("dip", "drift"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=_BASE / "data/processed_datasets")
    parser.add_argument("--raw-root", type=Path, default=_BASE / "data/raw")
    parser.add_argument("--raw-dip-dir", type=Path, default=None)
    parser.add_argument("--raw-amass-dir", type=Path, default=None)
    parser.add_argument("--min-frames", type=int, default=300)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--action-manifest", type=Path, default=None)
    parser.add_argument("--max-per-action", type=int, default=100)
    parser.add_argument("--video-seconds", type=int, default=30)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    max_frames = args.max_seconds * FPS
    raw_dip_dir = args.raw_dip_dir or (args.raw_root / "DIP_IMU")
    raw_amass_dir = args.raw_amass_dir or (args.raw_root / "AMASS")
    if args.suite == "dip":
        sequences = _load_dip_raw(raw_dip_dir, args.min_frames, max_frames)
    else:
        sequences = _load_amass_raw(
            raw_amass_dir, args.min_frames, max_frames, args.max_seqs or 0,
            action_manifest=args.action_manifest,
            max_per_action=args.max_per_action,
        )
    if not sequences:
        raise RuntimeError("No eligible sequences found")
    device = torch.device(args.device)
    model = _load_model(args.model, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rotation, translation, count, records = evaluate(sequences, model, device, max_frames, out_dir=args.out_dir)
    output = args.out_dir / "transpose_{}.npz".format(args.suite)
    np.savez_compressed(output, rotation=rotation, translation=translation, count=count, fps=FPS)
    write_standard_result(
        args.out_dir, "transpose", args.suite, rotation, count, FPS, 6,
        [0, 1, 2, 3, 4, 5], translation,
    )
    write_detailed_index(args.out_dir, "transpose", args.suite, FPS, records)
    if not args.no_video:
        bodymodel = transpose_art.ParametricModel(str(transpose_config.paths.smpl_file))
        for index, sequence in enumerate(sequences):
            pose = _field(sequence, "pose")
            acc = _field(sequence, "acc")
            ori = _field(sequence, "ori")
            tran = _field(sequence, "tran")
            length = min(len(pose), args.video_seconds * FPS)
            model.reset()
            from utils import normalize_and_concat
            imu = normalize_and_concat(
                acc[:length], ori[:length]
            ).to(device)
            pose_pred, tran_pred = model.forward_offline(imu)
            pose_pred, tran_pred = pose_pred.cpu(), tran_pred.cpu()
            pose_fk = torch.eye(3).view(1, 1, 3, 3).expand(length, 24, 3, 3).clone()
            root_ori = ori[:length, 5]
            for sensor, joint in {0: 18, 1: 19, 2: 4, 3: 5, 5: 0}.items():
                pose_fk[:, joint] = root_ori.transpose(-1, -2) @ ori[:length, sensor]
            tran_pred = tran_pred - tran_pred[:1] + tran[:1]
            with torch.no_grad():
                _, gt_joints = bodymodel.forward_kinematics(
                    pose[:length], tran=tran[:length]
                )
                _, pred_joints = bodymodel.forward_kinematics(
                    pose_pred, tran=tran_pred
                )
                _, fk_joints = bodymodel.forward_kinematics(
                    pose_fk, tran=tran[:length]
                )
            errors = angle_between_rotmats(
                pose_pred, pose[:length]
            )[:, [1, 2, 3, 6, 9]].mean(-1).numpy()
            fk_errors = angle_between_rotmats(
                pose_fk, pose[:length]
            )[:, [1, 2, 3, 6, 9]].mean(-1).numpy()
            render_comparison_video(
                gt_joints=gt_joints.numpy(),
                method_joints=pred_joints.numpy(),
                fk_joints=fk_joints.numpy(),
                method="TransPose", combo="full_6s",
                sequence=sequence, fps=FPS, out_dir=args.out_dir,
                seq_idx=index, max_seconds=args.video_seconds,
                render_fps=args.video_fps, method_errors=errors,
                fk_errors=fk_errors,
            )
    print("Saved: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
