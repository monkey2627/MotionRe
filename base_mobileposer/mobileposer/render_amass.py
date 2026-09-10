"""Render raw AMASS motion files as videos for coverage screening.

The renderer uses three synchronized orthographic views of the 24-joint SMPL
body. A skeleton renderer is used instead of a triangle-mesh renderer so that
thousands of AMASS sequences can be inspected on a headless server.

Run from ``code/base_mobileposer``:

    python -m mobileposer.render_amass --pattern "*lie*" --limit 20
    python -m mobileposer.render_amass --workers 4
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from mobileposer.articulate import math as articulate_math
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths


_BODY_MODEL: Optional[ParametricModel] = None
_DEVICE = torch.device("cpu")

_MANIFEST_FIELDS = [
    "input",
    "output",
    "source",
    "status",
    "error",
    "gender",
    "source_fps",
    "source_frames",
    "rendered_frames",
    "duration_sec",
    "elapsed_sec",
]
_COMPLETED_STATUSES = {"rendered", "skipped"}

_AMASS_TO_MOBILEPOSER = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=torch.float32,
)


def _patch_numpy_legacy_aliases() -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _get_body_model() -> ParametricModel:
    global _BODY_MODEL
    if _BODY_MODEL is None:
        _patch_numpy_legacy_aliases()
        if not paths.smpl_file.exists():
            raise FileNotFoundError(
                f"SMPL model not found: {paths.smpl_file}. "
                "Place basicmodel_m.pkl there or update mobileposer/config.py."
            )
        _BODY_MODEL = ParametricModel(str(paths.smpl_file), device=_DEVICE)
    return _BODY_MODEL


def _worker_init(device: str) -> None:
    global _BODY_MODEL, _DEVICE
    _DEVICE = torch.device(device)
    _BODY_MODEL = None
    torch.set_num_threads(1)


def _sequence_output_name(input_path: Path) -> str:
    name = input_path.name
    if name.endswith("_poses.npz"):
        return name[:-len("_poses.npz")] + ".mp4"
    return input_path.stem + ".mp4"


def _matches_pattern(relative_path: Path, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    candidates = (relative_path.as_posix(), relative_path.name)
    return any(
        fnmatch.fnmatch(candidate.lower(), pattern.lower())
        for candidate in candidates
        for pattern in patterns
    )


def _as_scalar(value: np.ndarray, default: str = "unknown") -> str:
    array = np.asarray(value)
    if array.size == 0:
        return default
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bytes, np.bytes_)):
        return scalar.decode("utf-8", errors="replace")
    return str(scalar)


def _read_source_fps(data) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate"):
        if key in data:
            fps = float(np.asarray(data[key]).reshape(-1)[0])
            if np.isfinite(fps) and fps > 0:
                return fps
    raise KeyError("expected a positive 'mocap_framerate' value")


def _load_raw_sequence(
    input_path: Path,
    target_fps: float,
    max_seconds: Optional[float],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    str,
    np.ndarray,
    int,
]:
    with np.load(input_path, allow_pickle=False) as data:
        if "poses" not in data or "trans" not in data:
            raise KeyError("expected 'poses' and 'trans' arrays")

        poses_np = np.asarray(data["poses"], dtype=np.float32)
        trans_np = np.asarray(data["trans"], dtype=np.float32)
        source_fps = _read_source_fps(data)
        gender = _as_scalar(data["gender"]) if "gender" in data else "unknown"
        betas_np = (
            np.asarray(data["betas"], dtype=np.float32).reshape(-1)
            if "betas" in data
            else np.zeros(10, dtype=np.float32)
        )

    if poses_np.ndim == 2:
        if poses_np.shape[1] % 3 != 0:
            raise ValueError(f"poses has invalid shape {poses_np.shape}")
        poses_np = poses_np.reshape(poses_np.shape[0], -1, 3)
    elif poses_np.ndim != 3 or poses_np.shape[-1] != 3:
        raise ValueError(f"poses has invalid shape {poses_np.shape}")
    if trans_np.ndim != 2 or trans_np.shape[1] != 3:
        raise ValueError(f"trans has invalid shape {trans_np.shape}")
    if poses_np.shape[1] < 24:
        raise ValueError(f"expected at least 24 pose joints, got {poses_np.shape[1]}")

    source_frames = min(poses_np.shape[0], trans_np.shape[0])
    if max_seconds is not None:
        source_frames = min(
            source_frames,
            max(1, int(round(max_seconds * source_fps))),
        )
    if source_frames < 1:
        raise ValueError("sequence contains no frames")

    stride = max(1, int(round(source_fps / target_fps)))
    frame_indices = np.arange(0, source_frames, stride, dtype=np.int64)
    poses_np = poses_np[frame_indices]
    trans_np = trans_np[frame_indices]
    if not np.isfinite(poses_np).all() or not np.isfinite(trans_np).all():
        raise ValueError("poses or trans contains NaN/Inf")

    if poses_np.shape[1] > 37:
        poses_np = poses_np.copy()
        poses_np[:, 23] = poses_np[:, 37]
    poses_np = poses_np[:, :24]

    shape_np = np.zeros(10, dtype=np.float32)
    shape_np[:min(10, betas_np.shape[0])] = betas_np[:10]

    pose = torch.from_numpy(np.ascontiguousarray(poses_np)).to(_DEVICE)
    tran = torch.from_numpy(np.ascontiguousarray(trans_np)).to(_DEVICE)
    shape = torch.from_numpy(shape_np).to(_DEVICE)
    amass_rot = _AMASS_TO_MOBILEPOSER.to(_DEVICE)

    pose_rot = articulate_math.axis_angle_to_rotation_matrix(pose).view(
        pose.shape[0],
        24,
        3,
        3,
    ).contiguous()
    pose_rot[:, 0] = amass_rot.matmul(pose_rot[:, 0])
    pose_rot = pose_rot.contiguous()
    tran = amass_rot.matmul(tran.unsqueeze(-1)).squeeze(-1).contiguous()
    render_fps = source_fps / stride
    return (
        pose_rot,
        shape,
        tran,
        source_fps,
        render_fps,
        gender,
        frame_indices,
        source_frames,
    )


def _prepare_positions(
    joints: np.ndarray,
    camera: str,
) -> Tuple[np.ndarray, float, float]:
    positions = joints.astype(np.float32, copy=True)
    if camera == "track":
        pelvis_xz = positions[:, 0, [0, 2]].copy()
        positions[:, :, 0] -= pelvis_xz[:, 0:1]
        positions[:, :, 2] -= pelvis_xz[:, 1:2]
    else:
        positions[:, :, 0] -= np.median(positions[:, 0, 0])
        positions[:, :, 2] -= np.median(positions[:, 0, 2])

    floor_height = float(np.percentile(positions[:, :, 1], 1.0))
    positions[:, :, 1] -= floor_height

    ground_coordinates = positions[:, :, [0, 2]]
    horizontal_extent = float(np.percentile(np.abs(ground_coordinates), 99.5))
    horizontal_radius = max(1.2, horizontal_extent + 0.15)
    vertical_extent = float(np.percentile(positions[:, :, 1], 99.5))
    vertical_max = max(2.0, vertical_extent + 0.15)
    return positions, horizontal_radius, vertical_max


def _joint_color(joint_index: int) -> str:
    left_joints = {1, 4, 7, 10, 13, 16, 18, 20, 22}
    right_joints = {2, 5, 8, 11, 14, 17, 19, 21, 23}
    if joint_index in left_joints:
        return "#42A5F5"
    if joint_index in right_joints:
        return "#EF5350"
    return "#66BB6A"


def _setup_figure(
    width: int,
    height: int,
    horizontal_radius: float,
    vertical_max: float,
    edges: Sequence[Tuple[int, int]],
):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    axes = [
        figure.add_subplot(1, 3, 1),
        figure.add_subplot(1, 3, 2),
        figure.add_subplot(1, 3, 3),
    ]
    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.05,
        top=0.84,
        wspace=0.03,
    )
    figure.patch.set_facecolor("#101418")

    panel_specs = [
        (
            "Front: x / y",
            (0, 1),
            (-horizontal_radius, horizontal_radius),
            (-0.1, vertical_max),
        ),
        (
            "Side: z / y",
            (2, 1),
            (-horizontal_radius, horizontal_radius),
            (-0.1, vertical_max),
        ),
        (
            "Top: x / z",
            (0, 2),
            (-horizontal_radius, horizontal_radius),
            (-horizontal_radius, horizontal_radius),
        ),
    ]

    line_sets = []
    scatter_sets = []
    for axis, (title, dimensions, x_limits, y_limits) in zip(axes, panel_specs):
        axis.set_facecolor("#101418")
        axis.set_title(title, color="white", fontsize=9, pad=4)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
        if title != "Top: x / z":
            axis.axhline(0.0, color="#5A626B", linewidth=0.8, alpha=0.8)
        else:
            axis.axhline(0.0, color="#343B43", linewidth=0.6, alpha=0.8)
            axis.axvline(0.0, color="#343B43", linewidth=0.6, alpha=0.8)

        panel_lines = []
        for _, child in edges:
            line, = axis.plot(
                [],
                [],
                color=_joint_color(child),
                linewidth=2.2,
                solid_capstyle="round",
            )
            panel_lines.append(line)
        line_sets.append((panel_lines, dimensions))
        scatter = axis.scatter(
            [],
            [],
            s=13,
            c="#F5F5F5",
            edgecolors="#101418",
            linewidths=0.4,
            zorder=3,
        )
        scatter_sets.append((scatter, dimensions))

    title = figure.suptitle("", color="white", fontsize=8, y=0.98)
    return figure, line_sets, scatter_sets, title


def _open_video_writer(output_path: Path, fps: float, width: int, height: int):
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "OpenCV could not open an MP4 writer. Install a video codec/FFmpeg "
            "on the execution server or use an OpenCV build with MP4 support."
        )
    return writer


@torch.no_grad()
def _render_one(
    input_path: Path,
    output_path: Path,
    display_name: str,
    target_fps: float,
    max_seconds: Optional[float],
    width: int,
    height: int,
    camera: str,
    overwrite: bool,
) -> Dict[str, object]:
    started = time.time()
    result: Dict[str, object] = {
        "input": str(input_path),
        "output": str(output_path),
        "source": display_name,
        "status": "error",
        "error": "",
        "gender": "unknown",
        "source_fps": "",
        "source_frames": "",
        "rendered_frames": "",
        "duration_sec": "",
        "elapsed_sec": "",
    }
    if output_path.exists() and not overwrite:
        result["status"] = "skipped"
        result["error"] = "output exists"
        result["elapsed_sec"] = f"{time.time() - started:.2f}"
        return result

    partial_output_path = output_path.with_name(
        f".{output_path.stem}.rendering{output_path.suffix}"
    )
    writer = None
    figure = None
    try:
        if partial_output_path.exists():
            partial_output_path.unlink()
        (
            pose,
            shape,
            tran,
            source_fps,
            render_fps,
            gender,
            frame_indices,
            source_frames,
        ) = _load_raw_sequence(input_path, target_fps, max_seconds)
        body_model = _get_body_model()
        _, joints = body_model.forward_kinematics(
            pose,
            shape=shape,
            tran=tran,
            calc_mesh=False,
        )
        positions, horizontal_radius, vertical_max = _prepare_positions(
            joints.detach().cpu().numpy(),
            camera,
        )

        edges = [
            (parent, child)
            for child, parent in enumerate(body_model.parent)
            if parent is not None
        ]
        figure, line_sets, scatter_sets, title = _setup_figure(
            width,
            height,
            horizontal_radius,
            vertical_max,
            edges,
        )
        writer = _open_video_writer(
            partial_output_path,
            render_fps,
            width,
            height,
        )
        import cv2

        for rendered_index, source_index in enumerate(frame_indices):
            current = positions[rendered_index]
            for panel_lines, dimensions in line_sets:
                x_dimension, y_dimension = dimensions
                for line, (parent, child) in zip(panel_lines, edges):
                    line.set_data(
                        [current[parent, x_dimension], current[child, x_dimension]],
                        [current[parent, y_dimension], current[child, y_dimension]],
                    )
            for scatter, dimensions in scatter_sets:
                x_dimension, y_dimension = dimensions
                scatter.set_offsets(current[:, [x_dimension, y_dimension]])

            title.set_text(
                f"{display_name}    {gender}    "
                f"t={source_index / source_fps:.2f}s    "
                f"frame={source_index + 1}/{source_frames}    "
                "blue=left, red=right"
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            writer.write(frame)

        writer.release()
        writer = None
        os.replace(partial_output_path, output_path)
        result.update(
            {
                "status": "rendered",
                "gender": gender,
                "source_fps": f"{source_fps:.3f}",
                "source_frames": str(source_frames),
                "rendered_frames": str(len(frame_indices)),
                "duration_sec": f"{source_frames / source_fps:.3f}",
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(result["error"])
    finally:
        if writer is not None:
            writer.release()
        if figure is not None:
            import matplotlib.pyplot as plt

            plt.close(figure)
        if partial_output_path.exists():
            try:
                partial_output_path.unlink()
            except PermissionError:
                pass
        result["elapsed_sec"] = f"{time.time() - started:.2f}"
    return result


def _render_worker(arguments: Tuple[object, ...]) -> Dict[str, object]:
    return _render_one(*arguments)


def _build_jobs(
    input_root: Path,
    output_root: Path,
    patterns: Sequence[str],
    datasets: Sequence[str],
    start: int,
    limit: Optional[int],
    manifest_state: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Tuple[Path, Path, str]]:
    dataset_names = {name.lower() for name in datasets}
    jobs: List[Tuple[Path, Path, str]] = []
    for input_path in sorted(input_root.rglob("*_poses.npz")):
        relative_path = input_path.relative_to(input_root)
        if dataset_names and relative_path.parts[0].lower() not in dataset_names:
            continue
        if not _matches_pattern(relative_path, patterns):
            continue
        output_path = (
            output_root
            / relative_path.parent
            / _sequence_output_name(input_path)
        )
        state = manifest_state.get(relative_path.as_posix()) if manifest_state is not None else None
        if (
            state
            and state.get("status") in _COMPLETED_STATUSES
            and output_path.exists()
        ):
            continue
        jobs.append((input_path, output_path, relative_path.as_posix()))

    if start:
        jobs = jobs[start:]
    if limit is not None and limit > 0:
        jobs = jobs[:limit]
    return jobs


def _write_manifest(
    manifest_path: Path,
    results: Iterable[Dict[str, object]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(
                {field: result.get(field, "") for field in _MANIFEST_FIELDS}
            )
        handle.flush()
        os.fsync(handle.fileno())


def _read_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return {}

    state: Dict[str, Dict[str, str]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = (row.get("source") or "").strip()
            if source:
                state[source] = row
    return state



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render selected raw AMASS *_poses.npz files as three-view MP4s."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=paths.raw_amass,
        help=f"Raw AMASS root (default: {paths.raw_amass})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=paths.amass_render_dir,
        help=f"Video root (default: {paths.amass_render_dir})",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Only render named top-level datasets, e.g. ACCAD CMU.",
    )
    parser.add_argument(
        "--pattern",
        nargs="*",
        default=[],
        help='Filename/path glob filters, e.g. --pattern "*lie*" "*crawl*".',
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip this many matched files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render at most this many matched files; omit for all files.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="Target FPS using uniform source-frame sampling (default: 15).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Render only the first N seconds of each sequence.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Video width (default: 1200).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=420,
        help="Video height (default: 420).",
    )
    parser.add_argument(
        "--camera",
        choices=("track", "global"),
        default="track",
        help="Track the pelvis on the ground plane or show global trajectory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes; use CPU when this is greater than one.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for SMPL FK, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render existing videos instead of skipping them.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore completed entries in the manifest and scan all matches.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected files without rendering.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.target_fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("target FPS, width, and height must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.workers > 1 and args.device != "cpu":
        parser.error("use --device cpu when --workers is greater than 1")
    if not args.input_root.exists():
        parser.error(f"input root does not exist: {args.input_root}")

    manifest_path = args.output_root / "_render_manifest.csv"
    manifest_state = (
        _read_manifest(manifest_path)
        if args.resume and not args.overwrite
        else {}
    )
    jobs = _build_jobs(
        args.input_root,
        args.output_root,
        args.pattern,
        args.datasets,
        args.start,
        args.limit,
        manifest_state=manifest_state,
    )
    print(f"Input root : {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Selected   : {len(jobs)} raw AMASS sequences")
    if args.resume and not args.overwrite:
        print(f"Resume     : enabled ({len(manifest_state)} manifest entries)")
    else:
        print("Resume     : disabled")
    print(f"View       : three-view SMPL skeleton, camera={args.camera}")

    if args.dry_run:
        for index, (_, output_path, display_name) in enumerate(jobs, start=1):
            print(f"{index:05d}  {display_name} -> {output_path}")
        return 0
    if not jobs:
        print("No files matched the selection.")
        return 0

    common_arguments = (
        args.target_fps,
        args.max_seconds,
        args.width,
        args.height,
        args.camera,
        args.overwrite,
    )
    results: List[Dict[str, object]] = []

    if args.workers == 1:
        _worker_init(args.device)
        for input_path, output_path, display_name in tqdm(
            jobs,
            desc="Rendering AMASS",
        ):
            result = _render_one(
                input_path,
                output_path,
                display_name,
                *common_arguments,
            )
            results.append(result)
            _write_manifest(manifest_path, [result])
    else:
        worker_arguments = [
            (input_path, output_path, display_name, *common_arguments)
            for input_path, output_path, display_name in jobs
        ]
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(args.device,),
        ) as executor:
            future_to_arguments = {
                executor.submit(_render_worker, arguments): arguments
                for arguments in worker_arguments
            }
            for future in tqdm(
                as_completed(future_to_arguments),
                total=len(future_to_arguments),
                desc="Rendering AMASS",
            ):
                arguments = future_to_arguments[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "input": str(arguments[0]),
                        "output": str(arguments[1]),
                        "source": str(arguments[2]),
                        "status": "error",
                        "error": f"worker failure: {type(exc).__name__}: {exc}",
                    }
                results.append(result)
                _write_manifest(manifest_path, [result])

    counts: Dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    summary = ", ".join(
        f"{key}={value}" for key, value in sorted(counts.items())
    )
    print(f"Finished    : {summary}")
    print(f"Manifest    : {manifest_path}")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
