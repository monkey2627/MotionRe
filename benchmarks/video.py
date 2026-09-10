"""Canonical comparison-video renderer for standard motion benchmarks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


SMPL_KINTREE = (
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5),
    (4, 7), (5, 8),
    (7, 10), (8, 11),
    (3, 6), (6, 9),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
)
LUMBAR_BONES = frozenset(((0, 3), (3, 6), (6, 9)))


def _safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")[:120] or "unknown"


def _sequence_value(sequence: object, key: str, default: object) -> object:
    if isinstance(sequence, dict):
        return sequence.get(key, default)
    return getattr(sequence, key, default)


def _canonical_combo(combo: str) -> str:
    return "_".join("hd" if part == "h" else part for part in str(combo).split("_"))


def comparison_video_path(
    out_dir: Path,
    method: str,
    combo: str,
    sequence: dict,
    seq_idx: int,
) -> Path:
    """Return the stable filename shared by every benchmark adapter."""
    action = _safe(_sequence_value(sequence, "action", "all"))
    source = _safe(_sequence_value(sequence, "source", f"seq{seq_idx:04d}"))
    return Path(out_dir) / (
        f"video_{_safe(method)}_{_canonical_combo(combo)}_"
        f"{action}_{source}_{seq_idx:04d}.mp4"
    )


def _draw_skeleton(ax, joints, color, title, lumbar_error=None, view_limits=None):
    ax.cla()
    for parent, child in SMPL_KINTREE:
        lumbar = (parent, child) in LUMBAR_BONES
        ax.plot(
            [joints[parent, 0], joints[child, 0]],
            [joints[parent, 1], joints[child, 1]],
            "-",
            color="#FF6F00" if lumbar else color,
            lw=3.5 if lumbar else 1.8,
        )
    ax.scatter(joints[:, 0], joints[:, 1], c=color, s=18, zorder=5)
    left, right, bottom, top = view_limits
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")
    label = title
    if lumbar_error is not None:
        label += f"\nlumbar: {float(lumbar_error):.1f} deg"
    ax.set_title(label, fontsize=10, color="white", pad=2, fontweight="bold")


def _normalise_camera(joint_sets: Sequence[np.ndarray]):
    arrays = [np.asarray(value, dtype=np.float64).copy() for value in joint_sets]
    all_joints = np.concatenate(arrays, axis=0)
    all_joints[:, :, 0] -= np.median(all_joints[:, 0, 0])
    all_joints[:, :, 2] -= np.median(all_joints[:, 0, 2])
    all_joints[:, :, 1] -= np.percentile(all_joints[:, :, 1], 1.0)
    horizontal_radius = max(
        1.2,
        float(np.percentile(np.abs(all_joints[:, :, [0, 2]]), 99.5)) + 0.15,
    )
    vertical_max = max(2.0, float(np.percentile(all_joints[:, :, 1], 99.5)) + 0.15)
    return np.split(all_joints, len(arrays), axis=0), (
        -horizontal_radius,
        horizontal_radius,
        -0.1,
        vertical_max,
    )


def render_comparison_video(
    *,
    gt_joints,
    method_joints,
    fk_joints,
    method: str,
    combo: str,
    sequence: dict,
    fps: int,
    out_dir: Path,
    seq_idx: int,
    max_seconds: float = 30,
    render_fps: int = 10,
    method_errors: Optional[np.ndarray] = None,
    fk_errors: Optional[np.ndarray] = None,
) -> Path:
    """Render GT, method and FK with the exact MobilePoser camera/layout."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.animation import FFMpegWriter
    import matplotlib.pyplot as plt

    gt = np.asarray(gt_joints)
    predicted = np.asarray(method_joints)
    fk = np.asarray(fk_joints)
    T = min(gt.shape[0], predicted.shape[0], fk.shape[0], int(max_seconds * fps))
    if T <= 0:
        raise ValueError("cannot render an empty sequence")
    stride = max(1, round(fps / render_fps))
    (gt, predicted, fk), view_limits = _normalise_camera(
        (gt[:T], predicted[:T], fk[:T])
    )

    out_path = comparison_video_path(out_dir, method, combo, sequence, seq_idx)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.patch.set_facecolor("#111122")
    fig.subplots_adjust(top=0.78, bottom=0.04, left=0.02, right=0.98, wspace=0.04)
    for axis in axes:
        axis.set_facecolor("#111122")

    writer = FFMpegWriter(
        fps=render_fps,
        metadata={"title": f"benchmark-{_safe(method)}-{_canonical_combo(combo)}"},
    )
    frames = range(0, T, stride)
    with writer.saving(fig, str(out_path), dpi=100):
        for t in frames:
            method_error = None if method_errors is None else method_errors[t]
            fk_error = None if fk_errors is None else fk_errors[t]
            _draw_skeleton(axes[0], gt[t], "#43A047", "Ground Truth", view_limits=view_limits)
            _draw_skeleton(
                axes[1], predicted[t], "#1E88E5",
                f"{method} ({combo})", method_error, view_limits,
            )
            _draw_skeleton(
                axes[2], fk[t], "#E53935", "FK baseline", fk_error, view_limits,
            )
            source = str(_sequence_value(sequence, "source", seq_idx))
            if len(source) > 42:
                source = source[:39] + "..."
            fig.suptitle(
                f"{method} | {_sequence_value(sequence, 'action', 'all')} | {source} | "
                f"t={t / fps:.1f}s",
                y=0.96,
                color="white",
                fontsize=13,
                fontweight="bold",
            )
            writer.grab_frame()
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path
