#!/usr/bin/env python3
"""Merge the six layout videos for each exercise into a 3x2 grid.

For every common second-level directory under ``results/.../layouts`` this
script creates ``before.mp4`` and ``after.mp4`` below ``<root>/merged``.
The tile order is alphabetical by layout-directory name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def read_layout_dirs(root: Path, exclude: Path | None = None) -> list[Path]:
    excluded = exclude.resolve() if exclude else None
    dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.resolve() != excluded
    )
    if len(dirs) != 6:
        raise RuntimeError(f"Expected 6 layout directories under {root}, found {len(dirs)}")
    return dirs


def common_exercises(layout_dirs: Iterable[Path]) -> list[str]:
    names = [
        {p.name for p in layout_dir.iterdir() if p.is_dir()}
        for layout_dir in layout_dirs
    ]
    return sorted(set.intersection(*names))


def open_captures(paths: list[Path]) -> tuple[list[cv2.VideoCapture], int, int, int, int]:
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(cap.isOpened() for cap in captures):
        for cap in captures:
            cap.release()
        raise RuntimeError("Could not open one or more input videos")

    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in captures]
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in captures]
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
    width, height = max(widths), max(heights)
    fps = min((fps for fps in fps_values if fps > 0), default=15.0)
    frame_count = min(
        (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures),
        default=0,
    )
    return captures, width, height, fps, frame_count


def merge_videos(paths: list[Path], output: Path, columns: int = 3, overwrite: bool = False) -> None:
    if output.exists() and not overwrite:
        print(f"  skip (already exists): {output}")
        return

    captures, tile_w, tile_h, fps, frame_count = open_captures(paths)
    rows = (len(paths) + columns - 1) // columns
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (tile_w * columns, tile_h * rows),
    )
    if not writer.isOpened():
        for cap in captures:
            cap.release()
        raise RuntimeError(f"Could not create output video: {output}")

    written = 0
    try:
        while True:
            frames = []
            for cap in captures:
                ok, frame = cap.read()
                if not ok:
                    frames = []
                    break
                if frame.shape[1] != tile_w or frame.shape[0] != tile_h:
                    frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            if len(frames) != len(captures):
                break

            canvas = np.zeros((tile_h * rows, tile_w * columns, 3), dtype=np.uint8)
            for i, frame in enumerate(frames):
                row, col = divmod(i, columns)
                canvas[row * tile_h:(row + 1) * tile_h,
                       col * tile_w:(col + 1) * tile_w] = frame
            writer.write(canvas)
            written += 1
    finally:
        for cap in captures:
            cap.release()
        writer.release()

    if written == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"No frames were written for {output}")
    print(f"  wrote {output} ({written} frames, {tile_w * columns}x{tile_h * rows}, {fps:g} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/ours_multi_finetune/layouts"),
        help="Directory containing the six layout directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory (default: <root>/merged)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing merged videos")
    args = parser.parse_args()

    output_root = args.output_root or args.root / "merged"
    layout_dirs = read_layout_dirs(args.root, output_root)
    exercises = common_exercises(layout_dirs)
    print("Layout order:", ", ".join(p.name for p in layout_dirs))
    print(f"Found {len(exercises)} common exercises")

    for exercise in exercises:
        print(f"[{exercise}]")
        for kind in ("before", "after"):
            paths = [layout / exercise / f"{kind}.mp4" for layout in layout_dirs]
            if not all(path.is_file() for path in paths):
                print(f"  skip {kind}: one or more files are missing")
                continue
            output = output_root / exercise / f"{kind}.mp4"
            merge_videos(paths, output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
