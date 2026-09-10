"""Classify rendered AMASS videos into broad motion categories.

The raw AMASS corpus does not provide one consistent action taxonomy.  This
tool therefore uses descriptive source paths first, then conservative motion
features from the associated ``*_poses.npz`` file as a fallback.  It copies
(never moves) each video once into its primary category and writes CSV files
so every automatic decision can be audited.

Run from ``code/base_mobileposer``::

    python -m mobileposer.classify_rendered_amass
    python -m mobileposer.classify_rendered_amass --dry-run
    python -m mobileposer.classify_rendered_amass --max-per-category 200
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


CATEGORY_ORDER = (
    "lying",
    "sitting",
    "walking",
    "running",
    "jumping",
    "crawling",
    "dancing",
    "sports",
    "interaction",
    "transitions",
    "standing",
    "other",
)

# More specific categories appear before generic motion words.  These terms
# intentionally match only source metadata, never pixels in the video.
KEYWORDS = {
    # Do not use generic words such as ``floor`` here: many standing/sports
    # sequences mention a floor without containing a lying posture.
    "lying": ("lie", "lying", "lay", "supine", "prone", "on back", "on stomach"),
    "sitting": ("sit", "seated", "chair", "sofa", "bench", "kneel"),
    "walking": ("walk", "stroll", "gait", "treadmill"),
    "running": ("run", "jog", "sprint"),
    "jumping": ("jump", "hop", "leap", "skip", "hurdle"),
    "crawling": ("crawl", "creep", "all fours"),
    "dancing": ("dance", "ballet", "salsa", "jazz", "hiphop"),
    "sports": (
        "basketball", "football", "soccer", "tennis", "golf", "baseball",
        "boxing", "martial", "karate", "taekwondo", "handball", "badminton",
        "volleyball", "ski", "skate", "swim", "exercise",
    ),
    "interaction": (
        "object", "lift", "carry", "throw", "catch", "push", "pull",
        "open", "close", "drink", "eat", "phone", "gesture", "punch",
    ),
    "transitions": (" to ", "stand up", "get up", "get down", "rise"),
    "standing": ("stand", "idle", "balance", "sway"),
}


def _normalise_text(path: Path) -> str:
    """Make filenames such as ``walk_01`` searchable as ordinary words."""
    return " " + re.sub(r"[^a-z0-9]+", " ", path.as_posix().lower()) + " "


def _keyword_category(source: Path) -> Optional[str]:
    text = _normalise_text(source)
    matches = []
    for category, terms in KEYWORDS.items():
        if any(" {} ".format(term.strip()) in text for term in terms):
            matches.append(category)
    if not matches:
        return None
    # A sequence explicitly describing a posture change is more useful in a
    # transition review set than in either endpoint posture set.
    if "transitions" in matches and len(matches) > 1:
        return "transitions"
    return next(category for category in CATEGORY_ORDER if category in matches)


def _motion_features(raw_path: Path) -> Optional[Tuple[float, float, float]]:
    """Return median horizontal speed, vertical excursion, and a diagnostic.

    AMASS uses Y as its nominal up axis.  The values are only a fallback for
    unlabeled files and are deliberately not used to override filename labels.
    """
    try:
        with np.load(raw_path, allow_pickle=False) as data:
            translation = np.asarray(data["trans"], dtype=np.float64)
            poses = np.asarray(data["poses"], dtype=np.float64)
            fps = float(np.asarray(data.get("mocap_framerate", 60)).reshape(-1)[0])
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if translation.ndim != 2 or translation.shape[0] < 3 or translation.shape[1] != 3:
        return None
    fps = fps if np.isfinite(fps) and fps > 0 else 60.0
    velocity = np.diff(translation, axis=0) * fps
    horizontal_speed = float(np.median(np.linalg.norm(velocity[:, (0, 2)], axis=1)))
    vertical_range = float(np.percentile(translation[:, 1], 95) - np.percentile(translation[:, 1], 5))
    if poses.ndim != 2 or poses.shape[1] < 3:
        return horizontal_speed, vertical_range, 1.0
    root = poses[:, :3]
    angles = np.linalg.norm(root, axis=1)
    # Rodrigues formula gives a rough root-orientation diagnostic.  It is
    # recorded for auditing only; it is intentionally not used to infer lying.
    axis = np.divide(root, angles[:, None], out=np.zeros_like(root), where=angles[:, None] > 1e-8)
    cos = np.cos(angles)
    upright = cos + axis[:, 1] * axis[:, 1] * (1.0 - cos)
    return horizontal_speed, vertical_range, float(np.median(np.abs(upright)))


def _feature_category(raw_path: Path) -> Tuple[str, str]:
    features = _motion_features(raw_path)
    if features is None:
        return "other", "unavailable"
    speed, vertical_range, uprightness = features
    detail = "speed={:.2f}; vertical_range={:.2f}; uprightness={:.2f}".format(
        speed, vertical_range, uprightness
    )
    # Root axis-angle is not a reliable torso-orientation estimate.  In
    # particular, arm/torso rotations can make upright motions look
    # horizontal.  Only infer locomotion from translation; leave posture
    # classes to explicit AMASS action names.
    if vertical_range > 0.22 and speed > 0.20:
        return "jumping", detail
    if speed >= 2.0:
        return "running", detail
    if speed >= 0.12:
        return "walking", detail
    return "standing", detail


def _video_files(render_root: Path) -> Iterable[Path]:
    yield from (path for path in render_root.rglob("*.mp4") if path.is_file())


def _raw_path_for_video(video: Path, render_root: Path, raw_root: Path) -> Path:
    relative = video.relative_to(render_root)
    return raw_root / relative.parent / (video.stem + "_poses.npz")


def _write_csv(path: Path, rows: List[Dict[str, str]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy rendered AMASS videos into action categories.")
    parser.add_argument(
        "--render-root", type=Path,
        default=_PROJECT_ROOT / "data/rendered/AMASS",
    )
    parser.add_argument(
        "--raw-root", type=Path,
        default=_PROJECT_ROOT / "data/raw/AMASS",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=_PROJECT_ROOT / "data/rendered/AMASS_by_action",
        help="Destination root; source videos are never modified.",
    )
    parser.add_argument("--max-per-category", type=int, default=None,
                        help="Copy at most N videos per category after deterministic sorting.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination video.")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Remove the previous AMASS_by_action classification before copying.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Classify and write reports without copying videos.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_per_category is not None and args.max_per_category <= 0:
        raise SystemExit("--max-per-category must be positive")
    if not args.render_root.is_dir():
        raise SystemExit("render root does not exist: {}".format(args.render_root))
    if args.rebuild and not args.dry_run:
        resolved_output = args.output_root.resolve()
        if resolved_output == args.render_root.resolve() or resolved_output == args.raw_root.resolve():
            raise SystemExit("refusing to rebuild a source directory")
        for category in CATEGORY_ORDER:
            category_dir = args.output_root / category
            if category_dir.is_dir():
                shutil.rmtree(category_dir)

    videos = sorted(_video_files(args.render_root))
    rows: List[Dict[str, str]] = []
    selected = Counter()
    copied = 0
    for video in videos:
        raw_path = _raw_path_for_video(video, args.render_root, args.raw_root)
        relative = video.relative_to(args.render_root)
        category = _keyword_category(relative)
        if category is None:
            category, detail = _feature_category(raw_path)
            method = "kinematic_fallback"
        else:
            detail = "matched source path keywords"
            method = "source_path_keywords"
        destination = args.output_root / category / relative
        include = args.max_per_category is None or selected[category] < args.max_per_category
        status = "not_selected_limit"
        if include:
            selected[category] += 1
            status = "dry_run" if args.dry_run else "already_exists"
            if not args.dry_run and (args.overwrite or not destination.exists()):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video, destination)
                copied += 1
                status = "copied"
        rows.append({
            "category": category,
            "method": method,
            "detail": detail,
            "status": status,
            "video": str(video),
            "raw_motion": str(raw_path),
            "destination": str(destination),
        })

    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = ("category", "method", "detail", "status", "video", "raw_motion", "destination")
    _write_csv(args.output_root / "classification_manifest.csv", rows, fields)
    summary = [
        {"category": category, "selected": str(selected[category]),
         "classified": str(sum(row["category"] == category for row in rows))}
        for category in CATEGORY_ORDER
    ]
    _write_csv(args.output_root / "summary.csv", summary, ("category", "selected", "classified"))
    print("Videos classified: {}".format(len(rows)))
    print("Videos copied: {}".format(copied))
    for entry in summary:
        print(
            "{:<12} selected={:<6} classified={}".format(
                entry["category"], entry["selected"], entry["classified"]
            )
        )
    print("Report: {}".format(args.output_root / "classification_manifest.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
