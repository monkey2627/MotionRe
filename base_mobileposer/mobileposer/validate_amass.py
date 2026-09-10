"""Validate raw AMASS pose/translation files for test readiness.

Run from ``code/base_mobileposer``::

    python -m mobileposer.validate_amass
    python -m mobileposer.validate_amass --report data/amass_validation.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check(path: Path, root: Path) -> Dict[str, str]:
    row = {"file": str(path.relative_to(root)), "status": "ok", "reason": ""}
    try:
        with np.load(path, allow_pickle=False) as data:
            if "poses" not in data or "trans" not in data:
                row.update(status="invalid", reason="missing poses or trans")
                return row
            poses = np.asarray(data["poses"])
            trans = np.asarray(data["trans"])
            if poses.ndim not in (2, 3):
                row.update(status="invalid", reason="poses must be 2D or 3D")
            elif poses.ndim == 2 and poses.shape[1] % 3:
                row.update(status="invalid", reason="poses columns are not divisible by 3")
            elif poses.ndim == 3 and poses.shape[-1] != 3:
                row.update(status="invalid", reason="poses last dimension is not 3")
            elif trans.ndim != 2 or trans.shape[1] != 3:
                row.update(status="invalid", reason="trans must have shape [T, 3]")
            elif poses.shape[0] != trans.shape[0]:
                row.update(status="invalid", reason="poses/trans frame count mismatch")
            elif not np.isfinite(poses).all() or not np.isfinite(trans).all():
                row.update(status="invalid", reason="NaN or Inf present")
            row["frames"] = str(trans.shape[0])
            row["pose_shape"] = str(tuple(poses.shape))
            row["trans_shape"] = str(tuple(trans.shape))
            row["translation_span"] = "{:.6g}".format(float(np.linalg.norm(np.ptp(trans, axis=0))))
            row["translation_displacement"] = "{:.6g}".format(float(np.linalg.norm(trans[-1] - trans[0])))
    except Exception as exc:  # corrupt npz, permission, or unsupported object dtype
        row.update(status="invalid", reason="{}: {}".format(type(exc).__name__, exc))
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AMASS *_poses.npz files.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/AMASS")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    files = sorted(args.root.rglob("*_poses.npz"))
    rows = [_check(path, args.root) for path in files]
    counts = Counter(row["status"] for row in rows)
    varying = sum(float(row.get("translation_span", 0)) > 1e-3 for row in rows)
    print("AMASS files: {}".format(len(rows)))
    print("Valid: {}  Invalid: {}  With varying trans: {}".format(counts["ok"], counts["invalid"], varying))
    reasons = Counter(row["reason"] for row in rows if row["status"] != "ok")
    for reason, count in reasons.most_common():
        print("  {}: {}".format(count, reason))
    report = args.report or args.root.parent / "amass_validation.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    fields = ("file", "status", "reason", "frames", "pose_shape", "trans_shape", "translation_span", "translation_displacement")
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Report: {}".format(report))
    return 0 if counts["invalid"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
