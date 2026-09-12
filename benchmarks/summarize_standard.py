"""Build comparison tables exclusively from canonical benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from standard_results import JOINT_GROUPS


def _read_result(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    arrays = np.load(path.with_name("standard_metrics.npz"))
    rotation = arrays["rotation_deg"]
    translation = arrays["translation_m"]
    count = arrays["frame_count"]
    valid = count > 0
    row = {
        "method": summary["method"],
        "suite": summary["suite"],
        "sensor_count": summary["sensor_count"],
        "evaluated_frames": int(valid.sum()),
        "all_rotation_deg": float(np.nanmean(rotation[valid][:, JOINT_GROUPS["all"]])),
        "lumbar_rotation_deg": float(np.nanmean(rotation[valid][:, JOINT_GROUPS["lumbar"]])),
        "translation_m": "",
        "run_status": "unknown",
        "failed_sequences_total": "",
        "eligible_for_ranking": False,
        "diagnostic_only": False,
        "analysis_status": "incomplete",
        "artifact_complete": False,
    }
    report_path = path.with_name("benchmark_report.json")
    manifest_path = path.with_name("detailed_metrics_manifest.json")
    detailed_path = path.with_name("detailed_metrics.jsonl")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row["run_status"] = report.get("status", "unknown")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failed_sequences = int(manifest.get("failed_sequences", 0))
        row["failed_sequences_total"] = failed_sequences
        records = []
        if detailed_path.exists():
            with detailed_path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        row["artifact_complete"] = (
            detailed_path.exists()
            and int(manifest.get("record_count", -1)) == len(records)
            and bool(records)
            and all(
                (
                    record.get("status") == "passed"
                    and record.get("array_file")
                    and (path.parent / str(record["array_file"])).exists()
                )
                or (
                    record.get("status") != "passed"
                    and bool(record.get("failure_reason"))
                )
                for record in records
            )
        )
        if not report_path.exists() and row["artifact_complete"]:
            row["run_status"] = "direct-artifacts"
        row["eligible_for_ranking"] = (
            row["run_status"] in ("passed", "direct-artifacts")
            and row["artifact_complete"]
            and failed_sequences == 0
        )
        row["diagnostic_only"] = row["artifact_complete"] and failed_sequences > 0
        row["analysis_status"] = (
            "diagnostic-only" if row["diagnostic_only"]
            else "ranking" if row["eligible_for_ranking"]
            else "incomplete"
        )
    if np.isfinite(translation[valid]).any():
        row["translation_m"] = "{:.6f}".format(float(np.nanmean(translation[valid])))
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize standard benchmark metrics.")
    parser.add_argument("--results-root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--suite", choices=("dip", "drift"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = []
    for path in sorted(args.results_root.rglob("standard_metrics.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("suite") == args.suite:
            rows.append(_read_result(path))
    if not rows:
        raise SystemExit("No standard metrics found for suite {}".format(args.suite))
    rows.sort(key=lambda row: row["method"])
    output = args.output or args.results_root / "{}_summary.csv".format(args.suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "method", "suite", "run_status", "eligible_for_ranking",
        "diagnostic_only", "analysis_status", "failed_sequences_total",
        "artifact_complete", "sensor_count", "evaluated_frames",
        "all_rotation_deg", "lumbar_rotation_deg", "translation_m",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote {} method rows to {}".format(len(rows), output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
