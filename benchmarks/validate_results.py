"""Validate benchmark completeness before using results in a paper table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark result completeness.")
    parser.add_argument("--results-root", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args(argv)
    failures = 0
    reports = sorted(args.results_root.rglob("benchmark_report.json"))
    report_dirs = {report_path.parent for report_path in reports}
    manifests = sorted(args.results_root.rglob("detailed_metrics_manifest.json"))
    standard_metrics = sorted(args.results_root.rglob("standard_metrics.json"))
    artifact_dirs = {path.parent for path in manifests} | {path.parent for path in standard_metrics}
    if not reports and not manifests:
        print("No benchmark_report.json or detailed_metrics_manifest.json files found")
        return 2
    for artifact_dir in sorted(artifact_dirs):
        if artifact_dir not in report_dirs:
            print("INCOMPLETE {}: missing benchmark_report.json".format(artifact_dir))
            failures += 1
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = report.get("status")
        detailed = report_path.with_name("detailed_metrics.jsonl")
        if status != "passed":
            print("INCOMPLETE {}: status={}".format(report_path.parent, status))
            failures += 1
            continue
        if report.get("suite") in ("dip", "drift") and not detailed.exists():
            print("MISSING detailed metrics {}".format(report_path.parent))
            failures += 1
            continue
        manifest_path = report_path.with_name("detailed_metrics_manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            failed_sequences = int(manifest.get("failed_sequences", 0))
            if failed_sequences:
                print("INCOMPLETE {}: failed_sequences={}".format(
                    report_path.parent, failed_sequences))
                failures += 1
                continue
        print("OK {}".format(report_path.parent))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
