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
    if not reports:
        print("No benchmark_report.json files found")
        return 2
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
        print("OK {}".format(report_path.parent))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
