"""Summarise detailed per-sequence benchmark outputs into action-level CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

try:
    from .detailed_results import MANDATORY_ACTIONS, action_summary
except ImportError:
    from detailed_results import MANDATORY_ACTIONS, action_summary


def read_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise detailed IMU benchmark results by action.")
    parser.add_argument("--results-root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = []
    for index in sorted(args.results_root.rglob("detailed_metrics.jsonl")):
        manifest = json.loads(index.with_name("detailed_metrics_manifest.json").read_text(encoding="utf-8"))
        report_path = index.with_name("benchmark_report.json")
        run_status = "unknown"
        if report_path.exists():
            run_status = json.loads(report_path.read_text(encoding="utf-8")).get("status", "unknown")
        records = read_records(index)
        standard_path = index.with_name("standard_metrics.json")
        artifact_complete = (
            standard_path.exists()
            and int(manifest.get("record_count", -1)) == len(records)
            and all(
                (
                    record.get("status") == "passed"
                    and record.get("array_file")
                    and (index.parent / str(record["array_file"])).exists()
                )
                or (
                    record.get("status") != "passed"
                    and bool(record.get("failure_reason"))
                )
                for record in records
            )
        )
        if not report_path.exists() and artifact_complete:
            # Some evaluators were run directly instead of through the
            # benchmark runner.  Keep the provenance distinction explicit,
            # while allowing complete, failure-free metric artifacts to be
            # used for numerical analysis without an expensive recomputation.
            run_status = "direct-artifacts"
        # A process can exit successfully after recovering individual sequence
        # failures.  Such a run is useful for diagnostics, but must not be
        # treated as a complete paper-table result.
        failed_sequences = sum(record.get("status") != "passed" for record in records)
        eligible_for_ranking = (
            run_status in ("passed", "direct-artifacts")
            and failed_sequences == 0
            and artifact_complete
        )
        diagnostic_only = artifact_complete and failed_sequences > 0
        if diagnostic_only:
            analysis_status = "diagnostic-only"
        elif eligible_for_ranking:
            analysis_status = "ranking"
        else:
            analysis_status = "incomplete"
        configurations = sorted({record.get("configuration") for record in records}, key=lambda value: str(value))
        for configuration in configurations:
            subset = [record for record in records if record.get("configuration") == configuration]
            for row in action_summary(subset, MANDATORY_ACTIONS):
                rows.append({"method": manifest["method"], "suite": manifest["suite"],
                             "run_status": run_status,
                             "eligible_for_ranking": eligible_for_ranking,
                             "diagnostic_only": diagnostic_only,
                             "analysis_status": analysis_status,
                             "failed_sequences_total": failed_sequences,
                             "artifact_complete": artifact_complete,
                             "configuration": configuration, **row})
    if not rows:
        raise SystemExit("No detailed_metrics.jsonl files found. Re-run evaluators with the detailed result contract.")
    output = args.output or args.results_root / "detailed_action_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote {} action rows to {}".format(len(rows), output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
