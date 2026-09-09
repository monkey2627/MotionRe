from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from .registry import BenchmarkOptions, CommandPlan, MethodSpec, build_plans, build_specs, get_spec, missing_requirements, status
except ImportError:
    from registry import BenchmarkOptions, CommandPlan, MethodSpec, build_plans, build_specs, get_spec, missing_requirements, status


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _parse_combos(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return ("all",)
    return tuple(values)


def _options(args: argparse.Namespace, root: Path) -> BenchmarkOptions:
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = root / out_root
    return BenchmarkOptions(
        root=root,
        out_root=out_root,
        smoke=args.smoke,
        min_frames=args.min_frames,
        max_seconds=args.max_seconds,
        max_seqs=args.max_seqs,
        max_seq=args.max_seq,
        device=args.device,
        model=args.model,
        combos=_parse_combos(args.combos),
        with_video=args.with_video,
    )


def _print_specs(specs: Iterable[MethodSpec], root: Path) -> None:
    print(f"{'NAME':<22} {'STATUS':<12} {'CATEGORY':<22} SENSORS")
    print("-" * 78)
    for spec in specs:
        print(f"{spec.name:<22} {status(spec, root):<12} {spec.category:<22} {spec.sensors}")


def _print_doctor(spec: MethodSpec, root: Path) -> None:
    current = status(spec, root)
    print(f"[{spec.name}] {spec.label}: {current}")
    if spec.working_dir is not None:
        print(f"  working_dir: {_relative(spec.working_dir, root)}")
    if spec.suites:
        print(f"  suites: {', '.join(spec.suites)}")
    missing = missing_requirements(spec, root)
    if missing:
        print("  missing:")
        for item in missing:
            print(f"    - {item}")
    if spec.notes:
        print(f"  notes: {spec.notes}")


def _write_report(plan: CommandPlan, root: Path, payload: dict) -> None:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = plan.output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {_relative(report_path, root)}")


def _run_plan(plan: CommandPlan, root: Path, dry_run: bool) -> int:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in plan.argv]
    payload = {
        "method": plan.method,
        "suite": plan.suite,
        "plan": plan.name,
        "cwd": _relative(plan.cwd, root),
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"\n[{plan.method}/{plan.suite}/{plan.name}]")
    print(f"  cwd: {_relative(plan.cwd, root)}")
    print(f"  command: {' '.join(command)}")
    if dry_run:
        payload["status"] = "dry-run"
        payload["return_code"] = None
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(plan, root, payload)
        return 0

    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=plan.cwd, check=False)
        return_code = completed.returncode
        result_status = "passed" if return_code == 0 else "failed"
    except OSError as exc:
        return_code = 127
        result_status = "error"
        payload["error"] = str(exc)
    payload["status"] = result_status
    payload["return_code"] = return_code
    payload["duration_seconds"] = round(time.monotonic() - started, 3)
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_report(plan, root, payload)
    return return_code


def _run_method(spec: MethodSpec, suite: str, options: BenchmarkOptions, dry_run: bool, allow_missing: bool) -> int:
    missing = missing_requirements(spec, options.root)
    if missing and not allow_missing:
        print(f"[{spec.name}] blocked; run doctor for missing prerequisites")
        for item in missing:
            print(f"  - {item}")
        return 2
    plans = build_plans(spec, suite, options)
    if not plans:
        print(f"[{spec.name}] has no executable adapter for suite {suite}")
        return 2
    return_codes = [_run_plan(plan, options.root, dry_run) for plan in plans]
    return max(return_codes)


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="Use short evaluation windows")
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--max-seq", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--combos", nargs="+", default=None)
    parser.add_argument("--out-root", default="benchmark_results")
    parser.add_argument("--with-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect repository motion benchmarks")
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("list", help="List all registered methods")

    doctor = subparsers.add_parser("doctor", help="Check method prerequisites")
    doctor.add_argument("--method", default=None)

    run = subparsers.add_parser("run", help="Run one method suite")
    run.add_argument("--method", required=True)
    run.add_argument("--suite", required=True)
    _add_run_options(run)

    run_all = subparsers.add_parser("run-all", help="Run one suite for all matching methods")
    run_all.add_argument("--suite", required=True)
    run_all.add_argument("--continue-on-error", action="store_true")
    _add_run_options(run_all)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = _repo_root()
    specs = build_specs(root)

    if args.action == "list":
        _print_specs(specs, root)
        return 0

    if args.action == "doctor":
        selected = specs if args.method is None else (get_spec(specs, args.method),)
        for spec in selected:
            _print_doctor(spec, root)
        return 0

    options = _options(args, root)
    if args.action == "run":
        spec = get_spec(specs, args.method)
        return _run_method(spec, args.suite, options, args.dry_run, args.allow_missing)

    if args.action == "run-all":
        matching = [spec for spec in specs if args.suite in spec.suites]
        if not matching:
            print(f"No registered methods provide suite {args.suite}")
            return 2
        failures: List[Tuple[str, int]] = []
        for spec in matching:
            code = _run_method(spec, args.suite, options, args.dry_run, args.allow_missing)
            if code:
                failures.append((spec.name, code))
                if not args.continue_on_error:
                    break
        if failures:
            print("\nFailures:")
            for name, code in failures:
                print(f"  - {name}: {code}")
            return 1
        return 0

    parser.error(f"Unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
