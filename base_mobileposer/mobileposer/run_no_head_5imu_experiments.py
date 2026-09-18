"""
Run the six no-head 5-IMU layout experiments end to end.

Default stages:
  preprocess -> train -> combine -> infer -> evaluate

Use --fast-dev-run for a quick wiring check before launching full training.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from mobileposer.config import paths
from mobileposer.no_head_layouts import LAYOUTS, generate_dip_layout_dataset, generate_layout_dataset, synthesis_metadata, validate_cached_dataset
from mobileposer.surface_imu import DEFAULT_CONFIG


DEFAULT_LAYOUTS = list(LAYOUTS)
DEFAULT_STAGES = ["preprocess", "train", "combine", "infer", "evaluate"]


def _run(cmd, env=None):
    print("\n$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env)


def _latest_numeric_dir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not dirs:
        raise RuntimeError(f"No numeric checkpoint directory found in {root}")
    return max(dirs, key=lambda p: int(p.name))


def _write_layout_manifest(layout_name: str, run_dir: Path, processed_dir: Path, args) -> None:
    layout = LAYOUTS[layout_name]
    manifest = synthesis_metadata(args.imu_mode, args.attachment_config)
    manifest_path = run_dir / "synthesis.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Conflicting synthesis manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    text = [
        f"layout: {layout_name}",
        f"labels: {layout['labels']}",
        f"smpl_joints: {layout['joints']}",
        "slot order: 0..4 are the five learned inputs; pelvis/root is stored as slot 5 for compatibility",
        f"processed_dir: {processed_dir}",
        "",
    ]
    (run_dir / "layout.txt").write_text("\n".join(text), encoding="utf-8")


def _split_round_robin(items, n_groups):
    return [items[i::n_groups] for i in range(n_groups)]


def _run_final_evaluation(args, layout_names, processed_root: Path, run_root: Path, result_root: Path, env=None):
    data_root = processed_root
    output_root = result_root / "evaluation"
    if args.eval_dataset == "dip":
        data_root = Path(args.dip_eval_root)
        output_root = result_root / "evaluation_dip"
        for layout_name in layout_names:
            generate_dip_layout_dataset(
                layout_name,
                data_root / layout_name,
                split="test",
                overwrite=args.overwrite_eval_data,
                mode=args.imu_mode, attachment_config=args.attachment_config, mesh_chunk_size=args.mesh_chunk_size,
            )

    eval_cmd = [
        sys.executable,
        "-m",
        "mobileposer.evaluate_no_head_5imu",
        "--checkpoint-root",
        str(run_root),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--layouts",
        *layout_names,
        "--device",
        "cuda:0" if torch.cuda.is_available() else "cpu",
        "--video-count",
        str(args.video_count),
        "--max-seconds",
        str(args.video_seconds),
        "--render-fps",
        str(args.video_fps),
    ]
    if args.max_seq is not None:
        eval_cmd += ["--max-seq", str(args.max_seq)]
    if args.no_video:
        eval_cmd.append("--no-video")
    if args.videos_only:
        eval_cmd.append("--videos-only")
        eval_cmd.append("--overwrite")
    _run(eval_cmd, env=env)


def _launch_parallel_workers(args, layout_names):
    gpus = [str(gpu) for gpu in args.gpus]
    chunks = _split_round_robin(layout_names, len(gpus))
    processed_root = Path(args.processed_root)
    run_root = Path(args.run_root)
    result_root = Path(args.result_root)
    run_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    worker_stages = [stage for stage in args.stages if stage != "evaluate"]
    if not worker_stages:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[0]
        _run_final_evaluation(args, layout_names, processed_root, run_root, result_root, env=env)
        return

    procs = []
    for gpu, chunk in zip(gpus, chunks):
        if not chunk:
            continue

        log_path = run_root / f"parallel_gpu{gpu}.log"
        cmd = [
            sys.executable,
            "-m",
            "mobileposer.run_no_head_5imu_experiments",
            "--layouts",
            *chunk,
            "--stages",
            *worker_stages,
            "--processed-root",
            args.processed_root,
            "--run-root",
            args.run_root,
            "--result-root",
            args.result_root,
            "--imu-mode", args.imu_mode,
            "--attachment-config", str(args.attachment_config),
            "--mesh-chunk-size", str(args.mesh_chunk_size),
            "--dip-eval-root", args.dip_eval_root,
            "--worker-gpu",
            gpu,
        ]
        if args.overwrite_eval_data:
            cmd.append("--overwrite-eval-data")
        if args.overwrite_data:
            cmd.append("--overwrite-data")
        if args.fast_dev_run:
            cmd.append("--fast-dev-run")
        if args.max_seq is not None:
            cmd += ["--max-seq", str(args.max_seq)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"[launch] gpu={gpu} layouts={chunk} log={log_path}")
        log_file = log_path.open("w", encoding="utf-8")
        procs.append((gpu, chunk, log_path, log_file, subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)))

    failures = []
    for gpu, chunk, log_path, log_file, proc in procs:
        rc = proc.wait()
        log_file.close()
        if rc != 0:
            failures.append((gpu, chunk, log_path, rc))
        else:
            print(f"[done] gpu={gpu} layouts={chunk} log={log_path}")

    if failures:
        for gpu, chunk, log_path, rc in failures:
            print(f"[failed] gpu={gpu} rc={rc} layouts={chunk} log={log_path}")
        raise SystemExit(1)

    if "evaluate" in args.stages:
        eval_env = os.environ.copy()
        eval_env["CUDA_VISIBLE_DEVICES"] = gpus[0]
        _run_final_evaluation(args, layout_names, processed_root, run_root, result_root, env=eval_env)


def main():
    parser = argparse.ArgumentParser(description="Run no-head 5-IMU MobilePoser experiments.")
    parser.add_argument("--layouts", nargs="+", default=["all"], help=f"Layout names or all: {DEFAULT_LAYOUTS}")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES,
                        choices=DEFAULT_STAGES,
                        help="Subset of stages to run.")
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--eval-dataset", choices=["train-source", "dip"], default="train-source",
                        help="Evaluation data source. train-source uses --processed-root; dip uses DIP_IMU s_09/s_10 synthesized for each layout.")
    parser.add_argument("--dip-eval-root", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--overwrite-eval-data", action="store_true",
                        help="Regenerate eval-only data such as DIP layout test files.")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--max-seq", type=int, default=None,
                        help="Limit inference/evaluation sequences for quick comparison.")
    parser.add_argument("--video-count", type=int, default=2,
                        help="Videos per layout during evaluation (default: 2).")
    parser.add_argument("--video-seconds", type=int, default=30,
                        help="Maximum seconds per video (default: 30).")
    parser.add_argument("--video-fps", type=int, default=10,
                        help="Rendered video FPS (default: 10).")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip GT/Prediction videos during evaluation.")
    parser.add_argument("--videos-only", action="store_true",
                        help="During evaluation, only regenerate videos and leave metrics/comparison.csv unchanged.")
    parser.add_argument("--gpus", nargs="+", default=None,
                        help="Run layout groups in parallel on these GPU ids, e.g. --gpus 0 1 2.")
    parser.add_argument("--worker-gpu", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--imu-mode', choices=['joint', 'surface'], default='joint')
    parser.add_argument('--attachment-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--mesh-chunk-size', type=int, default=128)
    args = parser.parse_args()
    suffix = '_surface' if args.imu_mode == 'surface' else ''
    for attr, default in {
        'processed_root': f'data/no_head_5imu{suffix}_processed',
        'dip_eval_root': f'data/no_head_5imu{suffix}_dip_test',
        'run_root': f'checkpoints/no_head_5imu{suffix}',
        'result_root': f'results/no_head_5imu{suffix}',
    }.items():
        if getattr(args, attr) is None:
            setattr(args, attr, str(paths.root_dir / default))
    synthesis_metadata(args.imu_mode, args.attachment_config)
    if args.fast_dev_run and any(s in args.stages for s in ['combine', 'infer', 'evaluate']):
        parser.error('--fast-dev-run does not save checkpoints; use --stages preprocess train or --stages train')

    layout_names = DEFAULT_LAYOUTS if args.layouts == ["all"] else args.layouts
    unknown = [name for name in layout_names if name not in LAYOUTS]
    if unknown:
        raise ValueError(f"Unknown layout(s): {unknown}. Available: {DEFAULT_LAYOUTS}")
    if args.videos_only and args.no_video:
        raise ValueError("--videos-only cannot be used with --no-video")

    if args.gpus and len(args.gpus) > 1:
        if args.worker_gpu is not None:
            raise ValueError("--gpus should only be used by the parent process")
        _launch_parallel_workers(args, layout_names)
        return

    if args.worker_gpu is not None:
        print(f"Worker bound to CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} (requested gpu={args.worker_gpu})")

    processed_root = Path(args.processed_root)
    run_root = Path(args.run_root)
    result_root = Path(args.result_root)
    run_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    for layout_name in layout_names:
        print("\n" + "=" * 80)
        print(f"Layout: {layout_name} | labels={LAYOUTS[layout_name]['labels']} | joints={LAYOUTS[layout_name]['joints']}")
        print("=" * 80)

        processed_dir = processed_root / layout_name
        layout_run_root = run_root / layout_name
        layout_results = result_root / layout_name
        layout_run_root.mkdir(parents=True, exist_ok=True)
        layout_results.mkdir(parents=True, exist_ok=True)
        _write_layout_manifest(layout_name, layout_run_root, processed_dir, args)

        env = os.environ.copy()
        env["MOBILEPOSER_PROCESSED_DATASETS"] = str(processed_dir)
        env["MOBILEPOSER_CHECKPOINT_DIR"] = str(layout_run_root)
        env["MOBILEPOSER_TRAIN_COMBOS"] = "all_5imu"
        env.setdefault("MOBILEPOSER_DISABLE_WANDB", "1")
        env.setdefault("WANDB_MODE", "offline")

        if "preprocess" in args.stages:
            generate_layout_dataset(layout_name, processed_dir, overwrite=args.overwrite_data, mode=args.imu_mode, attachment_config=args.attachment_config, mesh_chunk_size=args.mesh_chunk_size)
            if args.imu_mode == "surface":
                for split in ("train", "test"):
                    generate_dip_layout_dataset(layout_name, Path(args.dip_eval_root) / layout_name / ("train" if split == "train" else ""), split=split, overwrite=args.overwrite_eval_data, mode=args.imu_mode, attachment_config=args.attachment_config, mesh_chunk_size=args.mesh_chunk_size)

        if not any(stage in args.stages for stage in ("train", "combine", "infer")):
            continue

        checkpoint_dir = None
        if "train" in args.stages:
            expected = synthesis_metadata(args.imu_mode, args.attachment_config)
            training_files = list(processed_dir.glob("*.pt"))
            if not training_files:
                raise RuntimeError(f"No training data: {processed_dir}")
            for data_file in training_files:
                validate_cached_dataset(data_file, expected)
            before = set(p for p in layout_run_root.iterdir() if p.is_dir())
            cmd = [sys.executable, "-m", "mobileposer.train"]
            if args.fast_dev_run:
                cmd.append("--fast-dev-run")
            _run(cmd, env=env)
            after = set(p for p in layout_run_root.iterdir() if p.is_dir())
            new_dirs = sorted(after - before, key=lambda p: int(p.name) if p.name.isdigit() else -1)
            checkpoint_dir = new_dirs[-1] if new_dirs else _latest_numeric_dir(layout_run_root)
        else:
            checkpoint_dir = _latest_numeric_dir(layout_run_root)

        if args.fast_dev_run:
            continue
        model_path = checkpoint_dir / "base_model.pth"
        if "combine" in args.stages:
            _run(
                [
                    sys.executable,
                    "-m",
                    "mobileposer.combine_weights",
                    "--checkpoint-path",
                    str(checkpoint_dir),
                ],
                env=env,
            )

        if "infer" in args.stages:
            if not model_path.exists():
                raise FileNotFoundError(f"Missing combined model: {model_path}")
            cmd = [
                sys.executable,
                "-m",
                "mobileposer.infer_mobileposer",
                "--model",
                str(model_path),
                "--data-dir",
                str(processed_dir),
                "--output-dir",
                str(layout_results),
                "--combos",
                "all",
                "--n-sensors",
                "5",
            ]
            if args.max_seq is not None:
                cmd += ["--max-seq", str(args.max_seq)]
            _run(cmd, env=env)

    if "evaluate" in args.stages:
        _run_final_evaluation(args, layout_names, processed_root, run_root, result_root)

    print("\nAll requested no-head 5-IMU experiments finished.")


if __name__ == "__main__":
    main()
