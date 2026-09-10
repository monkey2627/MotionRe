from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    category: str
    sensors: str
    working_dir: Optional[Path]
    suites: Tuple[str, ...]
    required_paths: Tuple[str, ...] = ()
    required_any: Tuple[Tuple[str, ...], ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkOptions:
    root: Path
    out_root: Path
    smoke: bool = False
    min_frames: Optional[int] = None
    max_seconds: Optional[float] = None
    max_seqs: Optional[int] = None
    max_seq: Optional[int] = None
    device: Optional[str] = None
    model: Optional[str] = None
    combos: Tuple[str, ...] = ("all",)
    with_video: bool = False
    action_manifest: Optional[Path] = None
    max_per_action: int = 100

    @property
    def effective_min_frames(self) -> int:
        if self.min_frames is not None:
            return self.min_frames
        return 900 if self.smoke else 1800

    @property
    def effective_max_seconds(self) -> float:
        if self.max_seconds is not None:
            return self.max_seconds
        return 30 if self.smoke else 120

    @property
    def effective_max_seqs(self) -> int:
        if self.max_seqs is not None:
            return self.max_seqs
        return 3 if self.smoke else 10

    @property
    def effective_max_seq(self) -> int:
        if self.max_seq is not None:
            return self.max_seq
        return 20 if self.smoke else 50


@dataclass(frozen=True)
class CommandPlan:
    method: str
    suite: str
    name: str
    cwd: Path
    argv: Tuple[str, ...]
    output_dir: Path


def build_specs(root: Path) -> Tuple[MethodSpec, ...]:
    # Support both repository layouts used by this project:
    # local: ``MotionRecover/code/base_mobileposer``;
    # server: ``MotionRe/base_mobileposer``.
    code = root / "code" if (root / "code" / "base_mobileposer").is_dir() else root
    base = code / "base_mobileposer"
    no_use = code / "NoUse"

    return (
        MethodSpec(
            "mobileposer",
            "MobilePoser",
            "pure-imu",
            "1-5 IMUs",
            base,
            ("dip", "drift", "sensor-sweep"),
            (),
            notes="Primary sensor-count and mobile-inference baseline.",
        ),
        MethodSpec(
            "slimevr",
            "SlimeVR FK",
            "geometric-baseline",
            "1-4 IMUs",
            code / "slimevr",
            ("dip", "drift"),
            (
                "code/slimevr/evaluate_dip.py",
                "code/slimevr/evaluate_drift.py",
                "code/base_mobileposer/data/processed_datasets/eval/dip_test.pt",
            ),
            notes="Non-learning forward-kinematics lower bound; no translation estimate.",
        ),
        MethodSpec(
            "dip-imu",
            "DIP-IMU",
            "pure-imu",
            "6 IMUs",
            no_use / "DIP-IMU",
            ("dip", "drift"),
            (
                "code/NoUse/DIP-IMU/evaluate_dip.py",
                "code/NoUse/DIP-IMU/evaluate_drift.py",
                "code/NoUse/DIP-IMU/train_and_eval/models",
            ),
            notes="Fixed six-sensor historical baseline; requires a trained TensorFlow model.",
        ),
        MethodSpec(
            "pip",
            "PIP",
            "pure-imu",
            "6 IMUs",
            no_use / "PIP",
            ("dip", "drift", "native"),
            (
                "code/NoUse/PIP/evaluate_bridge.py",
                "code/NoUse/PIP/evaluate.py",
                "code/NoUse/PIP/data/weights.pt",
                "code/NoUse/PIP/data/dataset_work/DIP_IMU/test.pt",
                "code/NoUse/PIP/models/SMPL_male.pkl",
            ),
            notes="Physics-aware six-sensor baseline; requires RBDL and prepared data.",
        ),
        MethodSpec(
            "pnp",
            "PNP",
            "pure-imu",
            "6 IMUs",
            code / "PNP",
            ("dip", "drift"),
            (
                "code/PNP/evaluate_dip.py",
                "code/PNP/evaluate_drift.py",
                "code/PNP/data/test_datasets/dipimu.pt",
                "code/PNP/models/SMPL_male.pkl",
                "code/PNP/models/physics.urdf",
                "code/PNP/data/weights/PNP/weights.pt",
            ),
            notes="Six-sensor physical/non-inertial baseline; local test data exists but weights are missing.",
        ),
        MethodSpec(
            "tip",
            "TIP",
            "pure-imu-contact",
            "6 IMUs",
            no_use / "TIP",
            ("native",),
            (
                "code/NoUse/TIP/offline_testing_simple.py",
                "code/NoUse/TIP/output/model-with-dip9and10.pt",
                "code/NoUse/TIP/data/preprocessed_DIP_IMU_v0",
            ),
            notes="Terrain and contact-aware six-sensor baseline; expected model-ready data is absent locally.",
        ),
        MethodSpec(
            "dynaip",
            "DynaIP",
            "pure-imu-dynamics",
            "Sparse IMUs",
            no_use / "DynaIP",
            ("dip", "drift"),
            (
                "code/NoUse/DynaIP/evaluate_dip.py",
                "code/NoUse/DynaIP/evaluate_drift.py",
                "code/NoUse/DynaIP/datasets/work",
                "code/NoUse/DynaIP/smpl_models/smpl_male.pkl",
            ),
            required_any=(("code/NoUse/DynaIP/weights/DynaIP.pth", "code/DynaIP/weights/DynaIP.pth"),),
            notes="Part-based dynamics baseline; code and weights are split across two directories.",
        ),
        MethodSpec(
            "transpose",
            "TransPose",
            "pure-imu-realtime",
            "6 IMUs",
            no_use / "TransPose",
            ("dip", "drift", "native"),
            (
                "code/NoUse/TransPose/evaluate_bridge.py",
                "code/NoUse/TransPose/evaluate.py",
                "code/NoUse/TransPose/data/weights.pt",
                "code/NoUse/TransPose/data/dataset_work/DIP_IMU/test.pt",
            ),
            notes="Lightweight real-time six-sensor baseline; local example IMU exists but model/data are incomplete.",
        ),
        MethodSpec(
            "globalpose",
            "GlobalPose",
            "pure-imu-global",
            "6 IMUs",
            no_use / "GlobalPose",
            ("native",),
            (
                "code/NoUse/GlobalPose/test.py",
                "code/NoUse/GlobalPose/data/weights.pt",
                "code/NoUse/GlobalPose/data/test_datasets/dipimu.pt",
            ),
            notes="Physics-based global motion baseline; official local environment is Windows/Python 3.8.",
        ),
        MethodSpec(
            "imuposer",
            "IMUPoser",
            "mobile-device",
            "Phone/watch/earbud IMUs",
            no_use / "IMUPoser",
            (),
            (
                "code/NoUse/IMUPoser/README.md",
                "code/NoUse/IMUPoser/src/imuposer",
            ),
            notes="No standalone evaluation entrypoint or trained checkpoint is present in this checkout.",
        ),
        MethodSpec(
            "imucoco",
            "IMUCoCo",
            "flexible-placement",
            "Flexible placement",
            code / "IMUCoCo",
            ("dip", "drift"),
            (
                "code/IMUCoCo/evaluate_dip.py",
                "code/IMUCoCo/evaluate_drift.py",
                "code/IMUCoCo/saved_checkpoints/imucoco_best.pth",
                "code/IMUCoCo/saved_checkpoints/poser_dtp_best.pth",
            ),
            notes="Flexible-placement baseline; code is present but checkpoints and datasets are external.",
        ),
        MethodSpec(
            "wheelposer",
            "WheelPoser",
            "non-ergonomic",
            "4 IMUs",
            no_use / "WheelPoser",
            ("native",),
            (
                "code/NoUse/WheelPoser/README.md",
                "code/NoUse/WheelPoser/checkpoints",
            ),
            notes="Wheelchair-user and seated-motion stress test; requires external dataset/checkpoints.",
        ),
        MethodSpec(
            "uip",
            "Ultra Inertial Poser",
            "extra-sensing",
            "6 IMUs + UWB",
            no_use / "UltraInertialPoser",
            ("native",),
            (
                "code/NoUse/UltraInertialPoser/modules/evaluate/evaluator.py",
                "code/NoUse/UltraInertialPoser/config/model_args.json",
                "code/NoUse/UltraInertialPoser/data/test.pt",
            ),
            notes="Optional extra-sensing comparison; not a pure-IMU baseline.",
        ),
        MethodSpec(
            "sip",
            "SIP",
            "external-pure-imu",
            "6 IMUs",
            None,
            (),
            notes="Literature-only six-sensor arbitrary-motion baseline; source code is not in this checkout.",
        ),
        MethodSpec(
            "fdip",
            "FDIP",
            "external-mobile",
            "Sparse IMUs",
            None,
            (),
            notes="External lightweight/fast deep inertial pose baseline; acquire before final mobile comparison.",
        ),
        MethodSpec(
            "diffusionposer",
            "DiffusionPoser",
            "external-flexible",
            "Arbitrary sparse sensors",
            None,
            (),
            notes="External arbitrary-sensor method; must be evaluated for actual mobile latency.",
        ),
        MethodSpec(
            "probip",
            "ProbIP",
            "external-uncertainty",
            "Sparse IMUs",
            None,
            (),
            notes="External uncertainty-aware baseline for ambiguity under sensor reduction.",
        ),
        MethodSpec(
            "grip",
            "GRIP",
            "extra-contact-sensing",
            "IMUs + pressure",
            None,
            (),
            notes="External ground-contact comparison; not pure IMU.",
        ),
        MethodSpec(
            "tof-ip",
            "ToF-IP",
            "extra-sensing",
            "IMU + ToF",
            None,
            (),
            notes="External depth-assisted comparison; not pure IMU.",
        ),
        MethodSpec(
            "baroposer",
            "BaroPoser",
            "extra-sensing",
            "IMU + barometer",
            None,
            (),
            notes="External barometer-assisted global-motion comparison.",
        ),
        MethodSpec(
            "group-inertial-poser",
            "Group Inertial Poser",
            "extra-sensing",
            "IMU + ranging",
            None,
            (),
            notes="External multi-person/ranging method; outside the pure-IMU main table.",
        ),
    )


def get_spec(specs: Iterable[MethodSpec], name: str) -> MethodSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(name)


def _path_exists(root: Path, path: str) -> bool:
    """Check both the local ``code/`` and flat server repository layouts."""
    candidate = root / path
    if candidate.exists():
        return True
    if path.startswith("code/"):
        return (root / path[len("code/"):]).exists()
    return False


def missing_requirements(
    spec: MethodSpec,
    root: Path,
    suite: Optional[str] = None,
) -> List[str]:
    if spec.name == "mobileposer" and suite == "sensor-sweep":
        base = root / "code" / "base_mobileposer"
        if not base.is_dir():
            base = root / "base_mobileposer"
        required = (
            base / "checkpoints/weights.pth",
            base / "mobileposer/infer_mobileposer.py",
        )
        return [str(path.relative_to(root)) for path in required if not path.exists()]

    # These DIP adapters only need their evaluator and the checkpoint supplied
    # by the caller.  Their drift-only asset lists include AMASS-specific
    # resources and must not block real-DIP evaluation.
    if spec.name in {"dynaip", "dip-imu"} and suite == "dip":
        script = spec.working_dir / "evaluate_dip.py"
        return [str(script.relative_to(root))] if not script.exists() else []

    if spec.name in {"pip", "transpose"} and suite in {"dip", "drift"}:
        script = spec.working_dir / "evaluate_bridge.py"
        return [str(script.relative_to(root))] if not script.exists() else []

    missing = [path for path in spec.required_paths if not _path_exists(root, path)]
    for group in spec.required_any:
        if not any(_path_exists(root, path) for path in group):
            missing.append("one of: " + ", ".join(group))
    return missing


def status(spec: MethodSpec, root: Path) -> str:
    if spec.working_dir is None:
        return "external"
    missing = missing_requirements(spec, root)
    if missing:
        return "blocked"
    if not spec.suites:
        return "local-code"
    return "ready"


def _output_dir(options: BenchmarkOptions, method: str, suite: str, suffix: str = "") -> Path:
    base = options.out_root / method / suite
    return base / suffix if suffix else base


def _common_dip_args(options: BenchmarkOptions, output_dir: Path) -> List[str]:
    return [
        "--min_frames",
        str(options.effective_min_frames),
        "--max_seconds",
        str(int(options.effective_max_seconds)),
        "--out_dir",
        str(output_dir),
    ]


def _common_drift_args(options: BenchmarkOptions, output_dir: Path) -> List[str]:
    args = [
        "--min_frames",
        str(options.effective_min_frames),
        "--max_seqs",
        str(options.effective_max_seqs),
        "--max_seconds",
        str(int(options.effective_max_seconds)),
        "--out_dir",
        str(output_dir),
    ]
    if options.action_manifest:
        args += ["--action_manifest", str(options.action_manifest),
                 "--max_per_action", str(options.max_per_action)]
        args[args.index("--max_seqs") + 1] = "0"
    if not options.with_video:
        args += ["--no_video"]
    return args


def _with_device(args: List[str], device: Optional[str]) -> List[str]:
    return args + (["--device", device] if device else [])


def build_plans(spec: MethodSpec, suite: str, options: BenchmarkOptions) -> List[CommandPlan]:
    if spec.working_dir is None:
        return []
    if suite not in spec.suites:
        raise ValueError(f"{spec.name} does not provide suite {suite}")

    cwd = spec.working_dir
    python = sys.executable
    model = options.model
    if model:
        supplied_path = Path(model)
        # A path supplied from the benchmark runner's repository root must
        # remain valid after the child process changes into a method directory.
        if not supplied_path.is_absolute():
            model = str((options.root / supplied_path).resolve())
    plans: List[CommandPlan] = []

    if spec.name == "mobileposer":
        model = model or "checkpoints/weights.pth"
        if suite == "dip":
            output = _output_dir(options, spec.name, suite)
            args = ["-m", "mobileposer.evaluate_dip", "--model", model]
            args += _common_dip_args(options, output)
            args = _with_device(args, options.device)
            plans.append(CommandPlan(spec.name, suite, "DIP-IMU", cwd, tuple([python] + args), output))
        elif suite == "drift":
            output = _output_dir(options, spec.name, suite)
            args = ["-m", "mobileposer.evaluate_drift", "--model", model]
            args += ["--combos"] + list(options.combos)
            args += _common_drift_args(options, output)
            plans.append(CommandPlan(spec.name, suite, "drift", cwd, tuple([python] + args), output))
        else:
            for count in range(1, 6):
                output = _output_dir(options, spec.name, suite, f"n{count}")
                args = [
                    "-m",
                    "mobileposer.infer_mobileposer",
                    "--model",
                    model,
                    "--n-sensors",
                    str(count),
                    "--max-seq",
                    str(options.effective_max_seq),
                    "--output-dir",
                    str(output),
                ]
                plans.append(CommandPlan(spec.name, suite, f"{count}-sensor", cwd, tuple([python] + args), output))
        return plans

    if spec.name in {"slimevr", "pnp", "imucoco"}:
        script = "evaluate_dip.py" if suite == "dip" else "evaluate_drift.py"
        output = _output_dir(options, spec.name, suite)
        args = [script]
        args += _common_dip_args(options, output) if suite == "dip" else _common_drift_args(options, output)
        if spec.name == "imucoco" and suite == "drift":
            args += ["--combos", ",".join(options.combos)]
        elif spec.name == "pnp" and suite == "drift":
            args += ["--combos"] + list(options.combos)
        if spec.name not in {"slimevr", "pnp"}:
            args = _with_device(args, options.device)
        plans.append(CommandPlan(spec.name, suite, script, cwd, tuple([python] + args), output))
        return plans

    if spec.name == "dip-imu":
        output = _output_dir(options, spec.name, suite)
        model_path = model or "train_and_eval/models"
        if suite == "dip":
            args = ["evaluate_dip.py", "--model", model_path]
            args += _common_dip_args(options, output)
            plan_name = "DIP-IMU"
        else:
            args = [
                "evaluate_drift.py",
                "--model",
                model_path,
                "--amass_dir",
                "../base_mobileposer/data/processed_datasets",
            ]
            args += _common_drift_args(options, output)
            plan_name = "drift"
        plans.append(CommandPlan(spec.name, suite, plan_name, cwd, tuple([python] + args), output))
        return plans

    if spec.name == "dynaip":
        output = _output_dir(options, spec.name, suite)
        if model:
            model_path = model
        elif (cwd / "weights" / "DynaIP.pth").exists():
            model_path = "weights/DynaIP.pth"
        else:
            model_path = "../../DynaIP/weights/DynaIP.pth"
        script = "evaluate_dip.py" if suite == "dip" else "evaluate_drift.py"
        args = [script, "--model", model_path]
        args += _common_dip_args(options, output) if suite == "dip" else _common_drift_args(options, output)
        args = _with_device(args, options.device)
        plan_name = "DIP-IMU" if suite == "dip" else "drift"
        plans.append(CommandPlan(spec.name, suite, plan_name, cwd, tuple([python] + args), output))
        return plans

    if spec.name == "pip":
        output = _output_dir(options, spec.name, suite)
        if suite in {"dip", "drift"}:
            args = [
                "evaluate_bridge.py", "--suite", suite,
                "--model", model or "data/weights.pt",
                "--min-frames", str(options.effective_min_frames),
                "--max-seconds", str(int(options.effective_max_seconds)),
                "--out-dir", str(output),
            ]
            if suite == "drift":
                args += ["--max-seqs", str(options.effective_max_seqs)]
                if options.action_manifest:
                    args += ["--action-manifest", str(options.action_manifest),
                             "--max-per-action", str(options.max_per_action)]
            plans.append(CommandPlan(spec.name, suite, suite, cwd, tuple([python] + args), output))
            return plans
        plans.append(CommandPlan(spec.name, suite, "native", cwd, (python, "evaluate.py"), output))
        return plans

    if spec.name == "tip":
        output = _output_dir(options, spec.name, suite)
        args = [
            "offline_testing_simple.py",
            "--name_contains",
            "dipimu_s_09 dipimu_s_10",
            "--ours_path_name_kin",
            "output/model-with-dip9and10.pt",
            "--with_acc_sum",
            "--test_len",
            str(int(options.effective_max_seconds * 60)),
            "--compare_gt",
            "--seed",
            "42",
            "--five_sbp",
        ]
        plans.append(CommandPlan(spec.name, suite, "native", cwd, tuple([python] + args), output))
        return plans

    if spec.name == "transpose":
        output = _output_dir(options, spec.name, suite)
        if suite in {"dip", "drift"}:
            args = [
                "evaluate_bridge.py", "--suite", suite,
                "--model", model or "data/weights.pt",
                "--min-frames", str(options.effective_min_frames),
                "--max-seconds", str(int(options.effective_max_seconds)),
                "--out-dir", str(output),
            ]
            if suite == "drift":
                args += ["--max-seqs", str(options.effective_max_seqs)]
            args = _with_device(args, options.device)
            plans.append(CommandPlan(spec.name, suite, suite, cwd, tuple([python] + args), output))
            return plans
        plans.append(CommandPlan(spec.name, suite, "native", cwd, (python, "evaluate.py"), output))
        return plans

    if spec.name == "globalpose":
        output = _output_dir(options, spec.name, suite)
        plans.append(CommandPlan(spec.name, suite, "native", cwd, (python, "test.py"), output))
        return plans

    if spec.name == "wheelposer":
        output = _output_dir(options, spec.name, suite)
        script = (
            "scripts/2. Experiments/2.2 WheelPoser_3_Stage/"
            "2.2.2 Offline Evaluation/2.2.2.1 Evaluate_WheelPoser_Leave14.py"
        )
        plans.append(CommandPlan(spec.name, suite, "leave14", cwd, (python, script), output))
        return plans

    if spec.name == "uip":
        output = _output_dir(options, spec.name, suite)
        args = [
            "modules/evaluate/evaluator.py",
            "--network",
            "UIP",
            "--ckpt_path",
            model or "data/checkpoint.pt",
            "--data_dir",
            "data/test",
            "--eval_trans",
            "--eval_save_dir",
            str(output),
            "--model_args_file",
            "config/model_args.json",
        ]
        plans.append(CommandPlan(spec.name, suite, "native", cwd, tuple([python] + args), output))
        return plans

    return plans
