from pathlib import Path
import unittest

try:
    from .registry import build_specs, build_plans, BenchmarkOptions, status
    from .video import comparison_video_path
except ImportError:
    from registry import build_specs, build_plans, BenchmarkOptions, status
    from video import comparison_video_path


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script_root = Path(__file__).resolve().parents[1]
        cls.root = script_root.parent if (script_root / "code").is_dir() else script_root
        cls.specs = build_specs(cls.root)

    def test_method_names_are_unique(self):
        names = [spec.name for spec in self.specs]
        self.assertEqual(len(names), len(set(names)))

    def test_local_working_directories_exist(self):
        for spec in self.specs:
            if spec.working_dir is not None:
                self.assertTrue(spec.working_dir.exists(), spec.name)

    def test_mobileposer_smoke_plans_cover_all_sensor_counts(self):
        spec = next(spec for spec in self.specs if spec.name == "mobileposer")
        options = BenchmarkOptions(self.root, self.root / "benchmark_results", smoke=True)
        plans = build_plans(spec, "sensor-sweep", options)
        self.assertEqual(len(plans), 3)
        self.assertEqual([plan.name for plan in plans], ["4-sensor", "5-sensor", "6-sensor"])
        self.assertTrue(all(any("evaluate_dip" in argument for argument in plan.argv) for plan in plans))

    def test_external_methods_are_not_marked_ready(self):
        for spec in self.specs:
            if spec.working_dir is None:
                self.assertEqual(status(spec, self.root), "external")

    def test_drift_adapters_use_supported_common_options(self):
        options = BenchmarkOptions(
            self.root,
            self.root / "benchmark_results",
            smoke=True,
            device="cpu",
            with_video=False,
        )

        dynaip = next(spec for spec in self.specs if spec.name == "dynaip")
        dynaip_args = build_plans(dynaip, "drift", options)[0].argv
        self.assertIn("--out_dir", dynaip_args)
        self.assertIn("--no_video", dynaip_args)

        dip_imu = next(spec for spec in self.specs if spec.name == "dip-imu")
        dip_imu_args = build_plans(dip_imu, "drift", options)[0].argv
        self.assertIn("--no_video", dip_imu_args)

        pnp = next(spec for spec in self.specs if spec.name == "pnp")
        pnp_args = build_plans(pnp, "drift", options)[0].argv
        self.assertNotIn("--device", pnp_args)

        imucoco = next(spec for spec in self.specs if spec.name == "imucoco")
        imucoco_args = build_plans(imucoco, "drift", options)[0].argv
        self.assertIn("--out_dir", imucoco_args)
        self.assertIn("--no_video", imucoco_args)

    def test_bridge_drift_uses_action_manifest_coverage_options(self):
        options = BenchmarkOptions(
            self.root,
            self.root / "benchmark_results",
            smoke=True,
            with_video=False,
            action_manifest=self.root / "base_mobileposer/data/classification_manifest.csv",
        )
        for method in ("pip", "transpose"):
            spec = next(spec for spec in self.specs if spec.name == method)
            args = build_plans(spec, "drift", options)[0].argv
            self.assertEqual(args[args.index("--min-frames") + 1], "1")
            self.assertEqual(args[args.index("--max-seqs") + 1], "0")
            self.assertIn("--action-manifest", args)
            self.assertIn("--no-video", args)

    def test_comparison_video_path_is_canonical(self):
        path = comparison_video_path(
            self.root / "out",
            "DIP-IMU",
            "full_6s",
            {"action": "turn/walk", "source": "dataset[3]"},
            7,
        )
        self.assertEqual(path.name, "video_DIP-IMU_full_6s_turn_walk_dataset_3_0007.mp4")


if __name__ == "__main__":
    unittest.main()
