from pathlib import Path
import unittest

try:
    from .registry import build_specs, build_plans, BenchmarkOptions, status
except ImportError:
    from registry import build_specs, build_plans, BenchmarkOptions, status


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
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
        self.assertEqual(len(plans), 5)
        self.assertEqual([plan.name for plan in plans], ["1-sensor", "2-sensor", "3-sensor", "4-sensor", "5-sensor"])

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


if __name__ == "__main__":
    unittest.main()
