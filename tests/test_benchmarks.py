import unittest
import tempfile
import os

from spiralreality_AIT_onepass_aifcore_integrated.integrated.benchmark import run_benchmark


class BenchmarkPipelineTest(unittest.TestCase):
    def test_benchmark_generates_metrics_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = run_benchmark(
                languages=("es",),
                include_reflective=True,
                max_samples=4,
                output_dir=tmpdir,
                seed=101,
            )

            self.assertIn("baseline", metrics)
            self.assertIn("variants", metrics)
            self.assertIn("dataset", metrics)
            baseline = metrics["baseline"]
            self.assertIn("f1", baseline)
            latency = baseline["latency_ms"]
            self.assertGreater(latency["p95"], 0.0)
            self.assertLess(latency["p95"], 2000.0)

            variants = metrics["variants"]
            self.assertIn("noise", variants)
            self.assertIn("dialect", variants)
            self.assertIn("tempo_slow", variants)
            self.assertIn("tempo_fast", variants)

            json_path = os.path.join(tmpdir, "benchmark_report.json")
            md_path = os.path.join(tmpdir, "benchmark_report.md")
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))

            np_metrics = metrics["np_stub"]
            if np_metrics["available"]:
                self.assertGreaterEqual(np_metrics["linf"], 0.0)
                self.assertGreaterEqual(np_metrics["mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
