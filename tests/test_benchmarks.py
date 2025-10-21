import time
import unittest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig


class EncodeLatencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.texts = [
            "Bob considers two competing hypotheses while scanning for disconfirming evidence.",
            "内省によって仮説空間を保ち、外部証拠で逐次更新する。",
        ]
        self.segments = [
            ["Bob", "considers", "two", "competing", "hypotheses", "while", "scanning", "for", "disconfirming", "evidence", "."],
            ["内省", "に", "よって", "仮説", "空間", "を", "保ち", "、", "外部", "証拠", "で", "逐次", "更新", "する", "。"],
        ]

    def test_encode_latency_under_budget(self) -> None:
        ait = OnePassAIT(latent_dim=32, seed=3001)
        cfg = StudentTrainingConfig(epochs=32, batch_size=2, lr=0.05, validation_split=0.5, patience=4)
        ait.train_student(self.texts, self.segments, cfg=cfg)
        prompt = " ".join(self.texts)
        runs = 3
        start = time.perf_counter()
        for _ in range(runs):
            ait.encode(prompt)
        avg = (time.perf_counter() - start) / runs
        self.assertLess(avg, 1.0, f"encode latency too high: {avg}")


if __name__ == "__main__":
    unittest.main()
