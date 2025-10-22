import unittest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import (
    TRAIN_TEXTS,
    teacher_segments,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.boundary_cpp import compiled_backend_devices
from spiralreality_AIT_onepass_aifcore_integrated.integrated.boundary_julia import (
    has_julia_backend,
    julia_backend_devices,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.encoder_backends import has_external_adapter
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig


def segmentation_f1(text: str, gold_segments: list[str], predicted_segments: list[str]) -> float:
    def cuts(segments: list[str]) -> set[int]:
        idx = 0
        out = set()
        for tok in segments:
            idx += len(tok)
            out.add(idx)
        out.discard(len(text))
        return out

    gold = cuts(gold_segments)
    pred = cuts(predicted_segments)
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class BoundaryStudentIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        # Use the first bilingual anchors to keep the regression focused and fast.
        self.texts = [TRAIN_TEXTS[0], TRAIN_TEXTS[1]]
        self.segments = teacher_segments(self.texts)

    def test_student_training_f1(self) -> None:
        ait = OnePassAIT(latent_dim=24, seed=2024)
        cfg = StudentTrainingConfig(
            lr=0.05,
            epochs=8,
            batch_size=1,
            validation_split=0.5,
            patience=2,
            hidden_dim=20,
            emb_dim=14,
            window=2,
            phase_lr=0.3,
        )
        summary = ait.train_student(self.texts, self.segments, cfg=cfg)
        history = ait.student.history
        self.assertGreater(len(history), 1)
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
        self.assertGreater(ait.student.boundary_probs(self.texts[0]).shape[0], 0)
        self.assertIn("backend", summary)
        self.assertIn("encoder_backend", summary)
        self.assertIn("available_devices", summary)
        self.assertIn("encoder_devices", summary)

    def test_student_training_without_sequence_cache(self) -> None:
        ait = OnePassAIT(latent_dim=16, seed=3030)
        cfg = StudentTrainingConfig(
            lr=0.04,
            epochs=4,
            batch_size=1,
            validation_split=0.5,
            patience=2,
            cache_sequences=False,
            shuffle_train=False,
        )
        summary = ait.train_student(self.texts, self.segments, cfg=cfg)
        self.assertIn("cache_sequences", summary)
        self.assertFalse(summary["cache_sequences"])
        self.assertIn("cached_sequences", summary)
        self.assertEqual(summary["cached_sequences"], 0)
        self.assertGreaterEqual(summary.get("train_sequences", 0), 1)
        self.assertGreater(summary.get("train_tokens", 0), 0)
        self.assertIn("available_devices", summary)

    def test_encode_exposes_phase_and_gate_mask(self) -> None:
        ait = OnePassAIT(latent_dim=24, seed=2025)
        ait.train_student(
            self.texts,
            self.segments,
            cfg=StudentTrainingConfig(epochs=12, batch_size=2, validation_split=0.25, patience=3),
        )
        enc = ait.encode(self.texts[0])
        self.assertIn("phase_local", enc)
        self.assertIn("gate_mask", enc)
        self.assertEqual(enc["phase_local"].shape[0], len(self.texts[0]))
        self.assertEqual(enc["gate_mask"].shape[0], enc["gate_mask"].shape[1])
        diag = ait.gate_diagnostics()
        self.assertGreaterEqual(diag.mask_energy, 0.0)
        self.assertEqual(getattr(ait.encoder, "backend", "spectral-numpy"), "spectral-numpy")

    def test_backend_detectors_return_bool(self) -> None:
        self.assertIsInstance(has_julia_backend(), bool)
        self.assertIsInstance(has_external_adapter(), bool)
        self.assertIsInstance(compiled_backend_devices(), tuple)
        self.assertIsInstance(julia_backend_devices(), tuple)


if __name__ == "__main__":
    unittest.main()
