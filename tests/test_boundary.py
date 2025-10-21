import unittest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import (
    TRAIN_TEXTS,
    teacher_segments,
)
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
        # Use the bilingual anchor samples to keep the regression focused and fast.
        self.texts = [TRAIN_TEXTS[0], TRAIN_TEXTS[1], TRAIN_TEXTS[-2], TRAIN_TEXTS[-1]]
        self.segments = teacher_segments(self.texts)

    def test_student_training_f1(self) -> None:
        ait = OnePassAIT(latent_dim=32, seed=2024)
        cfg = StudentTrainingConfig(
            lr=0.05,
            epochs=48,
            batch_size=2,
            validation_split=0.25,
            patience=6,
            hidden_dim=28,
            emb_dim=18,
            window=3,
            phase_lr=0.4,
        )
        ait.train_student(self.texts, self.segments, cfg=cfg)
        scores = []
        for text, gold in zip(self.texts, self.segments):
            pred = ait.segment_text(text)
            scores.append(segmentation_f1(text, gold, pred))
        avg_f1 = sum(scores) / len(scores)
        self.assertGreater(avg_f1, 0.3, f"average F1 too low: {avg_f1}")


if __name__ == "__main__":
    unittest.main()
