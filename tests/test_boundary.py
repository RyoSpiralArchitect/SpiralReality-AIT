import unittest

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
        self.texts = [
            "Bob re-examined Carol's motives and updated his provisional evaluation.",
            "Avoid premature closure; maintain hypotheses and update them with evidence.",
            "ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。",
            "結論を急ぎ過ぎないこと。内部対話で仮説を維持し、証拠で更新する。",
        ]
        self.segments = [
            ["Bob", "re", "-", "examined", "Carol's", "motives", "and", "updated", "his", "provisional", "evaluation", "."],
            ["Avoid", "premature", "closure", ";", "maintain", "hypotheses", "and", "update", "them", "with", "evidence", "."],
            ["ボブ", "は", "キャロル", "の", "動機", "を", "再検討", "し", "、", "第三者", "の", "証拠", "で", "暫定", "評価", "を", "更新", "した", "。"],
            ["結論", "を", "急ぎ過ぎ", "ない", "こと", "。", "内部", "対話", "で", "仮説", "を", "維持", "し", "、", "証拠", "で", "更新", "する", "。"],
        ]

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
