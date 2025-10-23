import unittest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import naive_segments


class NaiveSegmentsTest(unittest.TestCase):
    def test_coalesces_consecutive_punctuation(self) -> None:
        text = "Wait... Really?!"
        segments = naive_segments(text)
        self.assertEqual(segments, ["Wait", "...", "Really", "?!"])

    def test_preserves_intra_word_hyphen(self) -> None:
        text = "co-operate resilience"
        segments = naive_segments(text)
        self.assertIn("co-operate", segments)
        self.assertNotIn("-", segments)

    def test_dash_surrounded_by_space_is_boundary(self) -> None:
        text = "word - break"
        segments = naive_segments(text)
        self.assertEqual(segments, ["word", "-", "break"])

    def test_em_dash_between_words_remains_boundary(self) -> None:
        text = "alpha—beta"
        segments = naive_segments(text)
        self.assertEqual(segments, ["alpha", "—", "beta"])

    def test_en_dash_between_words_remains_boundary(self) -> None:
        text = "alpha–beta"
        segments = naive_segments(text)
        self.assertEqual(segments, ["alpha", "–", "beta"])

    def test_figure_dash_between_words_remains_boundary(self) -> None:
        text = "alpha‒beta"
        segments = naive_segments(text)
        self.assertEqual(segments, ["alpha", "‒", "beta"])

    def test_horizontal_bar_between_words_remains_boundary(self) -> None:
        text = "alpha―beta"
        segments = naive_segments(text)
        self.assertEqual(segments, ["alpha", "―", "beta"])

    def test_handles_multilingual_sentence_with_cjk_punctuation(self) -> None:
        text = "こんにちは、世界！"
        segments = naive_segments(text)
        self.assertEqual(segments, ["こんにちは", "、", "世界", "！"])

    def test_treats_zero_width_space_as_boundary(self) -> None:
        text = "漢字\u200bテスト"
        segments = naive_segments(text)
        self.assertEqual(segments, ["漢字", "テスト"])

    def test_preserves_unicode_hyphen_within_token(self) -> None:
        text = "state‐of‐the‐art solution"
        segments = naive_segments(text)
        self.assertIn("state‐of‐the‐art", segments)
        self.assertNotIn("‐", segments)

    def test_mixed_script_sentence_with_dash_boundary(self) -> None:
        text = "добро пожаловать—世界"
        segments = naive_segments(text)
        self.assertEqual(segments, ["добро", "пожаловать", "—", "世界"])


if __name__ == "__main__":
    unittest.main()
