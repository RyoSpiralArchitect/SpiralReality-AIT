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


if __name__ == "__main__":
    unittest.main()
