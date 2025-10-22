import unittest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import teacher_segments
from spiralreality_AIT_onepass_aifcore_integrated.integrated.multilingual import (
    AVAILABLE_LANGUAGES,
    build_multilingual_corpus,
    language_histogram,
)


class MultilingualCorpusTest(unittest.TestCase):
    def test_multilingual_builder_registers_segments(self) -> None:
        texts, segments, tags = build_multilingual_corpus(
            languages=("ja", "es"), include_reflective=False, shuffle=False
        )
        self.assertGreater(len(texts), 0)
        self.assertEqual(len(texts), len(segments))
        self.assertEqual(len(texts), len(tags))

        registered = teacher_segments(texts)
        self.assertEqual(registered, [list(seg) for seg in segments])

        hist = language_histogram(tags)
        self.assertIn("es", hist)
        self.assertIn("ja", hist)
        self.assertEqual(hist["es"], 2)
        self.assertEqual(hist["ja"], 2)

    def test_available_languages_are_supported(self) -> None:
        subset = AVAILABLE_LANGUAGES[:3]
        texts, _, tags = build_multilingual_corpus(
            languages=subset, include_reflective=True, shuffle=False
        )
        self.assertEqual(len(texts), len(tags))
        allowed = set(subset) | {"reflective"}
        self.assertTrue(all(tag in allowed for tag in tags))


if __name__ == "__main__":
    unittest.main()

