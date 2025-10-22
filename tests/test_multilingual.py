import unittest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import (
    REFLECTIVE_LANGUAGES,
    corpus_catalog,
    corpus_license,
    teacher_segments,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.multilingual import (
    AVAILABLE_LANGUAGES,
    build_multilingual_corpus,
    language_histogram,
    language_statistics,
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

    def test_language_statistics_matches_histogram(self) -> None:
        texts, segments, tags = build_multilingual_corpus(
            languages=("ja", "de"), include_reflective=True, shuffle=False
        )
        stats = language_statistics(texts, segments, tags)
        hist = language_histogram(tags)

        self.assertEqual(set(stats.keys()), set(hist.keys()))
        for lang, info in stats.items():
            self.assertEqual(info["count"], float(hist[lang]))
            self.assertGreater(info["mean_chars"], 0)
            self.assertGreaterEqual(info["mean_tokens"], 1)
            self.assertGreater(info["mean_chars_per_token"], 0)

    def test_language_statistics_validates_lengths(self) -> None:
        with self.assertRaises(ValueError):
            language_statistics(["text"], [], [])

    def test_corpus_license_and_catalog(self) -> None:
        info = corpus_license()
        self.assertEqual(info["id"], "CC-BY-4.0")
        catalog = corpus_catalog(REFLECTIVE_LANGUAGES)
        self.assertIn("languages", catalog)
        self.assertIn("en", catalog["languages"])


if __name__ == "__main__":
    unittest.main()

