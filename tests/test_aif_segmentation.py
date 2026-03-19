from __future__ import annotations

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import TRAIN_TEXTS
from spiralreality_AIT_onepass_aifcore_integrated.integrated.multilingual import build_multilingual_corpus
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import (
    OnePassAIT,
    StudentTrainingConfig,
)


def test_segment_text_with_aif_emits_policy_metadata() -> None:
    ait = OnePassAIT(latent_dim=16, seed=11)
    text = TRAIN_TEXTS[0]

    tokens = ait.segment_text(text, use_aif=True)
    assert isinstance(tokens, list)
    assert tokens, "segmentation should return at least one token"

    result = ait.segment_text(text, return_metadata=True, use_aif=True)
    assert isinstance(result, dict)
    assert "tokens" in result
    assert "chosen_policy" in result
    assert result["chosen_policy"] in ait.policies
    assert "aif" in result
    assert "candidates" in result["aif"]


def test_select_policy_aif_uses_calibrated_total() -> None:
    ait = OnePassAIT(latent_dim=16, seed=17)
    selection = ait.select_policy_aif(TRAIN_TEXTS[0])

    assert selection["chosen_policy"] in ait.policies
    assert "uncertainty_pressure" in selection
    candidates = selection["candidates"]
    assert candidates

    chosen = min(candidates, key=lambda row: float(row["calibrated_total"]))
    assert selection["chosen_policy"] == chosen["policy"]
    for candidate in candidates:
        assert "preference_distance" in candidate
        assert "policy_delta_norm" in candidate
        assert "epistemic_cap" in candidate
        assert "calibrated_epistemic" in candidate
        assert "policy_inertia" in candidate
        assert "calibrated_total" in candidate


def test_select_policy_aif_no_longer_collapses_to_seek_evidence() -> None:
    texts, segments, _ = build_multilingual_corpus(
        languages=None,
        include_reflective=True,
        shuffle=False,
        seed=5042,
    )
    train_texts = list(texts[:12])
    train_segments = [list(seg) for seg in segments[:12]]

    ait = OnePassAIT(latent_dim=32, seed=5042)
    ait.train_student(
        train_texts,
        train_segments,
        cfg=StudentTrainingConfig(
            lr=0.05,
            epochs=8,
            batch_size=2,
            validation_split=0.25,
            patience=3,
            hidden_dim=20,
            emb_dim=14,
            window=2,
            phase_lr=0.3,
            cache_sequences=False,
            shuffle_train=False,
        ),
    )

    chosen = [ait.select_policy_aif(text)["chosen_policy"] for text in train_texts]
    assert any(policy != "SeekEvidence" for policy in chosen)
