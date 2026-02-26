from __future__ import annotations

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import TRAIN_TEXTS
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


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

