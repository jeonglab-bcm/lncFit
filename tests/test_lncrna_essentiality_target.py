"""Issue #56 v2: transcript-only fine-tuning with an RRA-derived essentiality target.

Target = (1 - rra_pvalue) if fold_change < 0 else 0, to 4 decimals; input is the
lncRNA transcript body only (no guide spacer)."""
import pytest

from lncfit.prompts import (
    build_lncrna_essentiality_prompt,
    build_lncrna_training_example,
    essentiality_score,
)
from lncfit.screen_data import LncRnaRecord


def _rec(target="Hum_XLOC_000001", cell_line="K562", rra_pvalue=0.5, fold_change=-1.0):
    return LncRnaRecord(
        target=target, cell_line=cell_line, day=14,
        rra_pvalue=rra_pvalue, fold_change=fold_change, label=0,
    )


def test_score_depletion_uses_one_minus_p():
    assert essentiality_score(0.6814, -0.28) == pytest.approx(0.3186)
    assert essentiality_score(0.0, -1.0) == pytest.approx(1.0)
    assert essentiality_score(1.0, -0.5) == pytest.approx(0.0)


def test_score_positive_fold_change_is_zero():
    # Even a highly significant enrichment is gated to 0 (not a depletion).
    assert essentiality_score(0.001, 0.266) == 0.0
    assert essentiality_score(0.5, 2.0) == 0.0


def test_score_zero_fold_change_is_zero():
    # fold_change == 0 is not a depletion (not strictly negative) -> 0.
    assert essentiality_score(0.01, 0.0) == 0.0


def test_prompt_is_transcript_only_single_placeholder():
    prompt = build_lncrna_essentiality_prompt("K562")
    assert prompt.count("<DNA>") == 1
    assert "K562" in prompt
    assert "log2" not in prompt.lower()  # no longer a log2FC prompt
    assert "between 0 and 1" in prompt


def test_training_example_transcript_only_and_4dp():
    transcripts = {"Hum_XLOC_000001": "ACGT" * 100}
    prompt, dna_seqs, target = build_lncrna_training_example(
        _rec(rra_pvalue=0.6814, fold_change=-0.28), transcripts
    )
    assert dna_seqs == ["ACGT" * 100]  # transcript only, no guide
    assert len(dna_seqs) == 1
    assert target == "0.3186"  # 4 decimals
    assert prompt.count("<DNA>") == 1


def test_training_example_positive_fc_targets_zero_string():
    transcripts = {"Hum_XLOC_000001": "ACGT" * 100}
    _, _, target = build_lncrna_training_example(
        _rec(rra_pvalue=0.001, fold_change=1.5), transcripts
    )
    assert target == "0.0000"


def test_training_example_missing_transcript_raises():
    with pytest.raises(KeyError):
        build_lncrna_training_example(_rec(target="no_such"), {"other": "ACGT"})
