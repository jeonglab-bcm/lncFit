"""Issue #56 redesign: build_training_example attaches the lncRNA transcript body
alongside the guide, and ChatNTCollator tokenizes an arbitrary number of
dna_sequences per example (not just the first)."""
from unittest.mock import MagicMock

import torch

from lncfit.prompts import build_training_example
from lncfit.screen_data import ScreenRecord
from scripts.finetune_chatnt import ChatNTCollator


def _record(target="Hum_XLOC_000001", fold_change=-1.5):
    return ScreenRecord(
        guide_id="g1", target=target, target_sequence="ACGTACGTACGTACGTACGTACG",
        cell_line="K562", day=14, replicate=0, fold_change=fold_change,
    )


def test_build_training_example_attaches_body_when_available():
    rec = _record()
    transcripts = {"Hum_XLOC_000001": "ACGT" * 50}
    prompt, dna_seqs, target = build_training_example(rec, transcripts)
    assert dna_seqs == [rec.target_sequence, transcripts["Hum_XLOC_000001"]]
    assert prompt.count("<DNA>") == 2
    assert target == "-1.5000"


def test_build_training_example_falls_back_to_guide_only_when_missing():
    rec = _record(target="Hum_XLOC_no_transcript")
    transcripts = {"Hum_XLOC_000001": "ACGT" * 50}
    prompt, dna_seqs, target = build_training_example(rec, transcripts)
    assert dna_seqs == [rec.target_sequence]
    assert prompt.count("<DNA>") == 1


def test_build_training_example_guide_only_when_no_transcripts_passed():
    rec = _record()
    prompt, dna_seqs, target = build_training_example(rec)
    assert dna_seqs == [rec.target_sequence]
    assert prompt.count("<DNA>") == 1


def _mock_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.bos_token_id = 2

    def _call(text, truncation=True, max_length=512):
        # Deterministic short token ids per character so tests are fast.
        return {"input_ids": [3 + (i % 5) for i in range(min(len(text), 5))]}

    tok.side_effect = _call
    return tok


def _mock_bio_tokenizer(seq_len=8):
    tok = MagicMock()

    def _call(seqs, return_tensors="pt", padding="max_length", truncation=True, max_length=512):
        result = MagicMock()
        result.input_ids = torch.zeros(len(seqs), seq_len, dtype=torch.long)
        return result

    tok.side_effect = _call
    return tok


def test_collator_handles_two_dna_sequences_per_example():
    collator = ChatNTCollator(
        english_tokenizer=_mock_tokenizer(),
        bio_tokenizer=_mock_bio_tokenizer(),
        max_length=32,
        bio_max_length=8,
    )
    batch = [
        {"prompt": "p1 <DNA> and <DNA>", "target": "-1.5000", "dna_sequences": ["ACGT", "ACGTACGTACGT"]},
        {"prompt": "p2 <DNA> and <DNA>", "target": "0.5000", "dna_sequences": ["TTTT", "GGGGCCCCAAAA"]},
    ]
    out = collator(batch)
    assert out["bio_tokens"].shape == (2, 2, 8)  # (batch, num_sequences, bio_max_length)
    assert out["english_tokens"].shape == (2, 32)
    assert out["labels"].shape == (2, 32)


def test_collator_handles_single_dna_sequence_per_example():
    collator = ChatNTCollator(
        english_tokenizer=_mock_tokenizer(),
        bio_tokenizer=_mock_bio_tokenizer(),
        max_length=32,
        bio_max_length=8,
    )
    batch = [
        {"prompt": "p1 <DNA>", "target": "-1.5000", "dna_sequences": ["ACGT"]},
    ]
    out = collator(batch)
    assert out["bio_tokens"].shape == (1, 1, 8)
