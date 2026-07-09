import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from lncfit.prompts import (
    build_essentiality_classification_prompt,
    build_essentiality_rationale_prompt,
)
from lncfit.inference import (
    essentiality_probability_from_logits,
    resolve_yes_no_token_ids,
    run_chatnt_zeroshot_classifier,
)


# ── prompts ──────────────────────────────────────────────────────────────────
def test_classification_prompt_is_yes_no_and_has_placeholders():
    p = build_essentiality_classification_prompt("K562", 3)
    assert p.count("<DNA>") == 3
    assert "Yes or No" in p
    assert "K562" in p


def test_rationale_prompt_asks_for_explanation():
    p = build_essentiality_rationale_prompt("HeLa", 1)
    assert p.count("<DNA>") == 1
    assert "explain" in p.lower()


# ── probability math ─────────────────────────────────────────────────────────
def test_probability_symmetric_logits_is_half():
    logits = np.zeros(10)
    # yes at idx 1, no at idx 2, equal logits -> 0.5
    assert essentiality_probability_from_logits(logits, [1], [2]) == pytest.approx(0.5)


def test_probability_yes_dominates():
    logits = np.zeros(10)
    logits[1] = 5.0  # yes token much larger
    logits[2] = 0.0  # no token
    p = essentiality_probability_from_logits(logits, [1], [2])
    assert p > 0.99


def test_probability_no_dominates():
    logits = np.zeros(10)
    logits[2] = 5.0  # no token much larger
    p = essentiality_probability_from_logits(logits, [1], [2])
    assert p < 0.01


def test_probability_in_unit_interval_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        logits = rng.normal(size=32)
        p = essentiality_probability_from_logits(logits, [3, 4], [5, 6])
        assert 0.0 <= p <= 1.0


def test_probability_renormalizes_over_yes_no_only():
    # A huge mass on an unrelated token must not change yes-vs-no ratio.
    logits = np.zeros(10)
    logits[0] = 100.0  # unrelated dominant token
    logits[1] = 1.0    # yes
    logits[2] = 0.0    # no
    p = essentiality_probability_from_logits(logits, [1], [2])
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), rel=1e-6)


def test_probability_nan_when_no_yes_no_mass():
    logits = np.full(10, -1e9)
    logits[0] = 100.0  # all mass on an unrelated token
    p = essentiality_probability_from_logits(logits, [1], [2])
    assert math.isnan(p)


# ── yes/no token resolution ──────────────────────────────────────────────────
def _fake_tokenizer(mapping):
    """Tokenizer whose __call__ returns .input_ids from a {text: [ids]} mapping."""
    tok = MagicMock()

    def call(text, add_special_tokens=False):
        out = MagicMock()
        out.input_ids = mapping.get(text, [])
        return out

    tok.side_effect = call
    tok.__call__ = call
    return tok


def test_resolve_yes_no_token_ids_takes_first_token_per_variant():
    tok = MagicMock()
    mapping = {
        "Yes": [11], "yes": [12], "YES": [13],
        " Yes": [21], " yes": [22], " YES": [23],
        "No": [31], "no": [32], "NO": [33],
        " No": [41], " no": [42], " NO": [43],
    }

    def call(text, add_special_tokens=False):
        r = MagicMock()
        r.input_ids = mapping.get(text, [])
        return r

    tok.side_effect = call
    yes_ids, no_ids = resolve_yes_no_token_ids(tok)
    assert set(yes_ids) == {11, 12, 13, 21, 22, 23}
    assert set(no_ids) == {31, 32, 33, 41, 42, 43}


# ── end-to-end orchestration with a fully mocked pipeline ────────────────────
def _mock_pipe(answer_logits, vocab=50, reason_text="Likely essential because ..."):
    """Build a mock ChatNT pipeline whose model.forward returns crafted logits."""
    pipe = MagicMock()

    # english tokens: [tok, tok, PAD, PAD] so first PAD is index 2 -> answer_pos = 1
    class _Ids:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    english = MagicMock()
    english.__getitem__ = lambda self, i: _Ids([7, 8, 0, 0])  # PAD id 0
    pipe.preprocess.return_value = {"english_tokens": english, "bio_tokens": MagicMock()}
    pipe.english_tokenizer.pad_token_id = 0

    # logits shape (1, seq=4, vocab); row at answer_pos=1 is answer_logits
    logits = np.zeros((1, 4, vocab))
    logits[0, 1] = answer_logits
    tensor = MagicMock()
    tensor.detach.return_value.cpu.return_value.numpy.return_value = logits
    pipe.model.return_value = {"logits": tensor}

    # yes token id 1, no token id 2 (space + bare variants map to same ids here)
    def tok_call(text, add_special_tokens=False):
        r = MagicMock()
        r.input_ids = {"Yes": [1], " Yes": [1], "yes": [1], " yes": [1], "YES": [1], " YES": [1],
                       "No": [2], " No": [2], "no": [2], " no": [2], "NO": [2], " NO": [2]}.get(text, [])
        return r

    pipe.english_tokenizer.side_effect = tok_call

    # reason generation returns a HF-style list
    pipe.return_value = [{"generated_text": reason_text}]
    return pipe


def test_classifier_returns_probability_and_reason():
    answer = np.zeros(50)
    answer[1] = 3.0  # yes >> no
    pipe = _mock_pipe(answer)
    result = run_chatnt_zeroshot_classifier("K562", ["ACGT"], pipe=pipe)
    assert 0.0 <= result["essentiality_probability"] <= 1.0
    assert result["essentiality_probability"] > 0.9
    assert result["predicted_label"] == "essential"
    assert result["model_rationale"] == "Likely essential because ..."
    assert result["cell_line"] == "K562"
    assert result["n_sequences"] == 1


def test_classifier_label_flips_below_threshold():
    answer = np.zeros(50)
    answer[2] = 3.0  # no >> yes
    pipe = _mock_pipe(answer)
    result = run_chatnt_zeroshot_classifier("K562", ["ACGT"], pipe=pipe)
    assert result["essentiality_probability"] < 0.1
    assert result["predicted_label"] == "non-essential"


def test_classifier_no_reason_skips_generation():
    answer = np.zeros(50)
    answer[1] = 1.0
    pipe = _mock_pipe(answer)
    result = run_chatnt_zeroshot_classifier("K562", ["ACGT"], with_reason=False, pipe=pipe)
    assert result["model_rationale"] is None
    assert result["rationale_prompt"] is None
    pipe.assert_not_called()  # no generation pass


def test_classifier_result_is_json_serialisable():
    import json
    answer = np.zeros(50)
    answer[1] = 2.0
    pipe = _mock_pipe(answer)
    result = run_chatnt_zeroshot_classifier("K562", ["ACGT"], pipe=pipe)
    json.dumps(result)  # must not raise
