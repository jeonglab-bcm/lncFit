import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lncfit.screen_data import ScreenRecord
from lncfit.prompts import build_training_example
from lncfit.parsers import parse_log2fc
from scripts.probe_prompt_to_json import build_probe_record


def _make_record(**kwargs) -> ScreenRecord:
    defaults = dict(
        guide_id="g1",
        target="LINC00001",
        target_sequence="ACGTACGTACGTACGTACGTACG",
        cell_line="K562",
        day=14,
        replicate=1,
        fold_change=-0.87,
    )
    defaults.update(kwargs)
    return ScreenRecord(**defaults)


# --- build_training_example ---

def test_build_training_example_single_dna():
    prompt, dna_sequences, _ = build_training_example(_make_record())
    assert prompt.count("<DNA>") == 1
    assert len(dna_sequences) == 1


def test_build_training_example_dna_matches_target_sequence():
    record = _make_record(target_sequence="TTTTACGTACGTACGTACGTACG")
    _, dna_sequences, _ = build_training_example(record)
    assert dna_sequences[0] == "TTTTACGTACGTACGTACGTACG"


def test_build_training_example_target_is_float_string():
    _, _, target_str = build_training_example(_make_record(fold_change=-0.87))
    assert float(target_str) == pytest.approx(-0.87)


# --- build_probe_record (dry-run JSON builder) ---

def test_dry_run_probe_record_has_expected_keys():
    record = _make_record()
    prompt, _, _ = build_training_example(record)
    probe = build_probe_record(record, prompt)
    assert set(probe.keys()) == {
        "cell_line", "guide_sequence", "prompt",
        "raw_response", "parsed_log2fc", "expected_log2fc", "parse_ok",
    }


def test_dry_run_probe_record_types():
    record = _make_record()
    prompt, _, _ = build_training_example(record)
    probe = build_probe_record(record, prompt)
    assert isinstance(probe["cell_line"], str)
    assert isinstance(probe["guide_sequence"], str)
    assert isinstance(probe["prompt"], str)
    assert isinstance(probe["expected_log2fc"], float)
    assert probe["raw_response"] is None
    assert probe["parsed_log2fc"] is None
    assert probe["parse_ok"] is False


def test_live_probe_record_types():
    record = _make_record()
    prompt, _, _ = build_training_example(record)
    probe = build_probe_record(record, prompt, raw_response="The log2FC is -1.23", parsed_log2fc=-1.23)
    assert isinstance(probe["raw_response"], str)
    assert isinstance(probe["parsed_log2fc"], float)
    assert probe["parse_ok"] is True


# --- parse_log2fc round-trips for ChatNT-style responses ---

def test_parse_clean_number():
    assert parse_log2fc("-1.23") == pytest.approx(-1.23)


def test_parse_number_in_sentence():
    assert parse_log2fc("The predicted log2FC value is -1.23 for this lncRNA.") == pytest.approx(-1.23)


def test_parse_unparseable_returns_none():
    assert parse_log2fc("I cannot predict a log2FC for this sequence.") is None


# --- JSON schema / serializability ---

def test_probe_record_json_serializable():
    record = _make_record()
    prompt, _, _ = build_training_example(record)
    probe = build_probe_record(record, prompt)
    restored = json.loads(json.dumps(probe))
    assert restored["cell_line"] == "K562"
    assert restored["expected_log2fc"] == pytest.approx(-0.87)
    assert restored["parse_ok"] is False
    assert restored["raw_response"] is None


def test_probe_record_cell_line_matches_record():
    record = _make_record(cell_line="THP1")
    prompt, _, _ = build_training_example(record)
    probe = build_probe_record(record, prompt)
    assert probe["cell_line"] == "THP1"


def test_probe_record_guide_sequence_matches_record():
    record = _make_record(target_sequence="GGGGACGTACGTACGTACGTACG")
    prompt, _, _ = build_training_example(record)
    probe = build_probe_record(record, prompt)
    assert probe["guide_sequence"] == "GGGGACGTACGTACGTACGTACG"
