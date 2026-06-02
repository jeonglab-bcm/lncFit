import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_chatnt import build_essentiality_prompt, parse_log2fc


def test_single_placeholder():
    prompt = build_essentiality_prompt("K562", 1)
    assert prompt.count("<DNA>") == 1


def test_multiple_placeholders():
    prompt = build_essentiality_prompt("K562", 3)
    assert prompt.count("<DNA>") == 3


def test_cell_line_as_plain_text():
    prompt = build_essentiality_prompt("HeLa", 1)
    assert "HeLa" in prompt
    assert "<HeLa>" not in prompt
    assert "[HeLa]" not in prompt


def test_parse_log2fc_positive():
    assert parse_log2fc("The predicted log2FC is 1.23") == pytest.approx(1.23)


def test_parse_log2fc_negative():
    assert parse_log2fc("Result: -0.87") == pytest.approx(-0.87)


def test_parse_log2fc_integer():
    assert parse_log2fc("log2FC: 2") == pytest.approx(2.0)


def test_parse_log2fc_no_value():
    assert parse_log2fc("No numeric prediction available.") is None


def test_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["run_chatnt.py", "--cell-line", "K562", "--dna-sequence", "ACGT", "--dry-run"],
    )
    from scripts.run_chatnt import main
    main()
    captured = capsys.readouterr()
    assert "<DNA>" in captured.out
    assert "dry-run" in captured.out
