import json

import pytest

from lncfit.pipeline import LncRnaPipeline
from lncfit.screen_data import LncRnaRecord, save_jsonl

_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]


def _synthetic_records(n_targets, offset=0):
    """n_targets lncRNAs x 5 cell lines, ~20% hit rate, 3 chromosomes."""
    records = []
    for i in range(n_targets):
        target = f"T{offset + i}"
        label = 1 if i % 5 == 0 else 0
        chrom = str(i % 3)
        for cell_line in _CELL_LINES:
            records.append(LncRnaRecord(
                target=target, cell_line=cell_line, day=14,
                rra_pvalue=0.01 if label else 0.5, fold_change=-1.0 if label else 0.1,
                label=label, chrom=chrom,
            ))
    return records


def _synthetic_transcript_sequences(n_targets, offset=0):
    bases = "ACGT"
    return {
        f"T{offset + i}": "".join(bases[(i * 7 + j * 3) % 4] for j in range(40))
        for i in range(n_targets)
    }


@pytest.fixture
def data_files(tmp_path):
    train_records = _synthetic_records(30, offset=0)
    test_records = _synthetic_records(10, offset=30)
    train_path = tmp_path / "train.jsonl.gz"
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(train_records, train_path)
    save_jsonl(test_records, test_path)

    seqs = {**_synthetic_transcript_sequences(30, 0), **_synthetic_transcript_sequences(10, 30)}
    seqs_path = tmp_path / "transcript_sequences.json"
    with open(seqs_path, "w") as fh:
        json.dump({gid: [seq, ""] for gid, seq in seqs.items()}, fh)

    return {"train": str(train_path), "test": str(test_path), "transcript_sequences": str(seqs_path)}


def _base_config(data_files, tmp_path, **overrides):
    config = {
        "data": data_files,
        "features": {"type": "kmer", "k": 3, "cell_embedding_dim": 0},
        "model": {"name": "logreg", "params": {"C": 1.0}},
        "tuning": {"method": "fixed"},
        "cv": {"strategy": "none"},
        "seed": 42,
        "output_dir": str(tmp_path / "runs"),
    }
    config.update(overrides)
    return config


def test_fixed_run_end_to_end(data_files, tmp_path):
    config = _base_config(data_files, tmp_path, cv={"strategy": "stratified", "n_splits": 3})
    pipeline = LncRnaPipeline(config)
    result = pipeline.run()

    run_dir = pipeline.output_dir
    runs = list(run_dir.iterdir())
    assert len(runs) == 1
    files = {p.name for p in runs[0].iterdir()}
    assert {"config.yaml", "best_params.json", "cv_scores.csv", "metrics.csv",
            "predictions.csv", "run_info.json"} <= files

    assert result["best_params"] == {"C": 1.0}
    assert "auroc" in result["overall"] and "auprc" in result["overall"]

    with open(runs[0] / "best_params.json") as fh:
        assert json.load(fh) == {"C": 1.0}

    import pandas as pd
    cv_scores = pd.read_csv(runs[0] / "cv_scores.csv")
    assert len(cv_scores) == 3  # one row per stratified fold

    predictions = pd.read_csv(runs[0] / "predictions.csv")
    assert len(predictions) == 10 * len(_CELL_LINES)  # test set size


def test_grid_tuning_picks_a_value_from_the_search_space(data_files, tmp_path):
    search_space_path = tmp_path / "logreg_search.yaml"
    search_space_path.write_text("C:\n  grid: [0.01, 100.0]\n")

    config = _base_config(
        data_files, tmp_path,
        model={"name": "logreg"},
        tuning={"method": "grid", "search_space": str(search_space_path), "metric": "auprc"},
        cv={"strategy": "stratified", "n_splits": 2},
    )
    pipeline = LncRnaPipeline(config)
    result = pipeline.run()

    assert result["best_params"]["C"] in (0.01, 100.0)

    import pandas as pd
    run_dir = list(pipeline.output_dir.iterdir())[0]
    cv_scores = pd.read_csv(run_dir / "cv_scores.csv")
    assert len(cv_scores) == 2  # one row per grid combo (only C varies)


def test_dnabert2_features_and_celligner_dim(tmp_path, data_files):
    import numpy as np

    targets = [f"T{i}" for i in range(30)] + [f"T{i}" for i in range(30, 40)]
    matrix = np.random.default_rng(0).random((len(targets), 4)).astype(np.float32)
    index = {t: i for i, t in enumerate(targets)}
    emb_path = tmp_path / "embeddings.npz"
    np.savez(emb_path, embeddings=matrix, index=json.dumps(index))

    data_no_seqs = {k: v for k, v in data_files.items() if k != "transcript_sequences"}
    config = _base_config(
        data_files, tmp_path,
        data=data_no_seqs,
        features={"type": "dnabert2", "embeddings": str(emb_path), "cell_embedding_dim": 2},
        model={"name": "logreg", "params": {"C": 1.0}},
        tuning={"method": "fixed"},
        cv={"strategy": "none"},
    )
    pipeline = LncRnaPipeline(config)
    result = pipeline.run()
    assert "auroc" in result["overall"]


def test_config_validation_rejects_tuning_without_cv(data_files, tmp_path):
    config = _base_config(
        data_files, tmp_path,
        tuning={"method": "grid", "search_space": "irrelevant.yaml"},
        cv={"strategy": "none"},
    )
    with pytest.raises(ValueError, match="cv.strategy"):
        LncRnaPipeline(config)


def test_config_validation_rejects_bad_feature_type(data_files, tmp_path):
    config = _base_config(data_files, tmp_path, features={"type": "bogus"})
    with pytest.raises(ValueError, match="features.type"):
        LncRnaPipeline(config)


def test_config_validation_requires_embeddings_for_dnabert2(data_files, tmp_path):
    config = _base_config(data_files, tmp_path, features={"type": "dnabert2"})
    with pytest.raises(ValueError, match="features.embeddings"):
        LncRnaPipeline(config)


def test_resample_applies_to_training_split_only(data_files, tmp_path, monkeypatch):
    """Resampling must never see validation/test rows -- doing so would change the
    class balance being measured against and silently invalidate the metrics."""
    import lncfit.pipeline as pipeline_module

    seen_sizes = []
    real_resample = pipeline_module.resample

    def spy(X, y, **kwargs):
        seen_sizes.append(len(y))
        return real_resample(X, y, **kwargs)

    monkeypatch.setattr(pipeline_module, "resample", spy)

    config = _base_config(
        data_files, tmp_path,
        model={"name": "logreg", "params": {"C": 1.0},
               "resample": {"method": "random_over"}},
        cv={"strategy": "stratified", "n_splits": 2},
    )
    # data_files fixture: 30 train targets and 10 test targets, x 5 cell lines
    n_train = 30 * len(_CELL_LINES)
    n_test = 10 * len(_CELL_LINES)

    LncRnaPipeline(config).run()

    assert seen_sizes, "resample() should have been called"
    # Every call is a training split: either a CV fold (< n_train) or the full
    # training set -- never the test set, and never train+test combined.
    assert all(s <= n_train for s in seen_sizes), seen_sizes
    assert n_train in seen_sizes, "final fit should resample the full training set"
    assert all(s != n_test for s in seen_sizes), "a test-sized split was resampled"


def test_resample_rejects_unknown_method(data_files, tmp_path):
    config = _base_config(
        data_files, tmp_path,
        model={"name": "logreg", "resample": {"method": "not_a_method"}},
    )
    with pytest.raises(ValueError, match="model.resample.method"):
        LncRnaPipeline(config)


def test_loco_nested_tuning_uses_only_training_cell_lines(monkeypatch, tmp_path):
    """Nested LOCO tuning must run its inner CV on the fold's TRAINING cell lines
    only. If the held-out cell line's rows leaked into the inner search, its own
    labels would help choose the hyperparameters used to predict it."""
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent / "scripts"))
    import run_cellline_loco as loco

    seen_row_counts = []
    real_tune = loco._tune_optuna

    def spy(model_name, seed, X, y, splits, *args, **kwargs):
        seen_row_counts.append(len(y))
        # Return a cheap fixed param set instead of actually searching.
        return {"max_depth": 3}

    monkeypatch.setattr(loco, "_tune_optuna", spy)

    records = _synthetic_records(20)
    data_path = tmp_path / "all.jsonl.gz"
    save_jsonl(records, data_path)
    seqs_path = tmp_path / "seqs.json"
    with open(seqs_path, "w") as fh:
        json.dump({gid: [seq, ""] for gid, seq in _synthetic_transcript_sequences(20).items()}, fh)

    n_total = len(records)
    n_cell_lines = len(_CELL_LINES)

    loco.run({
        "data": {"path": str(data_path), "transcript_sequences": str(seqs_path)},
        "features": {"type": "kmer", "k": 3, "cell_embedding_dim": 0},
        "model": {"name": "logreg"},
        "tuning": {"method": "optuna", "nested": True,
                   "search_space": "configs/search_spaces/logreg.yaml", "n_trials": 1},
        "seed": 42,
        "output_dir": str(tmp_path / "runs"),
    })

    assert len(seen_row_counts) == n_cell_lines, "expected one inner search per outer fold"
    # Each inner search sees exactly the other cell lines' rows, never the whole set.
    expected = n_total - n_total // n_cell_lines
    assert all(c == expected for c in seen_row_counts), seen_row_counts
    assert all(c < n_total for c in seen_row_counts)
