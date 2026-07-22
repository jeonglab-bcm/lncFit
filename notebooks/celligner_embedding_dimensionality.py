import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Celligner cell-line embedding — dimensionality comparison (issue #78)

        Every model in this project represents the cell line as a 5-column
        one-hot. Issue #78 added a real transcriptomic-similarity embedding
        instead, built by re-running the Celligner alignment method from
        scratch against current DepMap data (see `data/external/README.md` for
        full methodology and validation).

        Celligner only ever *publishes* a 2-D UMAP projection, but internally
        computes a richer 70-D PCA space right before that final UMAP step.
        This notebook lets you pick the embedding **dimensionality** as a
        hyperparameter -- 0 (off), 2 (UMAP), 10 or 70 (pre-UMAP PCA) -- and see
        the effect directly, instead of re-reading a static results table.

        **Headline finding from the precomputed sweep** (section 3 below):
        AUROC climbs steadily with more dimensions, but AUPRC -- the more
        informative metric here, given the ~5% positive rate -- actually peaks
        at dim=2 and gets *worse* at 10/70. More embedding columns isn't free;
        it adds noise/overfit risk for a model that only needs to distinguish
        5 categories. "Bigger" is not automatically "better."

        Run `uv run marimo edit notebooks/celligner_embedding_dimensionality.py`
        from the repo root to open this interactively.
        """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))

    import json

    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    from lncfit.features import build_lncrna_features, fit_vocab
    from lncfit.screen_data import LncRnaRecord, load_jsonl
    from lncfit.xgboost_model import evaluate_lncrna_by_group

    PROCESSED = REPO_ROOT / "data" / "processed"
    RESULTS = REPO_ROOT / "results" / "lncrna_rra_day14"
    K = 5
    SEED = 42
    return (
        K,
        LncRnaRecord,
        PROCESSED,
        RESULTS,
        SEED,
        build_lncrna_features,
        evaluate_lncrna_by_group,
        fit_vocab,
        json,
        load_jsonl,
        pd,
        train_test_split,
        xgb,
    )


@app.cell
def _(mo):
    mo.md("## 1. Load data (chr1 held-out split, k=5 kmer vocab)")
    return


@app.cell
def _(K, LncRnaRecord, PROCESSED, SEED, json, load_jsonl, train_test_split):
    from lncfit.features import fit_vocab as _fit_vocab

    train_records = load_jsonl(PROCESSED / "train_lncrna_day14_chrom1.jsonl.gz", record_cls=LncRnaRecord)
    test_records = load_jsonl(PROCESSED / "test_lncrna_day14_chrom1.jsonl.gz", record_cls=LncRnaRecord)
    with open(PROCESSED / "body_sequences_transcript.json") as _fh:
        _raw = json.load(_fh)
    transcript_sequences = {gid: seq for gid, (seq, _) in _raw.items()}

    _idx_train, _idx_es = train_test_split(
        range(len(train_records)), test_size=0.1,
        stratify=[r.label for r in train_records], random_state=SEED,
    )
    final_train = [train_records[i] for i in _idx_train]
    final_es = [train_records[i] for i in _idx_es]

    _train_targets = {r.target for r in final_train}
    vocab = _fit_vocab([transcript_sequences[t] for t in _train_targets if t in transcript_sequences], K)
    f"train={len(train_records):,}  final_train={len(final_train):,}  final_es={len(final_es):,}  test={len(test_records):,}  vocab={len(vocab)}"
    return final_es, final_train, test_records, transcript_sequences, vocab


@app.cell
def _(mo):
    mo.md("## 2. Interactive: pick the embedding dimensionality and fit a quick model")
    return


@app.cell
def _(mo):
    dim_dropdown = mo.ui.dropdown(
        options={"off (0)": 0, "UMAP (2)": 2, "pre-UMAP PCA (10)": 10, "pre-UMAP PCA (70)": 70},
        value="UMAP (2)",
        label="Celligner embedding dimensionality",
    )
    dim_dropdown
    return (dim_dropdown,)


@app.cell
def _(mo):
    mo.md(
        """
        Fits a **small, fast** model (200 trees, no early stopping) for quick
        interactive feedback -- this is a demo, not the tuned model. For the
        real tuned numbers (2000 trees, early-stopped, the best-known xgboost
        hyperparameters), run `scripts/run_celligner_embedding_comparison.py`;
        see the precomputed sweep in section 3 below.
        """
    )
    return


@app.cell
def _(
    K,
    build_lncrna_features,
    dim_dropdown,
    evaluate_lncrna_by_group,
    final_es,
    final_train,
    pd,
    test_records,
    transcript_sequences,
    vocab,
    xgb,
):
    _dim = dim_dropdown.value

    X_train, y_train, feature_cols = build_lncrna_features(
        final_train, transcript_sequences, k=K, vocab=vocab, celligner_embedding_dim=_dim,
    )
    X_es, y_es, _ = build_lncrna_features(
        final_es, transcript_sequences, k=K, vocab=vocab, celligner_embedding_dim=_dim,
    )
    X_test, y_test, _ = build_lncrna_features(
        test_records, transcript_sequences, k=K, vocab=vocab, celligner_embedding_dim=_dim,
    )

    _n_pos = int(y_train.sum())
    _n_neg = len(y_train) - _n_pos
    _scale_pos_weight = _n_neg / _n_pos if _n_pos > 0 else 1.0

    _model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="binary:logistic",
        scale_pos_weight=_scale_pos_weight,
        nthread=2,
        random_state=42,
    )
    _model.fit(X_train, y_train)
    y_pred_proba = _model.predict_proba(X_test)[:, 1]

    metrics_rows = evaluate_lncrna_by_group(test_records, y_test, y_pred_proba)
    f"n_features={X_train.shape[1]} (embedding dim={_dim})"
    pd.DataFrame(metrics_rows).round(4)
    return


@app.cell
def _(mo):
    mo.md("## 3. Precomputed full sweep (tuned xgboost, `scripts/run_celligner_embedding_comparison.py`)")
    return


@app.cell
def _(RESULTS, pd):
    _summary_path = RESULTS / "celligner_embedding_comparison" / "summary.csv"
    pd.read_csv(_summary_path) if _summary_path.exists() else "Run scripts/run_celligner_embedding_comparison.py first."
    return


@app.cell
def _(mo):
    mo.md(
        """
        See `data/external/README.md` for the full Celligner realignment
        methodology, the nearest-neighbor lineage-purity validation (K562/THP1
        validate cleanly at every dimensionality; MDA-MB-231 and HAP1 don't,
        for different reasons), and `results/lncrna_rra_day14/README.md` for
        the project-wide writeup this embedding sits in.
        """
    )
    return


if __name__ == "__main__":
    app.run()
