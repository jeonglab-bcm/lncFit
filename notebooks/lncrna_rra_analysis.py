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
        # lncRNA RRA-hit classifier — analysis reproduction

        Reproduces the lncRNA-level essentiality classifier from issues #60-#66:
        predict, per lncRNA x cell line, whether MAGeCK-RRA calls a Day-14
        depletion hit (RRA P value < 0.05 and log2 fold-change < 0).

        **Feature source note (issue #65):** features come from each lncRNA's own
        spliced transcript sequence (Sarropoulos et al. 2019, hg19/GRCh37 — see
        issue #66 on why hg38 was dropped), *not* the CRISPR guide spacer
        sequences the earlier #61/#62 runs mistakenly used. If you're comparing
        numbers against those PRs, the ones here are the corrected ones.

        Run `uv run marimo edit notebooks/lncrna_rra_analysis.py` from the repo
        root to open this interactively.
        """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))

    import gzip
    import json

    import pandas as pd

    from lncfit.constants import CELL_LINES
    from lncfit.features import build_lncrna_features, fit_vocab
    from lncfit.screen_data import LncRnaRecord, load_annotations, load_jsonl, load_target_groups
    from lncfit.splits import split_by_chrom
    from lncfit.xgboost_model import evaluate_lncrna_by_group
    import xgboost as xgb

    RAW = REPO_ROOT / "data" / "raw"
    PROCESSED = REPO_ROOT / "data" / "processed"
    RESULTS = REPO_ROOT / "results" / "lncrna_rra_day14"
    return (
        CELL_LINES,
        LncRnaRecord,
        PROCESSED,
        RAW,
        RESULTS,
        REPO_ROOT,
        build_lncrna_features,
        evaluate_lncrna_by_group,
        fit_vocab,
        gzip,
        json,
        load_annotations,
        load_jsonl,
        load_target_groups,
        pd,
        split_by_chrom,
        xgb,
    )


@app.cell
def _(mo):
    mo.md("## 1. Raw data: guide design table and lncRNA annotations")
    return


@app.cell
def _(RAW, load_annotations, load_target_groups, pd):
    target_groups = load_target_groups(RAW / "mmc2.xlsx")
    annotations = load_annotations(RAW / "mmc2.xlsx")

    group_counts = pd.Series(target_groups).value_counts()
    group_counts.name = "n_targets"
    group_counts
    return annotations, target_groups


@app.cell
def _(mo):
    mo.md(
        """
        `long non-coding RNA` is the group we restrict to (equivalent to the
        `Hum_XLOC_*` gene-id prefix, verified 1:1 in issue #60). Everything else
        (`protein-coding gene`, `essential protein-coding gene`, `non-targeting`)
        is a control and excluded from `load_rra`.

        ## 2. The lncRNA RRA Day-14 dataset
        """
    )
    return


@app.cell
def _(LncRnaRecord, PROCESSED, load_jsonl, pd):
    records = load_jsonl(PROCESSED / "lncrna_rra_day14.jsonl.gz", record_cls=LncRnaRecord)

    stats = pd.DataFrame(
        {
            "cell_line": [r.cell_line for r in records],
            "label": [r.label for r in records],
        }
    )
    per_cell_line = stats.groupby("cell_line")["label"].agg(n="count", n_hits="sum")
    per_cell_line["hit_rate"] = (per_cell_line["n_hits"] / per_cell_line["n"]).round(4)
    per_cell_line
    return (records,)


@app.cell
def _(mo, records):
    n_pos = sum(r.label for r in records)
    mo.md(
        f"""
        **{len(records):,}** records ({len({r.target for r in records}):,} unique
        lncRNAs x {len({r.cell_line for r in records})} cell lines),
        **{n_pos:,}** significant hits ({n_pos / len(records):.1%} overall positive rate).

        ## 3. Sequence source: each lncRNA's own spliced transcript (issue #65)
        """
    )
    return


@app.cell
def _(PROCESSED, json, mo, records):
    transcript_path = PROCESSED / "body_sequences_transcript.json"
    if transcript_path.exists():
        with open(transcript_path) as _fh:
            _raw = json.load(_fh)
        transcript_sequences = {gene_id: seq for gene_id, (seq, _) in _raw.items()}
        _targets = {r.target for r in records}
        _missing = _targets - set(transcript_sequences.keys())
        _msg = (
            f"Loaded **{len(transcript_sequences):,}** transcript sequences "
            f"({transcript_path.relative_to(transcript_path.parent.parent.parent)}). "
            f"{len(_missing)} / {len(_targets)} screen targets missing a sequence."
        )
    else:
        transcript_sequences = {}
        _msg = (
            "**Not found** — run `uv run python scripts/download_genome.py --extract` "
            "then `uv run python -m lncfit.sequence --sequence-type transcript` to "
            "regenerate `data/processed/body_sequences_transcript.json` (not committed; "
            "reproducible from the hg19 GTF + genome FASTA, see issue #65/#66). "
            "Cells below will show zero-vector features until this exists."
        )
    mo.md(_msg)
    return (transcript_sequences,)


@app.cell
def _(mo):
    mo.md("## 4. Interactive: build features and fit a quick model")
    return


@app.cell
def _(mo):
    k_dropdown = mo.ui.dropdown(options=["3", "4", "5", "6"], value="3", label="k-mer size")
    include_distance_checkbox = mo.ui.checkbox(label="include distance-to-nearest-gene feature")
    mo.hstack([k_dropdown, include_distance_checkbox])
    return include_distance_checkbox, k_dropdown


@app.cell
def _(mo):
    mo.md(
        """
        Fits a **small, fast** model (200 trees, 2 threads) for quick interactive
        feedback — this is a demo, not the tuned model. For the real tuned sweep,
        run `scripts/tune_lncrna_xgboost.py` (issue #62/#65), which uses Optuna +
        chromosome LOCO-CV; see `results/lncrna_rra_day14/README.md` for those numbers.
        """
    )
    return


@app.cell
def _(
    build_lncrna_features,
    evaluate_lncrna_by_group,
    fit_vocab,
    include_distance_checkbox,
    k_dropdown,
    pd,
    records,
    split_by_chrom,
    transcript_sequences,
    xgb,
):
    _k = int(k_dropdown.value)
    _include_distance = include_distance_checkbox.value

    train_records, test_records = split_by_chrom(records, test_chrom="1")

    _train_targets = {r.target for r in train_records}
    _train_seqs = [transcript_sequences[t] for t in _train_targets if t in transcript_sequences]
    _vocab = fit_vocab(_train_seqs, _k)

    X_train, y_train, _ = build_lncrna_features(
        train_records, transcript_sequences, k=_k, include_distance=_include_distance, vocab=_vocab,
    )
    X_test, y_test, _ = build_lncrna_features(
        test_records, transcript_sequences, k=_k, include_distance=_include_distance, vocab=_vocab,
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
    pd.DataFrame(metrics_rows).round(4)
    return


@app.cell
def _(mo):
    mo.md("## 5. Tuned sweep results (precomputed, issues #61/#62/#65)")
    return


@app.cell
def _(RESULTS, mo):
    _plot_path = RESULTS / "auroc_auprc_sweep.png"
    if _plot_path.exists():
        mo.image(str(_plot_path))
    else:
        mo.md(
            "Plot not found — run `uv run python scripts/plot_lncrna_auc_sweep.py` "
            "after the tuning sweep completes."
        )
    return


@app.cell
def _(mo):
    mo.md(
        """
        See `results/lncrna_rra_day14/README.md` for the full untuned-vs-tuned
        writeup, per-cell-line breakdown, and the honest discussion of what
        changed (and why) between the guide-sequence-based numbers in #61/#62/#64
        and the transcript-sequence-corrected numbers from #65.
        """
    )
    return


if __name__ == "__main__":
    app.run()
