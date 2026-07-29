#!/usr/bin/env python3
"""Expression-based lncRNA essentiality model: per-cell-line TPM + DNABERT-2 PCA.

The strongest ELIGIBLE model found under the no-measured-depletion rule (see
docs/PARTICIPATE.md). It uses no fold_change, no rra_pvalue and no guide-level
depletion from any cell line or day -- only baseline abundance, transcript
sequence and static annotation.

Features, all functions of (lncRNA) or (lncRNA, cell line):
  * log1p TPM in each of the 5 screen lines, from mmc2.xlsx S1C/S1E
  * summary of that profile: mean, max, min, sd, and breadth (lines with TPM > 1)
  * log1p TPM in the line whose label is being predicted, plus its deviation from
    the gene's mean -- the single most informative feature, because a transcript a
    cell barely expresses cannot be depleted to kill it
  * DNABERT-2 transcript embedding reduced to 32 PCs (32 beat 64/128 and the raw
    768; PCA is unsupervised so fitting it across all targets leaks nothing)
  * transcript length, GC, longest homopolymer, distance to nearest coding gene, strand

The prediction is a fixed 5-member ensemble of XGBoost classifiers whose members
were chosen by leave-one-cell-line-out AUPRC over the TRAINING lines only. Run
--validate to reproduce that selection number; THP1 is never used to choose
anything.

  python scripts/run_expression_xgb.py --validate     # honest LOCO estimate
  python scripts/run_expression_xgb.py --out results/.../submissions/<handle>-expression
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

_TRAIN = "data/holdout_thp1/train_thp1_holdout.jsonl.gz"
_TEST = "data/holdout_thp1/holdout_thp1_features.jsonl.gz"
_SEQS = "data/processed/body_sequences_transcript.json"
_DBERT = "data/processed/dnabert2_transcript_full.npz"
_MMC2 = "data/raw/mmc2.xlsx"
_TPM_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]

# The 5 ensemble members, in the order LOCO ranked them. `dbert` toggles the
# DNABERT-2 block; `profile` toggles the all-line TPM columns. Every member always
# gets the predicted-line TPM block, which is what carries most of the signal.
_MEMBERS = [
    dict(depth=3, dbert=True, profile=True),
    dict(depth=5, dbert=False, profile=True),
    dict(depth=5, dbert=False, profile=False),
    dict(depth=5, dbert=True, profile=True),
    dict(depth=3, dbert=True, profile=True, kmer_free=True),
]
_XGB = dict(objective="binary:logistic", eval_metric="aucpr", n_estimators=600,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
            min_child_weight=3, reg_alpha=1.0, reg_lambda=2.0, random_state=42,
            n_jobs=4)


def _read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _longest_homopolymer(s: str) -> int:
    run = best = 0
    prev = ""
    for ch in s:
        run = run + 1 if ch == prev else 1
        prev = ch
        best = max(best, run)
    return best


def _load_tpm(targets: list[str]) -> np.ndarray:
    """(n_targets, 5) log1p TPM. S1C (total RNA-seq) preferred, S1E (mRNA-seq) fills gaps.

    Both sheets carry a title row above the real header, hence skiprows=1 then
    promoting the first row. Where a line has both an untransduced and an
    RfxCas13d column we take RfxCas13d -- that is the population actually screened.
    """
    frames = []
    for sheet in ("S1C", "S1E"):
        d = pd.read_excel(_MMC2, sheet_name=sheet, skiprows=1)
        d.columns = list(d.iloc[0])
        d = d.iloc[1:].reset_index(drop=True)
        d = d.rename(columns={d.columns[0]: "lncRNA"})
        frames.append(d)

    out = np.zeros((len(targets), len(_TPM_LINES)))
    seen = np.zeros_like(out, dtype=bool)
    index = {t: i for i, t in enumerate(targets)}
    for d in frames:
        rows = {g: i for i, g in enumerate(d["lncRNA"])}
        for j, line in enumerate(_TPM_LINES):
            col = next((c for c in (f"{line} RfxCas13d", line) if c in d.columns), None)
            if col is None:
                continue
            vals = pd.to_numeric(d[col], errors="coerce").to_numpy()
            for t, i in index.items():
                if seen[i, j] or t not in rows:
                    continue
                v = vals[rows[t]]
                if not np.isnan(v):
                    out[i, j] = v
                    seen[i, j] = True
    missing = int((~seen).all(axis=1).sum())
    if missing:
        print(f"  note: {missing} target(s) had no TPM in any sheet, zero-filled")
    return np.log1p(out)


def _load_guides(targets):
    """Guide spacer sequences per target from mmc2.xlsx S1B.

    Read from the supplement rather than the processed screen files because those
    carry depletion values alongside the sequences. S1B is pure library design --
    no outcomes -- so it is eligible, and the published features file omits
    guide_sequences entirely.
    """
    d = pd.read_excel(_MMC2, sheet_name="S1B", skiprows=1)
    d.columns = list(d.iloc[0])
    d = d.iloc[1:]
    wanted = set(targets)
    out = {}
    for tgt, seq in zip(d["Target"], d["Sequence (5' - 3')"]):
        if tgt in wanted and isinstance(seq, str) and seq:
            out.setdefault(tgt, []).append(seq.upper())
    return out


def _build_blocks(targets, meta):
    """Returns (per-target base features, log-TPM matrix, dbert PCs)."""
    raw = json.load(open(_SEQS))
    seqs = {t: (max(v, key=len) if isinstance(v, list) and v else (v if isinstance(v, str) else ""))
            for t, v in raw.items()}
    guides = _load_guides(targets)
    print(f"  guide design: {len(guides)}/{len(targets)} targets covered (mmc2 S1B)")
    base = []
    for t in targets:
        s = (seqs.get(t) or "").upper()
        d = meta[t].get("distance_to_closest_pc_gene")
        gs = guides.get(t) or []
        ggc = [(g.count("G") + g.count("C")) / len(g) for g in gs] or [0.0]
        ga = [g.count("A") / len(g) for g in gs] or [0.0]
        gt = [g.count("T") / len(g) for g in gs] or [0.0]
        ghp = [_longest_homopolymer(g) for g in gs] or [0]
        base.append([
            float(len(s)), np.log1p(len(s)),
            (s.count("G") + s.count("C")) / len(s) if s else 0.0,
            float(_longest_homopolymer(s)),
            float(d) if d is not None else -1.0,
            np.log1p(d) if d is not None else -1.0,
            1.0 if meta[t].get("strand") == "+" else 0.0,
            float(len(gs)),
            float(np.mean(ggc)), float(np.std(ggc)),
            float(np.min(ggc)), float(np.max(ggc)),
            float(np.mean(ga)), float(np.mean(gt)),
            float(np.mean(ghp)), float(np.max(ghp)),
        ])
    z = np.load(_DBERT, allow_pickle=True)
    emb = z["embeddings"]
    id2row = json.loads(str(z["index"]))
    full = np.vstack([emb[id2row[t]] if t in id2row else np.zeros(emb.shape[1])
                      for t in targets])
    return np.array(base), _load_tpm(targets), PCA(n_components=32, random_state=0).fit_transform(full)


def _profile(lt):
    return np.column_stack([lt, lt.mean(1), lt.max(1), lt.min(1), lt.std(1),
                            (lt > np.log1p(1)).sum(1)])


def _line_block(lt, j):
    v = lt[:, j]
    return np.column_stack([v, v - lt.mean(1)])


def _design(member, base, lt, dbert, line_idx):
    parts = [base]
    if member.get("profile"):
        parts.append(_profile(lt))
    else:
        parts.append(lt)
    if member.get("dbert"):
        parts.append(dbert)
    parts.append(_line_block(lt, line_idx))
    return np.hstack(parts)


def _fit_predict(member, base, lt, dbert, train_rows, train_y, predict_line):
    """train_rows: list of (target_index, line_index). Features for a row use that
    row's own cell line, so train and predict see the same construction."""
    per_line = {j: _design(member, base, lt, dbert, j)
                for j in {j for _, j in train_rows}}
    Xtr = np.array([per_line[j][i] for i, j in train_rows])
    m = xgb.XGBClassifier(max_depth=member["depth"], **_XGB).fit(Xtr, train_y)
    return m.predict_proba(_design(member, base, lt, dbert, predict_line))[:, 1]


def _z(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="Leave-one-cell-line-out over the TRAINING lines; prints the "
                         "honest selection estimate. Never touches THP1 labels.")
    ap.add_argument("--out", default=None, help="Submission directory to write.")
    ap.add_argument("--submitter", default=None, help="GitHub handle for submission.yaml.")
    args = ap.parse_args()

    train = _read_jsonl(_TRAIN)
    test = _read_jsonl(_TEST)
    targets = sorted({r["target"] for r in train} | {r["target"] for r in test})
    meta = {r["target"]: r for r in train + test}
    ti = {t: i for i, t in enumerate(targets)}
    li = {l: j for j, l in enumerate(_TPM_LINES)}
    print(f"{len(targets):,} lncRNAs; {len(train):,} training rows; {len(test):,} to predict")
    base, lt, dbert = _build_blocks(targets, meta)

    train_lines = sorted({r["cell_line"] for r in train})
    by_line = {c: [(ti[r["target"]], li[c], r["label"]) for r in train if r["cell_line"] == c]
               for c in train_lines}

    if args.validate:
        print(f"\nLeave-one-cell-line-out over {train_lines} (THP1 not involved):")
        aps = []
        for held in train_lines:
            rows = [(i, j) for c in train_lines if c != held for i, j, _ in by_line[c]]
            y = [lab for c in train_lines if c != held for _, _, lab in by_line[c]]
            ens = np.mean([_z(_fit_predict(m, base, lt, dbert, rows, np.array(y), li[held]))
                           for m in _MEMBERS], axis=0)
            yt = np.zeros(len(targets))
            for i, _, lab in by_line[held]:
                yt[i] = lab
            ap_ = average_precision_score(yt, ens)
            aps.append(ap_)
            print(f"  hold out {held:<12} AUROC {roc_auc_score(yt, ens):.4f}  AUPRC {ap_:.4f}")
        print(f"\n  mean training-line LOCO AUPRC = {np.mean(aps):.4f}")
        print("  This is the number to select on. Do NOT select on the THP1 score.")

    if not args.out:
        return
    if not args.submitter:
        raise SystemExit("--out requires --submitter")

    rows = [(i, j) for c in train_lines for i, j, _ in by_line[c]]
    y = np.array([lab for c in train_lines for _, _, lab in by_line[c]])
    target_line = test[0]["cell_line"]
    print(f"\nTraining on {train_lines} -> predicting {target_line}")
    ens = np.mean([_z(_fit_predict(m, base, lt, dbert, rows, y, li[target_line]))
                   for m in _MEMBERS], axis=0)
    # Scale to [0, 1]: the scorer only reads ranking, but y_pred_proba should look
    # like a probability. It is an uncalibrated ranking score, not a real one.
    scaled = (ens - ens.min()) / (ens.max() - ens.min())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target": r["target"], "cell_line": r["cell_line"],
                   "y_pred_proba": scaled[ti[r["target"]]]} for r in test]) \
      .to_csv(out / "predictions.csv", index=False)
    (out / "submission.yaml").write_text(
        f"submitter: {args.submitter}\n"
        'model: "XGBoost ensemble on per-cell-line lncRNA TPM + DNABERT-2 PCA-32"\n'
        "uses_measured_depletion: false\n"
        "description: >\n"
        "  Five-member XGBoost ensemble over baseline lncRNA abundance (log1p TPM in each\n"
        "  of the 5 screen lines from mmc2.xlsx S1C/S1E, plus the predicted line's own TPM\n"
        "  and its deviation from the gene's mean), a 32-PC DNABERT-2 transcript embedding,\n"
        "  and static annotation. No fold_change, rra_pvalue or guide-level depletion from\n"
        "  any cell line or day. Members selected by leave-one-cell-line-out AUPRC over the\n"
        "  training lines only; reproduce with scripts/run_expression_xgb.py --validate.\n"
        "  y_pred_proba is an uncalibrated ranking score.\n")
    (out / "config.yaml").write_text(
        "script: scripts/run_expression_xgb.py\n"
        f"train_path: {_TRAIN}\n"
        f"test_path: {_TEST}\n"
        "features:\n"
        "  - log1p TPM per screen cell line (mmc2.xlsx S1C total RNA-seq, S1E mRNA-seq)\n"
        "  - TPM profile summary: mean, max, min, sd, breadth (lines with TPM > 1)\n"
        "  - predicted-line TPM and its deviation from the gene's mean\n"
        "  - DNABERT-2 transcript embedding, 32 principal components\n"
        "  - transcript length, GC, longest homopolymer, distance to nearest coding gene, strand\n"
        "excluded_features:\n"
        "  - fold_change (any cell line, any day)\n"
        "  - rra_pvalue (any cell line)\n"
        "  - guide-level depletion\n"
        "model:\n"
        "  estimator: XGBoost binary:logistic, aucpr\n"
        f"  ensemble_members: {len(_MEMBERS)}\n"
        "  max_depths: [3, 5, 5, 5, 3]\n"
        "  n_estimators: 600\n"
        "  learning_rate: 0.03\n"
        "  seed: 42\n"
        "selection:\n"
        "  method: leave-one-cell-line-out over HAP1/K562/MDA-MB-231\n"
        "  heldout_cell_outcomes_used: false\n")
    print(f"Wrote {out}/predictions.csv, submission.yaml, config.yaml")


if __name__ == "__main__":
    main()
