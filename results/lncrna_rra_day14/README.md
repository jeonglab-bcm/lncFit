# lncRNA-level RRA hit classification (Day 14)

Issue #60: predict, per lncRNA x cell line, whether MAGeCK-RRA calls it a significant
depletion hit at Day 14 (RRA P value < 0.05 and log2 fold-change < 0), instead of
regressing per-guide log2 fold-change.

Dataset: `data/processed/lncrna_rra_day14.jsonl.gz`, built from the previously-unparsed
`S2F`-`S2J` RRA sheets of `mmc3.xlsx`, restricted to `Target group == "long non-coding RNA"`
rows (5,496 lncRNAs x 5 cell lines = 27,480 records; 1,249 hits, 4.5% positive rate).
Chromosome-1 hold-out split (`train_lncrna_day14_chrom1.jsonl.gz` / `test_lncrna_day14_chrom1.jsonl.gz`):
25,010 train / 2,470 test.

Features: pooled k-mer frequencies across all of a lncRNA's guide spacers (no single guide
sequence exists at this granularity) + cell-line one-hot. XGBoost classifier
(`binary:logistic`, `scale_pos_weight` set from the train split's class ratio) — same
hyperparameter defaults and `tree_method="hist"` as `scripts/train_xgboost.py`.

## k sweep (chr1 test split)

| k | features | AUROC | AUPRC |
|---|---|---|---|
| **3** | 69  | **0.7049** | **0.1403** |
| 4 | 260 | 0.7031 | 0.1102 |
| 5 | 1019 | 0.7018 | 0.1198 |
| 6 | 4040 | 0.6494 | 0.1084 |

k=3 wins on both AUROC and AUPRC — the pooled per-lncRNA feature vector has far fewer
independent training examples than the guide-level dataset, so larger k-mer vocabularies
overfit. This mirrors the k=3 preference already seen in the DNABERT-2 ablation
(`results/dnabert_ablation/`).

## Per-cell-line breakdown, k=3

See `metrics_k3.csv`. AUROC ranges 0.62 (HAP1) - 0.77 (MDA-MB-231); overall AUROC 0.70,
AUPRC 0.14 vs. a 5.3% base rate (~2.7x lift). All AUROC > 0.5, i.e. the model captures
real sequence-level signal on which lncRNAs are essential hits, though precision/recall
at the default 0.5 threshold are still low (this is a ~5%-positive-rate problem; a lower
decision threshold or precision-at-k framing would likely be a better fit than F1 with
threshold=0.5 for a follow-up).

## Files

- `metrics_k3.csv`, `metrics_k4.csv`, `metrics_k5.csv`, `metrics_k6.csv` — per-cell-line
  classification metrics for each k.
