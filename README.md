# lncFit


## Pipeline overview

The trainable classifier pipeline (`lncfit.pipeline.LncRnaPipeline`, driven by
`scripts/run_pipeline.py --config <file>.yaml`) turns raw screen data and
lncRNA sequences into a feature matrix, then trains/tunes/evaluates a model
against it. See [`configs/README.md`](configs/README.md) for the full config
schema and ready-to-run examples.

![lncFit pipeline diagram](docs/diagrams/pipeline.svg)

## Leaderboard

Live site: **https://jeonglab-bcm.github.io/lncFit/**

Anyone can train a model with the pipeline above and submit its predictions to be
scored against the real held-out labels -- CI validates and scores every submission
independently (it never trusts a submitted metrics file), so nothing gets on the
board without being checked. There are two challenges, testing two different kinds
of generalization:

| Challenge | Holds out | Question it answers |
|---|---|---|
| [`lncrna_rra_day14`](results/lncrna_rra_day14/leaderboard/README.md) | chromosome 1 | Does this generalize to an lncRNA the model has never seen? |
| [`lncrna_rra_day14_cellline_loco`](results/lncrna_rra_day14_cellline_loco/leaderboard/README.md) | one cell line at a time | Does this generalize to a cell line the model has never seen at all? |

The Day-14 chromosome challenge also supports a strictly longitudinal model
that uses the experiment's earlier Day-7 RRA and guide measurements:

```bash
uv run python scripts/run_day7_longitudinal.py \
  --config configs/pipeline/xgboost_day7_longitudinal.yaml
```

The source workbooks contain Day 7 and Day 14 only (there is no Day 17
measurement). The longitudinal feature builder never uses Day-14 values as
features and streams guide records to keep memory bounded. The model combines
a binary hit classifier with a significance regressor; the latter receives
continuous Day-14 p-values from training chromosomes only, never chromosome 1.

### How to submit

1. Train a model and get its `predictions.csv` -- either `scripts/run_pipeline.py`
   for the chromosome-held-out challenge, or `scripts/run_cellline_loco.py` for the
   cell-line-LOCO one. See the challenge-specific READMEs linked above for the
   exact file layout and required `submission.yaml` fields.
2. On a branch named `submission/<your-name-or-team>-<short-slug>` (the prefix is
   required -- CI only runs for those), add
   `results/<challenge>/leaderboard/submissions/<your-name-or-team>-<short-slug>/`
   with your `predictions.csv`, `submission.yaml`, and (encouraged) `config.yaml`.
3. Open a PR. CI recomputes AUROC/AUPRC from your `predictions.csv` against the
   real labels, fails the check if anything's missing/malformed, and on success
   commits the updated leaderboard (both the `LEADERBOARD.md` on GitHub and the
   live page above) right onto your PR branch.
4. Get it reviewed and merged.
