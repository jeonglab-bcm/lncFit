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
scored. CI validates and scores every submission independently -- it never trusts a
submitted metrics file.

> **This is an honour-system board, not a blind benchmark.** The `label` being
> predicted is computed from the *published* supplementary tables of the source
> screen paper (`data/raw/mmc3.xlsx`, sheets S2F-S2J), so THP1's answers are
> derivable from public data no matter what this repo ships -- we tried
> encrypting them and it was security theatre. What the repo does do is make the
> honest path the easy one: `data/holdout_thp1/` gives you a training file with
> THP1 removed and a THP1 features file with `label` blanked to `-1`, so following
> the instructions never puts the answers in front of you. Please don't go
> looking. If you need a genuinely blind evaluation, it has to hold out data that
> isn't in the paper.

### The challenge: blind THP1 hold-out

| | |
|---|---|
| **Train on** | HAP1, K562, MDA-MB-231 (labels public) |
| **Predict** | THP1 (5,496 rows, labels private) |
| **Excluded** | HEK293FT -- not a real cancer line, no Celligner data |
| **Ranked by** | AUPRC, with AUROC shown alongside |

It asks whether a model generalizes to a cell line it has never seen a single row
of. Data files:

| File | Contents |
|---|---|
| `data/holdout_thp1/train_thp1_holdout.jsonl.gz` | training rows, labels included |
| `data/holdout_thp1/holdout_thp1_features.jsonl.gz` | THP1 rows to predict, `label` blanked to `-1` |

Scoring reads THP1's rows straight from `data/processed/lncrna_rra_day14.jsonl.gz`
(see the challenge's `challenge.yaml`) -- no separate answers file, since hiding
one would not have hidden anything.

### How to submit

1. Train on `train_thp1_holdout.jsonl.gz` and predict every row of
   `holdout_thp1_features.jsonl.gz`. Any model is fair game;
   `scripts/run_pipeline.py` is the supported path (see
   [`configs/README.md`](configs/README.md)).
2. On a branch named `submission/<your-github-handle>-<short-slug>` (the prefix is
   required -- CI only runs for those), add
   `results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/<same-name>/`
   with your `predictions.csv` (`target`, `cell_line`, `y_pred_proba`),
   `submission.yaml`, and (encouraged) `config.yaml`. See the
   [challenge README](results/lncrna_rra_day14_thp1_holdout/leaderboard/README.md)
   for the exact format -- `submitter` must be a real GitHub handle.
3. Open a PR **from a branch on this repo, not a fork** (fork PRs can't get write
   access to push the regenerated board). CI recomputes AUROC/AUPRC from your
   `predictions.csv`, fails the check if anything's missing or malformed, and on
   success commits the updated leaderboard and live page onto your PR branch.
4. Get it reviewed and merged.

> **A caution on reading the scores.** THP1 has 202 positives, so a single-run
> AUPRC has a 95% bootstrap CI roughly ±0.02 wide. Differences smaller than that
> are noise. `configs/README.md` documents a case where CV ranking and test AUPRC
> were perfectly *anti*-correlated (Spearman -1.0) on the older chromosome-held-out
> split -- don't tune against the leaderboard.
