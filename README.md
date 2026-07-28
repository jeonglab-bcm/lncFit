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

### The challenge: THP1 hold-out

Can a model predict lncRNA essentiality in a cell line it has never seen a single
row of?

| | |
|---|---|
| **Train on** | [HAP1, K562, MDA-MB-231](data/holdout_thp1/train_thp1_holdout.jsonl.gz) -- 16,488 rows, labels included |
| **Predict** | [THP1](data/holdout_thp1/holdout_thp1_features.jsonl.gz) -- 5,496 rows, 202 positives (3.7%), labels withheld |
| **Excluded** | HEK293FT -- not a real cancer line, no Celligner data |
| **Ranked by** | AUPRC, with AUROC shown alongside |
| **Scored by** | [`scripts/build_leaderboard.py`](scripts/build_leaderboard.py) in CI, per [`challenge.yaml`](results/lncrna_rra_day14_thp1_holdout/leaderboard/challenge.yaml) -- never from a submitted metrics file |

### Enter in three commands

```bash
git clone https://github.com/jeonglab-bcm/lncFit.git && cd lncFit && git lfs pull
python3 scripts/make_barebones_submission.py --submitter YOUR-HANDLE \
    --out results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/YOUR-HANDLE-barebones
python3 scripts/score_submission.py results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/YOUR-HANDLE-barebones
```

That builds a valid submission with no dependencies and no genome download, then
prints the exact AUROC/AUPRC CI will publish.

**Full walkthrough: [`docs/PARTICIPATE.md`](docs/PARTICIPATE.md)** -- setup, a
starter config, the submission format, CI troubleshooting, the rules, and what has
already been tried. It is the one place any of that is documented; everything else
links to it.

> **This is an honour-system board, not a blind benchmark.** `label` comes from the
> *published* supplementary tables of the source paper, so THP1's answers are a
> journal download away no matter what this repo ships. `data/holdout_thp1/` exists
> to make the honest path the easy one. See
> [the rules](docs/PARTICIPATE.md#the-rules).
