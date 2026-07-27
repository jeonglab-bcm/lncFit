# How to enter the lncFit challenge

Predict which lncRNAs are essential in a cell line your model has never seen.

Live board: **https://jeonglab-bcm.github.io/lncFit/**

| | |
|---|---|
| **Train on** | HAP1, K562, MDA-MB-231 — 16,488 rows, labels included |
| **Predict** | THP1 — 5,496 rows, labels withheld |
| **Ranked by** | AUPRC (AUROC shown alongside) |
| **Positives in the test set** | 202 (3.7%) |
| **Excluded entirely** | HEK293FT — not a real cancer line, no Celligner coordinates |

A row is one lncRNA × cell line. `label = 1` means knocking that lncRNA out
depleted that cell line by day 14 (`rra_pvalue < 0.05 and fold_change < 0`).

The task is *cross-cell-line generalization*: nothing about THP1's own response is
in your training data. This is harder than predicting an unseen lncRNA in a cell
line you've trained on, and it's why scores here look low — a leading AUPRC of
~0.24 against a 0.037 base rate is a ~6× enrichment, not a weak result.

---

## The 60-second version

Three commands, nothing to install beyond Python, no genome download:

```bash
git clone https://github.com/jeonglab-bcm/lncFit.git && cd lncFit
git lfs pull
python3 scripts/make_barebones_submission.py \
    --submitter YOUR-GITHUB-HANDLE \
    --out results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/YOUR-HANDLE-barebones
```

That writes a complete, valid submission. Push it on a `submission/` branch (§4) and
it will score **AUROC 0.7085 / AUPRC 0.2000**.

The model is one line of arithmetic — for each lncRNA, the negated mean knockout
fold-change across the three training cell lines, a pan-essentiality prior with no
learned parameters, no sequence features and no cell-line features. Read the script;
it's 60 lines of standard library.

**Its score currently beats four of the five real submissions on the board**,
including every entry built on DNABERT-2 embeddings and tuned XGBoost. So treat it
as the bar, not the floor: if your model can't clear 0.2000 AUPRC, it hasn't learned
anything a one-liner doesn't already know. The rest of this document is about
clearing it.

---

## 1. Set up

For anything beyond the barebones baseline:

```bash
git clone https://github.com/jeonglab-bcm/lncFit.git
cd lncFit
git lfs pull            # the .jsonl.gz data files are LFS pointers until you do this
uv sync                 # or: pip install -r requirements.txt
```

**`git lfs pull` is not optional.** Skip it and the data files are a few hundred
bytes of pointer text, and you'll get `gzip.BadGzipFile: Not a gzipped file` at the
first read.

Two files are the challenge:

| File | What it is |
|---|---|
| `data/holdout_thp1/train_thp1_holdout.jsonl.gz` | your training data, labels included |
| `data/holdout_thp1/holdout_thp1_features.jsonl.gz` | the rows to predict — `label`, `rra_pvalue` and `fold_change` are all absent |

### Sequence features need one download

The repo ships the screen data and the Celligner cell-line coordinates, but not the
45 MB extracted transcript sequences (they're derived, so they're gitignored). To
build them you need the hg19 genome FASTA, which is ~900 MB compressed:

```bash
mkdir -p data/raw/genome
curl -o data/raw/genome/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz \
  https://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz

# The lncRNA GTF is already in the repo. This writes
# data/processed/body_sequences_transcript.json (~5 min).
uv run python -m lncfit.sequence --sequence-type transcript
```

You can skip this entirely if you bring your own embeddings, or if you work from the
training-outcome columns the way the barebones baseline does.

## 2. Make a prediction

A starter config is committed. It trains XGBoost on 5-mer frequencies and writes a
submittable `predictions.csv` — a real model rather than a one-liner, though note it
scores *below* the barebones baseline (AUPRC 0.1456 vs 0.2000), which tells you
something about how much of this task sequence features actually explain:

```bash
uv run python scripts/run_pipeline.py --config configs/pipeline/thp1_holdout_starter.yaml
```

It prints a CV score, then:

```
Test set has no labels (withheld) -- skipping metrics, writing predictions only.
Run saved -> results/lncrna_rra_day14_thp1_holdout/pipeline_runs/run_xgboost_<timestamp>
Predictions for submission -> .../predictions.csv
```

**Reporting no test metrics is correct, not a failure.** You don't have THP1's
labels, so there is nothing to score against locally; the board scores you.

Now change something — `configs/README.md` documents every knob (model, feature
type, embeddings, PCA, resampling, tuning, CV). Any model at all is fair game; the
pipeline is a convenience, not a requirement. If you write your own, just emit a
CSV with the three required columns.

### The CV number is optimistic — don't select on it

The starter's stratified K-fold puts the *same* cell lines in train and validation,
so it flatters models that memorize cell-line-specific signal, which is exactly
what this task punishes. For an honest estimate, hold out a training cell line:

```bash
uv run python scripts/run_cellline_loco.py --config configs/cellline_loco/xgboost_kmer_celligner2.yaml
```

That rotates each cell line into the validation slot, which mirrors the real task.

## 3. Package the submission

One directory, named the same as your branch's slug:

```
results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/<your-handle>-<slug>/
├── predictions.csv     required
├── submission.yaml     required
└── config.yaml         optional, strongly encouraged
```

**`predictions.csv`** — needs `target`, `cell_line`, `y_pred_proba`, covering all
5,496 THP1 rows exactly once:

```csv
target,cell_line,y_pred_proba
Hum_XLOC_000006,THP1,0.000282
Hum_XLOC_000008,THP1,0.001744
```

`y_pred_proba` is a *score*, not a hard call — anything monotone in confidence
works, since AUROC and AUPRC only read the ranking. Extra columns are ignored, so
copying a pipeline run's file straight over is fine. Rows for excluded cell lines
are tolerated too.

**`submission.yaml`**:

```yaml
submitter: your-github-handle     # must be a real handle -- it gets linked
model: xgboost + DNABERT-2 + distance-to-gene, max_depth=9
description: >
  What you did, especially the parts CI cannot see: how features were built,
  whether you used the training file unmodified, how you picked hyperparameters.
```

## 4. Open the PR

```bash
git checkout -b submission/<your-handle>-<slug>
git add results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/<your-handle>-<slug>
git commit -m "Submission: <short description> (<your-handle>)"
git push -u origin HEAD
gh pr create --fill
```

Two constraints that will silently cost you a CI run if you miss them:

- **The branch must start with `submission/`.** CI is gated on that prefix so it
  stays off the repo's other PRs.
- **Push to a branch on this repo, not a fork.** Fork PRs can't get the write
  access CI needs to push the regenerated board back onto your branch.

CI then recomputes AUROC/AUPRC from your `predictions.csv` against the real labels,
fails red if the submission is malformed, and on success commits the updated
`LEADERBOARD.md` and live page onto your PR branch. Merge when it's green.

### If CI fails

| Message | Fix |
|---|---|
| `missing predictions.csv` / `missing submission.yaml` | Both are required; check the directory name matches your slug. |
| `submission.yaml missing field(s): ['model']` | `submitter` and `model` are both mandatory. |
| `submitter '...' doesn't look like a GitHub handle` | Use your actual username — letters, digits, single hyphens, ≤39 chars. |
| `predictions.csv missing column(s): [...]` | Exactly `target`, `cell_line`, `y_pred_proba`. |
| `has duplicate (target, cell_line) rows` | One row per lncRNA × cell line. |
| `is missing N row(s) required by the held-out test set` | You must cover all 5,496 THP1 rows. Usually a filtered or partial run. |
| `has N row(s) not in the held-out test set` | Targets that aren't in the challenge — often a stale target list. |
| No CI run at all | Branch name doesn't start with `submission/`, or nothing under `submissions/` changed. |
| `gzip.BadGzipFile` | You forgot `git lfs pull`. |

---

## The rules

**This is an honour-system board, not a blind benchmark, and it cannot be made
one.** `label` comes from the *published* supplementary tables of the source screen
paper (`data/raw/mmc3.xlsx`, sheets S2F–S2J). THP1's answers are a journal download
away regardless of what this repo ships. We tried encrypting a copy and removed it
as security theatre.

So the ask is simple: **don't look.** Concretely, do not read THP1's labels from —

- `data/raw/mmc3.xlsx` sheet S2J
- `data/processed/lncrna_rra_day14.jsonl.gz` (contains all cell lines, labels
  included — this is what the scorer reads)
- `data/processed/test_THP1.jsonl.gz`, `data/processed/train_THP1.jsonl.gz`
- any historical `predictions.csv` under `results/` with a `y_true` column
- `rra_pvalue` / `fold_change` for THP1 anywhere — `label` is *defined* as
  `rra_pvalue < 0.05 and fold_change < 0`, so those two columns *are* the label.
  (The published features file used to include them while blanking `label` to -1.
  That withheld nothing, and it's fixed — but the columns still exist upstream.)

`data/holdout_thp1/` exists so that following these instructions never puts the
answer key in front of you.

The other rule, less about honesty than about not fooling yourself: **don't tune
against the leaderboard.** Pick your model with CV on the training cell lines.

## How to read the scores

**THP1 has 202 positives, so a single-run AUPRC carries a 95% bootstrap CI roughly
±0.02 wide. Treat smaller gaps as noise.** One positive gene moving in the ranking
is worth ~0.005 AUPRC.

This is not a hypothetical caution. On an earlier chromosome-held-out version of
this task, CV ranking and test AUPRC came out perfectly *anti*-correlated (Spearman
−1.0) across an SVM `C` sweep: the configs that scored best on the test set were the
ones CV liked least. With ~100 positives, a couple of genes landing differently
moved the metric more than any real modelling improvement did. See the "AUPRC does
not respond to model selection" section of [`configs/README.md`](../configs/README.md).

AUPRC is still the right metric for a 3.7%-positive problem — the shortage of
positives is the problem, not the choice of metric.

### What's already been tried

Worth knowing before you spend a weekend re-deriving it. From this repo's history:

| Tried | Outcome |
|---|---|
| **`-mean(training fold_change)`, no learning at all** | **AUROC 0.7085 / AUPRC 0.2000 — beats 4 of 5 real submissions.** Start here. |
| Celligner cell-line embeddings (UMAP-2, PCA-10/70) | Cannot help *on this challenge*: see below. Mixed on AUPRC when it was scored across cell lines. |
| DNABERT-2 embeddings (768-dim, mean-pooled) | Beat k-mers modestly. |
| PCA on embeddings | Looked like a win at one seed; vanished under 4-seed testing. |
| Optuna tuning | Converged to `max_depth=4`, which *underperformed* a fixed `max_depth=9`. |
| SMOTE / random over- and under-sampling | `sampling_ratio` matters more than the method. |
| Exact RBF SVM vs Nystroem approximation | Nystroem(1000) was 52× faster *and* scored better — it regularizes 844 correlated dims. |
| Rank-averaged ensembling, seed averaging, nested tuning | All improved AUROC and *degraded* AUPRC. AUPRC is dominated by the very top of the ranking. |

The strongest entry so far treats it as a guide-level transfer problem rather than a
gene-level one.

### Cell-line features cannot move this metric

Worth internalizing before you spend time on cell embeddings: **the test set is a
single cell line.** Every cell-line-level feature — the one-hot, all 70 Celligner
dimensions — takes the same value on all 5,496 scored rows, so it cannot change their
relative order, and AUROC and AUPRC read nothing but the order. Such features can
still help *indirectly*, by shaping what the model learns from the three training
cell lines, but they carry zero direct signal at scoring time.

Only gene-level signal moves the needle here. That's also why the barebones
pan-essentiality prior does as well as it does.

## Checklist

- [ ] `git lfs pull` done
- [ ] Trained on `train_thp1_holdout.jsonl.gz` only
- [ ] `predictions.csv` has `target`, `cell_line`, `y_pred_proba` for all 5,496 THP1 rows
- [ ] `submission.yaml` has a real GitHub handle in `submitter`, plus `model`
- [ ] `config.yaml` included so someone else can reproduce it
- [ ] Branch is `submission/<handle>-<slug>` on this repo, not a fork
- [ ] You didn't look at THP1's labels
- [ ] Your improvement is bigger than ±0.02, or you validated it across seeds
- [ ] You beat the barebones baseline (AUPRC 0.2000) — or you know why you didn't

Questions, or something in here wrong or unclear? Open an issue.
