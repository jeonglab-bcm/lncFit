# How to enter the lncFit challenge

Predict which lncRNAs are essential in a cell line your model has never seen.

Live board: **https://jeonglab-bcm.github.io/lncFit/**

**This is the only document you need.** The READMEs all link here rather than
repeat any of it. If you read one section, read [the 60-second version](#the-60-second-version).

| | |
|---|---|
| **Train on** | [HAP1, K562, MDA-MB-231](../data/holdout_thp1/train_thp1_holdout.jsonl.gz) — 16,488 rows, labels included |
| **Predict** | [THP1](../data/holdout_thp1/holdout_thp1_features.jsonl.gz) — 5,496 rows, labels withheld |
| **Ranked by** | AUPRC (AUROC shown alongside) |
| **Positives in the test set** | 202 (3.7%) |
| **Excluded entirely** | HEK293FT — not a real cancer line, no Celligner coordinates |

A row is one lncRNA × cell line. `label = 1` means knocking that lncRNA out
depleted that cell line by day 14 (`rra_pvalue < 0.05 and fold_change < 0`).

The task is *cross-cell-line generalization*: nothing about THP1's own response is
in your training data. This is harder than predicting an unseen lncRNA in a cell
line you've trained on, and it's why scores here look low — a leading eligible AUPRC
of ~0.17 against a 0.037 base rate is a ~4.6× enrichment, not a weak result.

---

## What you are actually predicting

### The biology, briefly

**Long non-coding RNAs** are transcripts over ~200 nt that are not translated into
protein. Humans have tens of thousands of them, and for the large majority nobody
knows what — if anything — they do. Unlike protein-coding genes, they have no reading
frame, no codon structure, and no conserved domains to recognise, so the usual tricks
for inferring function from sequence mostly do not apply. Sequence conservation across
species is also weak, which removes the other standard signal.

The 5,496 lncRNAs here were knocked down with **CRISPR-Cas13** (RNA-targeting, so it
degrades the transcript rather than cutting DNA) in a pooled screen. Cells were
sequenced at day 0, day 7, and **day 14**; a lncRNA whose guides *dropped out* of the
population by day 14 was needed for those cells to grow. That dropout is
`fold_change`; `rra_pvalue` is the robust-rank-aggregation statistic over that
lncRNA's guides. Hence `label = (rra_pvalue < 0.05 and fold_change < 0)`.

**Why only ~3.7% are hits.** Most lncRNA knockdowns do nothing measurable to
proliferation. That is the genuine biological finding of screens like this, not a
defect in the data — and it is why AUPRC, not accuracy, is the metric.

**Why this is hard from sequence alone.** You are being asked whether degrading a
particular transcript kills a particular cell line, given only that transcript's
sequence. There is no reading frame to score, little conservation to exploit, and the
relevant mechanism (what the lncRNA regulates, and whether that pathway matters in
*this* lineage) is not visibly encoded in its nucleotides. The measured ceiling on
this board — ~0.17 AUPRC, ~4.6× enrichment — is best read as a real limit, not a
modelling failure.

### The cell lines

| Line | Lineage | Origin | Hits / 5,496 | Rate | In this challenge |
|---|---|---|---|---|---|
| **THP1** | Myeloid | Acute monocytic leukemia (AML) | 202 | 3.7% | **the test set** |
| **HAP1** | Myeloid | Near-haploid, derived from CML (KBM-7) | 235 | 4.3% | training |
| **K562** | Myeloid | Chronic myelogenous leukemia, BCR-ABL⁺ | 401 | 7.3% | training |
| **MDA-MB-231** | Breast | Triple-negative breast adenocarcinoma | 157 | 2.9% | training |
| HEK293FT | Embryonic kidney | Adenovirus-transformed, **not a cancer line** | 254 | 4.6% | **excluded** |

Lineage assignments come from this repo's own Celligner analysis — see
[`data/external/README.md`](../data/external/README.md), which also documents that
K562 and THP1 land in their lineage cluster cleanly (15/15) while HAP1 is a genuine
unexplained outlier within Myeloid, and MDA-MB-231 sits in a noisier Breast cluster.

**Why HEK293FT is excluded:** it is not a cancer line (embryonic kidney, transformed
with adenoviral DNA), it has no Celligner coordinates, and "what does this cell need
to proliferate" means something different in an immortalised non-tumour line. Its rows
are dropped from scoring entirely.

**Three of the four scored lines are myeloid, including the test line.** You might
expect that to make the two myeloid training lines the most useful. It does not:

| Predicting THP1 from one line's measured depletion | AUPRC | Same lineage? |
|---|---|---|
| MDA-MB-231 | **0.2133** | no (Breast) |
| HAP1 | 0.1581 | yes (Myeloid) |
| K562 | 0.1019 | yes (Myeloid) |

The one breast line is the *best* single predictor of THP1, and this holds for every
target line — MDA-MB-231 is the strongest source everywhere, K562 the weakest.
Transferability here tracks **screen quality, not lineage**: K562 has by far the most
hits (401, 7.3%) and the lowest depletion correlation with every other line (Spearman
0.16–0.24), while MDA-MB-231 has the fewest hits (157, 2.9%) and the highest (up to
0.29 with THP1). Pairwise hit-set overlap says the same thing — THP1∩MDA-MB-231
Jaccard 0.125 versus THP1∩K562 0.060.

Read that as a caution about *weighting your training lines equally*, not as licence
to use depletion as a feature — which
[the rules forbid](#no-measured-depletion-as-a-feature--any-cell-line-any-day). The
numbers above are descriptive analysis of the training data, produced to explain the
task; reproduce them with `scipy` and `sklearn` on the training lines if you want.

Cross-line depletion correlations are low in absolute terms (Spearman **0.16–0.29**
for every pair). That is the hard ceiling on transfer, and it is why no model on this
board clears ~0.24 even when allowed to use depletion directly.

---

## The 60-second version

```bash
git clone https://github.com/jeonglab-bcm/lncFit.git && cd lncFit
git lfs pull
uv sync

# train a compliant sequence-based model (k-mer XGBoost, leave-one-cell-line-out)
uv run python scripts/run_cellline_loco.py \
    --config configs/cellline_loco/xgboost_kmer.yaml

# copy its predictions.csv into a submission dir, add submission.yaml, then:
uv run python scripts/score_submission.py \
    results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/YOUR-HANDLE-kmer
```

That reproduces the `baseline-xgboost-kmer` entry, **AUPRC 0.1636**. Push it on a
`submission/` branch ([§4](#4-open-the-pr)) and it appears on the board. Full detail in
[§2](#2-make-a-prediction) and [§3](#3-package-the-submission).

### Read this before you pick a method

The single most important fact about this challenge:

| Model | AUPRC | Eligible? |
|---|---|---|
| `-mean(training fold_change)` — one line, no parameters | 0.2000 | **No** — measured depletion |
| Best sequence-only entry (DNABERT-2 + distance) | 0.1696 | Yes |
| Tuned DNABERT-2 + Celligner + Optuna, 50 trials | 0.1268 | Yes |

A one-liner that averages three columns of measured depletion beats every model that
tried to learn biology from sequence. That is why **measured depletion is now banned as
a feature** ([the rules](#no-measured-depletion-as-a-feature--any-cell-line-any-day)) —
it answers "is this gene pan-essential?", which needs no sequence understanding and
cannot generalise to an unscreened lncRNA.

So the honest bar is **0.1696, not 0.2000**, and clearing it is genuinely open. Measured
attempts to date, all fold-safe and selected on training-line CV, land in a narrow band:

| Approach | training-line LOCO | THP1 |
|---|---|---|
| k-mer / DNABERT-2 / both, 9 configs (depth 3–7) | 0.1585–0.1726 | 0.1331–0.1585 |

and remember the **±0.06 CI** — everything in that table is statistically one number.
[`scripts/make_barebones_submission.py`](../scripts/make_barebones_submission.py) still
exists, now labelled ineligible, as the demonstration of the shortcut it represents.

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
| [`data/holdout_thp1/train_thp1_holdout.jsonl.gz`](../data/holdout_thp1/train_thp1_holdout.jsonl.gz) | your training data, labels included |
| [`data/holdout_thp1/holdout_thp1_features.jsonl.gz`](../data/holdout_thp1/holdout_thp1_features.jsonl.gz) | the rows to predict — `label`, `rra_pvalue` and `fold_change` are all absent |

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

[`configs/pipeline/thp1_holdout_starter.yaml`](../configs/pipeline/thp1_holdout_starter.yaml)
is committed and wired up for this challenge. It trains XGBoost on 5-mer frequencies
via [`scripts/run_pipeline.py`](../scripts/run_pipeline.py) and writes a submittable
`predictions.csv` — a real model rather than a one-liner, though note it scores *below*
the barebones baseline (AUPRC 0.1456 vs 0.2000), which tells you something about how
much of this task sequence features actually explain:

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

Now change something — [`configs/README.md`](../configs/README.md) documents every knob
(model, feature type, embeddings, PCA, resampling, tuning, CV). Any model at all is fair
game; the pipeline is a convenience, not a requirement. If you write your own, just emit
a CSV with the three required columns.

### The CV number is optimistic — don't select on it

The starter's stratified K-fold puts the *same* cell lines in train and validation,
so it flatters models that memorize cell-line-specific signal, which is exactly
what this task punishes. For an honest estimate, hold out a training cell line:

```bash
uv run python scripts/run_cellline_loco.py --config configs/cellline_loco/xgboost_kmer_celligner2.yaml
```

[`scripts/run_cellline_loco.py`](../scripts/run_cellline_loco.py) rotates each cell line
into the validation slot, which mirrors the real task.

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

### Check it before you push

```bash
python3 scripts/score_submission.py \
    results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/<your-handle>-<slug>
```

[`scripts/score_submission.py`](../scripts/score_submission.py) runs the same two
functions CI runs, against the same [`challenge.yaml`](../results/lncrna_rra_day14_thp1_holdout/leaderboard/challenge.yaml),
so a green run here is a green run there and the numbers match exactly. It catches
every format error in the [table below](#if-ci-fails) without costing you a PR round
trip, and exits non-zero if the submission is invalid.

It does have to read the held-out labels to compute a score. That's a sanity check, not
a licence to iterate against it — see [don't tune against the leaderboard](#the-rules)
and [how to read the scores](#how-to-read-the-scores).

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

`python3 scripts/score_submission.py <your-submission-dir>` reproduces every one of
these locally, with the same message.

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

### No measured depletion as a feature — any cell line, any day

**Your model may not use measured knockdown outcomes as input features.** Not
THP1's, and *not the training cell lines' either*. Specifically banned as features:

- `fold_change` / log2FC, from any cell line, any day
- `rra_pvalue`, from any cell line
- guide-level depletion from `data/processed/screen_records.jsonl.gz`
- anything derived from the above (per-gene mean depletion, hit counts across
  training lines, replicate-consistency summaries, depletion order statistics)

The training lines' **`label` is still your supervision target** — without it there
is no supervised task. The rule is about what goes into `X`, not `y`.

**Why:** this challenge is meant to ask *can you predict lncRNA essentiality from
sequence?* A model that averages measured depletion in three other cell lines is
answering a different and much easier question — "is this gene pan-essential?" — and
it needs no sequence understanding whatsoever. It also cannot generalise to an
unscreened lncRNA, which is the only case where a predictor would actually be
useful. The zero-parameter baseline `-mean(training fold_change)` scores 0.2000
AUPRC, beating four of five sequence-based entries; that gap measures the shortcut,
not progress.

Legal: transcript sequence and anything computed from it (k-mers, DNABERT-2 or other
sequence-model embeddings, length, GC, structure), guide-design descriptors, static
annotation (`distance_to_closest_pc_gene`, strand, chrom), and cell-line covariates
such as Celligner. Note that cell-line features cannot change your THP1 ranking at
all — every scored row is THP1, so they are constant across the whole test set.

This rule was added **2026-07-28** and is not retroactive blame: entries submitted
before it declared their features openly and complied with the rules as written.
Affected entries are marked ineligible on the board, with their scores kept visible.

The other rule, less about honesty than about not fooling yourself: **don't tune
against the leaderboard.** Pick your model with CV on the training cell lines.

## How to read the scores

**THP1 has 202 positives, so a single-run AUPRC carries a 95% bootstrap CI roughly
±0.06 wide. Treat smaller gaps as noise.** One positive gene moving in the ranking
is worth ~0.005 AUPRC.

Measured, not estimated — 2000 row-bootstrap resamples of the submitted
`predictions.csv` files:

| Entry | AUPRC | 95% CI | width |
|---|---|---|---|
| crosscell-depletion-guide | 0.2364 | [0.1819, 0.3030] | 0.121 |
| dnabert2-dist | 0.1696 | [0.1212, 0.2234] | 0.102 |

So **any gap under ~0.06 AUPRC is not measurable on this board.** Chasing a
"marginal" improvement over the leader is chasing noise. (The 0.067 gap between
those two *is* real — paired bootstrap p≈0.001 — so the top slot itself is not an
artefact. It is small gaps that mean nothing.)

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
| ~~`-mean(training fold_change)`~~ | AUPRC 0.2000 with no learning — **now ineligible**, and the reason the feature ban exists. |
| Celligner cell-line embeddings (UMAP-2, PCA-10/70) | Cannot help *on this challenge*: see below. Mixed on AUPRC when it was scored across cell lines. |
| DNABERT-2 embeddings (768-dim, mean-pooled) | Beat k-mers modestly. |
| PCA on embeddings | Looked like a win at one seed; vanished under 4-seed testing. |
| Optuna tuning | Converged to `max_depth=4`, which *underperformed* a fixed `max_depth=9`. |
| SMOTE / random over- and under-sampling | `sampling_ratio` matters more than the method. |
| Exact RBF SVM vs Nystroem approximation | Nystroem(1000) was 52× faster *and* scored better — it regularizes 844 correlated dims. |
| Rank-averaged ensembling, seed averaging, nested tuning | All improved AUROC and *degraded* AUPRC. AUPRC is dominated by the very top of the ranking. |

### Eligible sequence-only search, 2026-07-28 (~300 configs)

A systematic sweep under the feature ban, everything fold-safe and selected on
training-line LOCO with THP1 scored once. Reproduce the descriptive numbers with
[`scripts/describe_challenge_data.py`](../scripts/describe_challenge_data.py).

| Tried | Result |
|---|---|
| DNABERT-2 **PCA-32** + static/guide features, depth 5 | **Best eligible model found: trainLOCO 0.1417, THP1 0.1599** |
| DNABERT-2 full 768-d vs PCA-32/64/128 | PCA-**32** won. More representation was consistently *worse*. |
| k-mers at k=4, 5, 6 + reverse-complement-collapsed variants | All below DNABERT-2. k=6 (4,096-d) worst — dimensionality without signal. |
| Static/guide-design features alone (16 dims) | 0.0987 trainLOCO. Length, GC, homopolymers and guide composition carry almost nothing. |
| One row per (target, cell line) vs one row per target | **Row-level won by a wide margin** (0.1417 vs 0.1297). Collapsing to a soft per-gene label loses real information, even though the features are identical across a gene's rows. |
| Soft-label regression vs binary classification | Classification won at row level; regression won at target level. |
| `scale_pos_weight` 1 vs 5 | No consistent benefit. |
| Logistic regression (C = 0.01–1) for model diversity | Below XGBoost everywhere. |
| **Top-k ensembling (k = 2, 3, 5, 8, 12)** | **Every ensemble scored *below* the single best model.** Consistent with the seed/rank-averaging result above: AUPRC is set by the very top of the ranking, and averaging blurs it. |

**Nothing eligible has yet beaten 0.1696** — and 0.1599 vs 0.1696 is well inside the
±0.06 CI, so treat them as tied rather than ranked. The honest read is that eligible
performance saturates near **0.15–0.17 AUPRC (~4–4.6× enrichment)**, and that this
reflects how little transcript sequence says about cell-line-specific essentiality
rather than a shortage of tuning. If you clear it convincingly, that is a real result.

Two dead ends worth not repeating: adding guide-level depletion and replicate
consistency to a cross-cell model made it *worse* (0.2010 vs 0.2061 trainLOCO), and
selecting a config by its THP1 score is actively misleading — across 12 configs, THP1
AUPRC was **uncorrelated** with the training-line LOCO signal, and the config that
scored best on THP1 was not the one honest selection picked.

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
- [ ] `scripts/score_submission.py <dir>` prints "Valid" — this covers the format
      checks (columns, all 5,496 rows, no duplicates, real GitHub handle)
- [ ] `config.yaml` included so someone else can reproduce it
- [ ] Branch is `submission/<handle>-<slug>` on this repo, not a fork
- [ ] You didn't look at THP1's labels while building the model
- [ ] Your improvement is bigger than ±0.02, or you validated it across seeds
- [ ] You beat the barebones baseline (AUPRC 0.2000) — or you know why you didn't

Questions, or something in here wrong or unclear? Open an issue.
