# External reference data

## `celligner_cell_line_umap.csv`

Cell-line embedding coordinates for issue #78 (replace/augment one-hot cell-line
encoding with a real transcriptomic similarity embedding), built by re-running the
[Celligner](https://github.com/broadinstitute/celligner) alignment method
(Warren et al., *Nat Commun* 2021) from scratch against **current** DepMap data,
so that HAP1 (added to DepMap after the original 2020 Celligner data release) is
included.

**Why re-run instead of using the published output:** the published
`Celligner_data` Figshare release (all 5 versions checked) is frozen at DepMap
19Q4 and does not include HAP1 at all. HEK293FT is not a cancer cell line and has
never been in CCLE/DepMap under any release, so it is not included here either
and falls back to a zero vector wherever this embedding is used.

**Method:** faithfully reimplemented from the original R source
(`Celligner_methods.R`, `Celligner_helpers.R`, `global_params.R` in
[broadinstitute/Celligner_ms](https://github.com/broadinstitute/Celligner_ms)) —
per-dataset mean-centering, Louvain clustering, cluster-mean-subtracted
contrastive PCA (top 10 components, top 4 regressed out), limma-ranked
differentially-expressed genes for the MNN neighbor subset, mutual-nearest-neighbor
batch correction (custom tricube-weighted port of `modified_mnnCorrect`), a final
70-D PCA, and UMAP (`n_neighbors=10, min_dist=0.5, metric=euclidean`) — using
`irlba`/`uwot`/`igraph`/`FNN`/`limma` in R instead of Seurat/batchelor (lighter
dependency footprint; `uwot` is what Seurat's `RunUMAP` wraps internally anyway).

Inputs:
- TCGA/Treehouse tumor expression: `TumorCompendium_v10_PolyA_hugo_log2tpm_58581genes_2019-07-25.tsv`
  (UCSC Xena Treehouse) — unchanged since the original Celligner publication.
- CCLE cell-line expression: DepMap 24Q4 Public (`OmicsExpressionProteinCodingGenesTPMLogp1.csv`),
  the current release, which includes HAP1 (`ACH-002475`).
- Gene panel: HGNC `hgnc_complete_set_7.24.2018.txt` (same snapshot Celligner
  originally used), restricted to protein-coding + other non-(non-coding
  RNA/pseudogene) genes shared between both expression matrices (18,460 genes).

**This is a fresh, self-consistent realignment, not a projection onto the
published coordinates** — every cell line's coordinates (including K562,
MDA-MB-231, THP1) were recomputed from this run, so they will not numerically
match `Celligner_info.csv`'s published `UMAP_1`/`UMAP_2` values.

## Validation

Eyeballing distances among just our 4 target cell lines isn't a real validation —
it can't distinguish "the alignment reflects real biology" from "these 4 points
happen to be arranged plausibly by chance." Instead: computed each of the 4
targets' **k=15 nearest CCLE neighbors** in the aligned UMAP space, and checked
what fraction share the *same Oncotree lineage* (`Model.csv`'s
`OncotreeLineage`) — then compared that against the same nearest-neighbor-purity
statistic computed for **all 1,668 lineage-annotated CCLE cell lines**, not just
our 4, as a baseline for what "good" and "bad" purity look like in this specific
alignment.

**Baseline (all 1,668 CCLE lines):** mean same-lineage purity 53.8%, median
53.3% — well above what 34 categories of very unequal size would give by chance.
Purity varies a lot by lineage: well-populated ones cluster very cleanly
(Lymphoid 96.1% n=186, Myeloid 88.8% n=74, Skin 74.9% n=120), while small/rare
lineages are noisy (Adrenal Gland, Ampulla of Vater, Vulva/Vagina: 0%, but
n=1-4 each). This confirms the alignment mechanism itself is sound and
recovering real biological structure, which is the necessary context for judging
the 4 target lines below.

| cell line | true lineage | same-lineage neighbors (of 15) | lineage's own average purity | run-to-run coordinate shift |
|---|---|---|---|---|
| **K562** | Myeloid | **15/15** | 88.8% (n=74) | 2.51 |
| **THP1** | Myeloid | **15/15** | 88.8% (n=74) | 1.25 (most stable) |
| MDA-MB-231 | Breast | 0/15 | 54.9% (n=70) | 3.10 |
| **HAP1** | Myeloid | **0/15** | 88.8% (n=74) | **7.04 (least stable)** |

- **K562, THP1: validated.** Both land with maximal same-lineage purity, in a
  lineage that already clusters very cleanly overall.
- **MDA-MB-231: 0/15, but plausible.** Breast lines only average 54.9% purity
  to begin with (a noisier lineage than Myeloid), and MDA-MB-231 is a
  well-documented mesenchymal-like, triple-negative outlier among breast cancer
  cell lines (unlike more common epithelial/luminal breast lines) — a real
  discrepancy from naive expectation, but with a defensible biological
  explanation. Not confirmed further than that.
- **HAP1: 0/15, and NOT explained.** Myeloid lines cluster at 88.8% purity
  overall — HAP1 is a genuine outlier *within an otherwise very clean lineage*,
  not just "different from 3 comparison points." Checked for an obvious
  technical explanation and found none: HAP1's raw expression has no unusual
  zero-count or NaN rate versus K562/THP1, and its raw whole-transcriptome
  correlation to K562 (0.858), THP1 (0.833), and MDA-MB-231 (0.855) are all
  similar — raw correlation doesn't discriminate here either way. HAP1's
  position is also the **least stable of the 4 across two independent runs of
  the identical pipeline on identical data** (shift of 7.04 UMAP units, vs.
  1.25–3.10 for the other 3). Three independent signals (lineage-purity outlier
  in an otherwise-clean lineage, no raw-data explanation, worst run-to-run
  stability) all point the same way.

**Bottom line: treat HAP1's specific coordinates in `celligner_cell_line_umap.csv`
with real skepticism** (flagged as `"UNRELIABLE"` in that file's `reliability`
column) until this is investigated further — kept rather than zero-filled per
explicit request, not because it's been shown trustworthy. K562 and THP1 are
solid; MDA-MB-231 has a plausible but unconfirmed explanation for its own lower
purity.

Validation figure: `results/lncrna_rra_day14/celligner_embedding_comparison/alignment_validation.png`
(`scripts/plot_celligner_validation.py`) — all 1,673 CCLE cell lines colored by
Oncotree lineage, tumor samples as a gray background cloud, target cell lines
circled and labeled.

**Known deviations from the original method** (approximations, not exact
reproductions): clustering uses a plain KNN-graph + Louvain (`igraph`) rather than
Seurat's specific SNN-graph construction; this only affects the intermediate
cluster-mean-subtraction and DE-gene-subset-selection steps, not the core cPCA/MNN
math. Tumor samples used in the validation figure's background have disease-type
annotations from Treehouse's `clinical_TumorCompendium_v10_PolyA_2019-07-25.tsv`
(joined on sample ID, ~99.99% match rate), but tumor lineage/disease is not used
anywhere in the alignment math itself, only for the validation plot.

Reproduction script (not checked into this repo — large raw inputs, one-off run):
available on request; downloads ~11GB of raw expression + clinical data and takes
~10 minutes to run on a 32GB-RAM machine. Re-running produces qualitatively
similar but not numerically identical coordinates (UMAP + Louvain clustering are
not perfectly deterministic even with a fixed seed) — see the run-to-run shift
column above.
