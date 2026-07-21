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
match `Celligner_info.csv`'s published `UMAP_1`/`UMAP_2` values, though the
qualitative structure matches (verified: K562-THP1, both blood/leukemia lineage,
are much closer to each other than either is to MDA-MB-231, a breast line, same
as in the original publication).

**Caveat on HAP1's position:** it lands near MDA-MB-231 rather than near the
other blood-lineage lines (K562, THP1) in this alignment. This may be a genuine
biological signal — HAP1 is known to grow adherently (unlike typical suspension
leukemia lines) and has been extensively altered by its haploidization/selection
process relative to its CML-derived parental line (KBM-7) — but it has not been
independently verified beyond the sanity checks above, and should be treated with
appropriate caution.

**Known deviations from the original method** (approximations, not exact
reproductions): clustering uses a plain KNN-graph + Louvain (`igraph`) rather than
Seurat's specific SNN-graph construction; this only affects the intermediate
cluster-mean-subtraction and DE-gene-subset-selection steps, not the core cPCA/MNN
math. Tumor samples have no lineage/subtype annotations (Treehouse clinical
metadata was not fetched, since it's not used mathematically by the algorithm).

Reproduction script (not checked into this repo — large raw inputs, one-off run):
available on request; downloads ~6.2GB of raw expression data and takes ~10
minutes to run on a 32GB-RAM machine.
