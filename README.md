# lncFit

Which representation of a lncRNA best separates essential from non-essential lncRNAs,
given the cell line? EDA first, models later.

## Data

- `data/raw/mmc2.xlsx`, `mmc3.xlsx` -- supplementary tables of Liang et al., Cell Genomics
  2026: per-lncRNA annotations, guide library, expression (S1C-S1F), and CRISPR-Cas13
  screen results for HAP1, HEK293FT, K562, MDA-MB-231 and THP1.
- `data/processed/lncrna_rra_day14.jsonl.gz` -- one row per (lncRNA, cell line):
  `fold_change`, `rra_pvalue`, `label` (`rra_pvalue < 0.05 and fold_change < 0`).
  5,496 lncRNAs x 5 lines.
- `data/processed/*.npz` -- DNABERT-2 and Nucleotide Transformer sequence embeddings.
- `data/external/` -- Sarropoulos developmental atlas, phastCons elements, GENCODE map.

Rebuild or fetch with the scripts in `scripts/` (`build_processed*.py`, `download_*.py`,
`embed_sequences.py`).

## What earlier work found

Earlier modelling (removed; see git history up to #121) found one signal that transfers
across cell lines -- the lncRNA's own expression in that line -- and that much of the
apparent model skill came from recognising genes rather than learning anything that transfers.
