# eda/

Exploratory analysis. Findings are logged in `docs/index.html`.

- `embedding_separation.py` — EDA 1: for each embedding and cell line, are essential
  lncRNAs each other's nearest neighbours more than chance? Output: `embedding_separation.csv`.

- `kmer_distributions.py` — marimo notebook: k-mer count distributions (k=3-6)
  across the lncRNA transcript corpus (issue #65's corrected sequences). Static
  snapshot: `kmer_count_histograms.png`. Run interactively with
  `uv run marimo edit eda/kmer_distributions.py` (requires
  `data/processed/body_sequences_transcript.json` — see the notebook's first
  cell for how to regenerate it if missing).
