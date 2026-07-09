# eda/

Exploratory analysis — not part of the validated pipeline in `results/`. Each
notebook here is a standalone question, not a claim; promote anything load-bearing
into `results/` + a script once it's actually informing a modeling decision.

- `kmer_distributions.py` — marimo notebook: k-mer count distributions (k=3-6)
  across the lncRNA transcript corpus (issue #65's corrected sequences). Static
  snapshot: `kmer_count_histograms.png`. Run interactively with
  `uv run marimo edit eda/kmer_distributions.py` (requires
  `data/processed/body_sequences_transcript.json` — see the notebook's first
  cell for how to regenerate it if missing).
