import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # EDA: k-mer count distributions (k=3-6)

        How skewed is k-mer usage across the lncRNA transcript corpus, and how does
        that skew grow with k? This is exploratory context for why k=3 tends to
        generalize better than k=6 in `results/lncrna_rra_day14/README.md` — a
        4^6 = 4096-slot vocabulary spread over a few thousand training lncRNAs means
        many k-mers are rare, so their frequencies are noisy, high-variance features.

        Uses each lncRNA's own spliced transcript sequence (issue #65's corrected
        feature source), restricted to the 5,496 lncRNAs in the CRISPR screen.
        """
    )
    return


@app.cell
def _():
    import gzip
    import itertools
    import json
    import sys
    from collections import Counter
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))

    BASES = "ACGT"
    KS = [3, 4, 5, 6]
    TRANSCRIPTS_PATH = REPO_ROOT / "data" / "processed" / "body_sequences_transcript.json"
    SCREEN_PATH = REPO_ROOT / "data" / "processed" / "lncrna_rra_day14.jsonl.gz"
    return (
        BASES,
        Counter,
        KS,
        SCREEN_PATH,
        TRANSCRIPTS_PATH,
        gzip,
        itertools,
        json,
        mticker,
        np,
        plt,
    )


@app.cell
def _(mo):
    mo.md("## 1. Load the transcript corpus (screen targets only)")
    return


@app.cell
def _(SCREEN_PATH, TRANSCRIPTS_PATH, gzip, json, mo, np):
    if not TRANSCRIPTS_PATH.exists():
        sequences = []
        _msg = (
            f"**Not found**: `{TRANSCRIPTS_PATH.name}`. Run "
            "`uv run python scripts/download_genome.py --extract` then "
            "`uv run python -m lncfit.sequence --sequence-type transcript` to regenerate "
            "(not committed — reproducible from the hg19 GTF + genome FASTA, issue #65/#66)."
        )
    else:
        with open(TRANSCRIPTS_PATH) as _fh:
            _all_seqs = json.load(_fh)
        with gzip.open(SCREEN_PATH, "rt") as _fh:
            _screen_targets = {json.loads(_line)["target"] for _line in _fh}
        sequences = [_all_seqs[t][0] for t in _screen_targets if t in _all_seqs]
        _lens = np.array([len(s) for s in sequences])
        _msg = (
            f"**{len(sequences):,}** transcript sequences "
            f"({len(_screen_targets):,} screen targets, {len(_screen_targets) - len(sequences)} missing). "
            f"Length: min {_lens.min():,} / median {int(np.median(_lens)):,} / "
            f"max {_lens.max():,} bp, total {_lens.sum():,} bp."
        )
    mo.md(_msg)
    return (sequences,)


@app.cell
def _(mo):
    mo.md("## 2. k-mer count histograms, k=3-6")
    return


@app.cell
def _(BASES, Counter, KS, itertools, sequences):
    def count_kmers(seqs, k):
        """{kmer: total count across the corpus} for every valid ACGT-only window."""
        counts = Counter()
        for seq in seqs:
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                if all(c in BASES for c in kmer):
                    counts[kmer] += 1
        return counts

    kmer_counts_by_k = {}
    for _k in KS:
        _counts = count_kmers(sequences, _k)
        _vocab = ["".join(p) for p in itertools.product(BASES, repeat=_k)]
        kmer_counts_by_k[_k] = {kmer: _counts.get(kmer, 0) for kmer in _vocab}
    return (kmer_counts_by_k,)


@app.cell
def _(KS, kmer_counts_by_k, mticker, plt):
    # Categorical palette, fixed order (blue/aqua/yellow/violet) — one hue per k.
    _colors = {3: "#2a78d6", 4: "#1baf7a", 5: "#eda100", 6: "#4a3aa7"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for _ax, _k in zip(axes.flat, KS):
        _counts = list(kmer_counts_by_k[_k].values())
        _n_possible = 4**_k
        _n_observed = sum(1 for c in _counts if c > 0)
        _ax.hist(_counts, bins=40, color=_colors[_k])
        _ax.set_yscale("log")
        _ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        _ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        _ax.set_title(f"k={_k}  ({_n_observed:,}/{_n_possible:,} k-mers observed)", fontsize=10)
        _ax.set_xlabel("count across corpus", fontsize=9)
        _ax.set_ylabel("# of k-mers (log scale)", fontsize=9)
        for _spine in ("top", "right"):
            _ax.spines[_spine].set_visible(False)

    fig.suptitle("k-mer count distribution across the lncRNA transcript corpus", fontsize=12)
    plt.tight_layout()
    fig
    return (fig,)


@app.cell
def _(mo):
    mo.md(
        """
        Higher k => more possible k-mers (4^k) spread over the same ~4M bp of total
        sequence, so the count distribution stretches further to the right and gets
        more zero/low-count k-mers on the left. That's the mechanism behind k=3
        outperforming k=6 in the classifier: a 6-mer's observed frequency in any one
        lncRNA is a much noisier estimate than a 3-mer's.

        ## 3. Skew summary + most/least common k-mers
        """
    )
    return


@app.cell
def _(mo):
    k_dropdown = mo.ui.dropdown(options=["3", "4", "5", "6"], value="6", label="k-mer size")
    k_dropdown
    return (k_dropdown,)


@app.cell
def _(k_dropdown, kmer_counts_by_k, mo, np, pd):
    _k = int(k_dropdown.value)
    _counts = kmer_counts_by_k[_k]
    _vals = np.array(list(_counts.values()))
    _n_possible = 4**_k
    _n_observed = int((_vals > 0).sum())

    _summary = mo.md(
        f"""
        **k={_k}**: {_n_observed:,}/{_n_possible:,} observed ({_n_observed / _n_possible:.1%}).
        Count range [{_vals.min():,}, {_vals.max():,}], median {int(np.median(_vals)):,},
        mean {_vals.mean():.1f}, std {_vals.std():.1f}
        (coefficient of variation {_vals.std() / _vals.mean():.2f} — higher = more skewed).
        """
    )

    _sorted_items = sorted(_counts.items(), key=lambda kv: kv[1], reverse=True)
    _top = pd.DataFrame(_sorted_items[:10], columns=["kmer", "count"])
    _bottom = pd.DataFrame(_sorted_items[-10:], columns=["kmer", "count"])

    mo.vstack([_summary, mo.hstack([mo.vstack([mo.md("**Top 10**"), _top]), mo.vstack([mo.md("**Bottom 10**"), _bottom])])])
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


if __name__ == "__main__":
    app.run()
