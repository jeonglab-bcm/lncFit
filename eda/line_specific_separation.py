"""EDA 5: are cell-line-specific essential lncRNAs separated differently from shared ones?

Every screened lncRNA falls into one group:

  never         not essential in any of the five cell lines
  only <line>   essential in that line and no other (one group per line)
  shared        essential in two or more lines; also reported for three or more, because
                with these hit rates about 97 of the 134 two-line lncRNAs are expected by
                chance alone (independent lines), against about 4 of the 40 lncRNAs in three or
                more lines

Two questions, each answered with the EDA-1 neighbour test (20 nearest neighbours by cosine,
after standardising and PCA, scored as the group fraction among the neighbours; z of the AUPRC
against 100 label shuffles):

  A  group vs never       does the embedding put the lncRNAs of this group near each other,
                          among {group} + {never}? Neighbours are found within that subset.
  B  specific vs shared   among lncRNAs essential somewhere, are the single-line ones separable
                          from the shared ones? (label 1 = essential in exactly one line)

Groups with fewer than 15 members are skipped. The embeddings are the eight of EDA 1, including
the GC + length and random controls.

Usage:
  uv run python eda/line_specific_separation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from embedding_separation import _LINES, load_embeddings_all, load_labels, neighbours  # noqa: E402
from within_tissue_separation import z_score  # noqa: E402

_OUT = "eda/line_specific_separation.csv"
_MIN = 15
_K = 20


def main() -> None:
    genes, y = load_labels()
    Y = np.column_stack([y[c] for c in _LINES])
    n_lines = Y.sum(1)
    groups = {"never": n_lines == 0, "shared (>= 2 lines)": n_lines >= 2,
              "shared (>= 3 lines)": n_lines >= 3}
    for i, c in enumerate(_LINES):
        groups[f"only {c}"] = (n_lines == 1) & (Y[:, i] == 1)

    p = np.array([y[c].mean() for c in _LINES])
    expect = np.zeros(6); expect[0] = 1.0                      # lines independent, same rates
    for pi in p:
        nxt = expect * (1 - pi); nxt[1:] += expect[:-1] * pi; expect = nxt
    print(f"{len(genes):,} screened lncRNAs; essential in k lines (observed / expected if the "
          f"lines were independent):")
    print("  " + ", ".join(f"{k}: {(n_lines == k).sum():,} / {expect[k] * len(genes):,.0f}"
                           for k in range(6)))
    print("groups: " + ", ".join(f"{g} {m.sum()}" for g, m in groups.items()))

    emb, _ = load_embeddings_all(genes)
    rng = np.random.default_rng(0)
    rows = []

    def test(name, idx, lab, question):
        for en, X in emb.items():
            nbr = neighbours(X[idx], _K)
            rows.append({"question": question, "group": name, "n": len(idx),
                         "positives": int(lab.sum()), "embedding": en,
                         "z": round(z_score(lab, nbr, rng), 1)})

    for g, m in groups.items():
        if g == "never" or m.sum() < _MIN:
            continue
        idx = np.flatnonzero(m | groups["never"])
        test(g, idx, m[idx].astype(int), "A: group vs never")
    ess = np.flatnonzero(n_lines >= 1)
    only = (n_lines == 1)[ess].astype(int)
    test("exactly one line vs shared", ess, only, "B: specific vs shared")

    df = pd.DataFrame(rows)
    df.to_csv(_OUT, index=False)
    for q in df.question.unique():
        sub = df[df.question == q]
        pos = sub.drop_duplicates("group").set_index("group").positives
        t = sub.pivot(index="embedding", columns="group", values="z").reindex(list(emb))
        t = t[[g for g in sub.group.unique()]]
        t.columns = [f"{c} [{pos[c]}]" for c in t.columns]
        print(f"\n{q}  (z; members in brackets)")
        print(t.to_string())
    print(f"\n-> {_OUT}")


if __name__ == "__main__":
    main()
