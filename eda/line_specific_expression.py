"""EDA 7: line-specific lncRNAs, plus essential ones from anywhere -- does sequence give three groups?

Hypothesis: collect the lncRNAs that are expressed specifically in one cell line, add the lncRNAs
that are essential in any cell line, and look at sequence embeddings. If sequence separates the
line-specific lncRNAs, three groups should appear:

  N  not essential in any cell line               (line-specific expression, never essential)
  T  essential in its own line only               (specifically expressed in line L, essential in L
                                                   and in no other line)
  P  essential regardless of line                 (essential in two or more lines; the control: its
                                                   essentiality does not depend on one line)
  E  essential, but not where it is expressed     (specifically expressed in L, essential in exactly
                                                   one other line; shown for completeness)
  B  broadly expressed, never essential           (expressed, tau < 0.8, essential nowhere): the fair
                                                   comparison for P, because N is line-specific by
                                                   construction while P is not, so P versus N differs
                                                   in expression by design

Line-specific expression: mRNA-seq TPM in the five lines (mmc2 S1E); an lncRNA is specific to a line
when it is expressed (TPM >= 1) in at least one line and its tau over the five lines is >= 0.8; the
line is the one with the highest TPM. The five lines are pooled for the tests because each line alone
has only 10-20 members of T.

Test (EDA 1's neighbour test): for each group against N, the 20 nearest neighbours inside {group} + N,
scored as the fraction in the group; z of the AUPRC against 100 label shuffles. Also reported:
how many lncRNAs fall in each group.

Usage:
  uv run python eda/line_specific_expression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from embedding_separation import _LINES, load_embeddings_all, load_labels, neighbours  # noqa: E402
from within_tissue_separation import z_score  # noqa: E402

_OUT = "eda/line_specific_expression_groups.csv"
_TAU, _MIN = 0.8, 15


def main() -> None:
    genes, y = load_labels()
    emb, mrna = load_embeddings_all(genes)
    Y = np.column_stack([y[c] for c in _LINES])
    n_ess = Y.sum(1)
    M = np.column_stack([np.expm1(mrna[c]) for c in _LINES])
    top = M.argmax(1)
    mx = M.max(1, keepdims=True)
    tau = (1 - M / np.where(mx > 0, mx, 1)).sum(1) / (len(_LINES) - 1)
    specific = (M.max(1) >= 1) & (tau >= _TAU)
    own = Y[np.arange(len(genes)), top] == 1                    # essential in the line it is specific to

    groups = {
        "N  never essential": specific & (n_ess == 0),
        "T  essential in own line only": specific & (n_ess == 1) & own,
        "P  essential in 2+ lines": n_ess >= 2,
        "E  essential only elsewhere": specific & (n_ess == 1) & ~own,
        "B  broad, never essential": (M.max(1) >= 1) & (tau < _TAU) & (n_ess == 0),
    }
    pd.DataFrame({"lncRNA": genes, "specific_line": np.where(specific, np.array(_LINES)[top], ""),
                  "tau": tau.round(3), "n_lines_essential": n_ess,
                  "group": [next((k[0] for k, m in groups.items() if m[i]), "") for i in range(len(genes))]
                  }).to_csv(_OUT, index=False)

    print(f"line-specific expressed lncRNAs (tau >= {_TAU}, TPM >= 1): {specific.sum():,} of {len(genes):,}")
    print("groups: " + ", ".join(f"{k.split()[0]} {m.sum()}" for k, m in groups.items()))
    for i, c in enumerate(_LINES):
        s = specific & (top == i)
        print(f"  specific to {c:<11} {s.sum():>4}  T: {(groups['T  essential in own line only'] & s).sum():>3}  "
              f"E: {(groups['E  essential only elsewhere'] & s).sum():>3}")

    rng = np.random.default_rng(0)
    rows = []
    N = groups["N  never essential"]
    for gname, gm in groups.items():
        if gname[0] in "NB":
            continue
        idx = np.flatnonzero(gm | N)
        lab = gm[idx].astype(int)
        for en, X in emb.items():
            nbr = neighbours(X[idx], 20)
            rows.append({"group": gname, "members": int(gm.sum()), "embedding": en,
                         "z": round(z_score(lab, nbr, rng), 1)})
    # T versus P directly: do the two kinds of essential lncRNA differ from each other?
    tp = groups["T  essential in own line only"] | groups["P  essential in 2+ lines"]
    idx = np.flatnonzero(tp)
    lab = groups["T  essential in own line only"][idx].astype(int)
    for en, X in emb.items():
        rows.append({"group": "T versus P", "members": int(lab.sum()), "embedding": en,
                     "z": round(z_score(lab, neighbours(X[idx], 20), rng), 1)})

    # P against B: both broadly expressed, so expression is no longer different by construction.
    Bm, Pm = groups["B  broad, never essential"], groups["P  essential in 2+ lines"]
    idx = np.flatnonzero(Pm | Bm)
    lab = Pm[idx].astype(int)
    for en, X in emb.items():
        rows.append({"group": "P versus B", "members": int(Pm.sum()), "embedding": en,
                     "z": round(z_score(lab, neighbours(X[idx], 20), rng), 1)})

    df = pd.DataFrame(rows)
    df.to_csv("eda/line_specific_expression_separation.csv", index=False)
    t = df.pivot(index="embedding", columns="group", values="z").reindex(list(emb))
    t = t[list(df.group.unique())]
    mem = df.drop_duplicates("group").set_index("group").members
    t.columns = [f"{c} [{mem[c]}]" for c in t.columns]
    print("\nz (group against N; 'T versus P' compares T with P; 'P versus B' compares P with broadly expressed never-essential lncRNAs); members in brackets")
    print(t.to_string())
    print(f"\n-> {_OUT}, eda/line_specific_expression_separation.csv")


if __name__ == "__main__":
    main()
