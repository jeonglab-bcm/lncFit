"""SSL step 4: make the sequence-vs-expression embedding tissue-specificity aware.

Step 3 kept model A (k-mers vs developmental atlas + cell-line TPM, hard InfoNCE) and found
contrastive learning *against* tissue specificity does not help. Here tissue specificity is
an explicit requirement instead: a linear head on each 128-d embedding must reconstruct the
lncRNA's tissue profile (log2 mean RPKM in 7 organs + tau, standardised), and its error is
added to the contrastive loss with weight `lambda`. Linear, so the embedding has to carry
tissue specificity in a form any downstream linear read-out can use.

  lambda = 0   model A as in step 3 (the reference)
  lambda = 1   tissue-aware

Measured, mean (sd) over seeds:
  tissue R2    5-fold ridge probe from the embedding to the tissue profile, held-out R2,
               averaged over the 8 targets; also tau alone. 1.0 = fully encoded.
  z            the EDA-1 essential neighbour test per cell line, on the 5,496 screened
               lncRNAs -- tissue awareness must not cost separation.

The tissue profile is computed from the same atlas the expression view contains, so the
expression embedding is expected to encode it already; the question that matters most is
the sequence embedding.

Held-out genes: by default the encoders train only on the 26,165 unscreened lncRNAs, and every
number is measured on the 5,496 screened ones, which training never saw. Training and scoring
on the same genes lets the sequence encoder memorise each gene's own expression -- raw k-mers
probe at tissue R2 -0.03, yet an encoder trained on those genes reached 0.48 on them.
--no-holdout reproduces that in-sample setting.

Usage:
  uv run python contrastive/train_tissue_aware.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from train_clip import Encoder, augment, clip_loss, embed, separation_z, standardise  # noqa: E402
from train_tissue import tissue_view  # noqa: E402

_VIEWS = "data/processed/ssl_views.npz"


def train(S, E, T, lam, seed, epochs, device, rows, batch=1024, lr=1e-3, drop=0.1, noise=0.1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc_s, enc_e = Encoder(S.shape[1]).to(device), Encoder(E.shape[1]).to(device)
    head_s, head_e = nn.Linear(128, T.shape[1]).to(device), nn.Linear(128, T.shape[1]).to(device)
    log_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    params = [*enc_s.parameters(), *enc_e.parameters(), log_scale]
    if lam > 0:
        params += [*head_s.parameters(), *head_e.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    St, Et, Tt = (torch.from_numpy(x).to(device) for x in (S, E, T))
    for _ in range(epochs):
        enc_s.train(); enc_e.train()
        perm = rng.permutation(rows)
        for i in range(0, len(perm) - batch + 1, batch):
            idx = torch.from_numpy(perm[i:i + batch]).to(device)
            a = enc_s(augment(St[idx], drop, noise))
            b = enc_e(augment(Et[idx], drop, noise))
            loss = clip_loss(a, b, log_scale)
            if lam > 0:
                loss = loss + lam * (F.mse_loss(head_s(a), Tt[idx]) + F.mse_loss(head_e(b), Tt[idx])) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                log_scale.clamp_(0, np.log(100))
    return enc_s, enc_e


def tissue_r2(X: np.ndarray, T: np.ndarray, n=6000, seed=0) -> tuple[float, float]:
    """Held-out R2 of a ridge probe, mean over tissue targets and for tau (last column)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), X[idx], T[idx], cv=5)
    r2 = 1 - ((T[idx] - pred) ** 2).sum(0) / ((T[idx] - T[idx].mean(0)) ** 2).sum(0)
    return float(r2.mean()), float(r2[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--report", default="contrastive/tissue_aware_eval.json")
    ap.add_argument("--save", default="data/processed/ssl_tissue_aware_seed0.npz")
    ap.add_argument("--no-holdout", action="store_true",
                    help="train on the screened lncRNAs too (in-sample; memorisation risk)")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    z = np.load(_VIEWS)
    keep = z["has_seq"] & z["has_cellline"] & z["has_atlas"]
    genes = z["genes"][keep]
    S = standardise(z["seq"][keep])
    E = standardise(np.hstack([z["atlas"][keep], z["cellline"][keep]]))
    T = standardise(tissue_view(genes))
    s = z["screened"][keep]
    lines = list(z["lines"])
    y = {c: z[f"label_{c}"][keep][s].astype(int) for c in lines}
    rng = np.random.default_rng(0)
    res: dict[str, dict[str, list[float]]] = {}

    rows = np.arange(len(S)) if args.no_holdout else np.flatnonzero(~s)
    print(f"training on {len(rows):,} lncRNAs; measuring on the {s.sum():,} screened "
          f"({'in-sample' if args.no_holdout else 'held out of training'})")

    def record(name, X):
        r = res.setdefault(name, {})
        rt, rtau = tissue_r2(X[s], T[s])
        r.setdefault("tissue_R2", []).append(rt)
        r.setdefault("tau_R2", []).append(rtau)
        for c in lines:
            r.setdefault(c, []).append(float(separation_z(X[s], y[c], rng)))

    record("raw: k-mers", S)
    for seed in range(args.seeds):
        for lam in args.lambdas:
            enc_s, enc_e = train(S, E, T, lam, seed, args.epochs, device, rows)
            zs, ze = embed(enc_s, S, device), embed(enc_e, E, device)
            tag = f"lambda={lam:g}"
            record(f"{tag}: sequence", zs)
            record(f"{tag}: expression", ze)
            record(f"{tag}: both", np.hstack([zs, ze]))
            if seed == 0 and lam == max(args.lambdas):
                np.savez_compressed(args.save, genes=genes, seq_emb=zs, expr_emb=ze)
        print(f"  seed {seed} done", flush=True)

    cols = ["tissue_R2", "tau_R2"] + lines
    print(f"\nmean (sd) over {args.seeds} seeds")
    print(f"{'embedding':<24}" + "".join(f"{c:>14}" for c in cols))
    for name, r in res.items():
        print(f"{name:<24}" + "".join(
            f"{np.mean(r[c]):>8.2f} ({np.std(r[c]):.2f})" if c.endswith("R2")
            else f"{np.mean(r[c]):>9.1f} ({np.std(r[c]):.1f})" for c in cols))
    Path(args.report).write_text(json.dumps(
        {"args": vars(args), "results": {n: {c: [round(v, 3) for v in r[c]] for c in cols}
                                         for n, r in res.items()}}, indent=1))
    print(f"-> {args.report}, {args.save}")


if __name__ == "__main__":
    main()
