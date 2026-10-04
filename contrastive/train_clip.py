"""SSL step 2: first contrastive run -- sequence view against expression view.

CLIP-style: two MLP encoders, one per view, trained so a lncRNA's sequence embedding is
closest to its own expression embedding among everything in the batch (symmetric InfoNCE,
learnable temperature). No essentiality label is read during training.

  sequence view    3/4/5-mer frequencies (1,344 dims)
  expression view  developmental atlas (297) + cell-line mRNA TPM (5), log1p

Trained on every lncRNA with all views (31,661; 26,165 never screened), 5% held out to
watch cross-view retrieval. Augmentation: input dropout and Gaussian noise on the
standardised features.

Evaluation (labels used here only): the EDA-1 neighbour test on the 5,496 screened
lncRNAs, per cell line -- z of the 20-nearest-neighbour essential score against 200 label
shuffles -- for the learned embeddings next to the raw views and the GC + length control.
Also: how much of GC content a linear model can read back from each embedding.

Usage:
  uv run python contrastive/train_clip.py
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent / "eda"))
from embedding_separation import neighbours  # noqa: E402

_VIEWS = "data/processed/ssl_views.npz"


def standardise(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


class Encoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 512, d_out: int = 128, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def augment(x: torch.Tensor, drop: float, noise: float) -> torch.Tensor:
    return x * (torch.rand_like(x) > drop) + noise * torch.randn_like(x)


def clip_loss(a, b, log_scale):
    logits = a @ b.T * log_scale.exp()
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


@torch.no_grad()
def embed(enc, X, device, bs=4096):
    enc.eval()
    return np.vstack([enc(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy()
                      for i in range(0, len(X), bs)])


def retrieval(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Seq -> expression retrieval among the given rows: top-1 and top-1% accuracy."""
    sims = a @ b.T
    rank = (sims > np.diag(sims)[:, None]).sum(1)
    return float((rank == 0).mean()), float((rank < max(1, len(a) // 100)).mean())


def separation_z(X: np.ndarray, y: np.ndarray, rng, k=20, shuffles=200) -> float:
    nbr = neighbours(X, k)
    auprc = average_precision_score(y, y[nbr].mean(1))
    null = []
    for _ in range(shuffles):
        p = rng.permutation(y)
        null.append(average_precision_score(p, p[nbr].mean(1)))
    return (auprc - np.mean(null)) / np.std(null)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/ssl_clip_v1.npz")
    ap.add_argument("--report", default="contrastive/clip_v1_eval.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    z = np.load(_VIEWS)
    keep = z["has_seq"] & z["has_cellline"] & z["has_atlas"]
    genes = z["genes"][keep]
    S = standardise(z["seq"][keep])
    E = standardise(np.hstack([z["atlas"][keep], z["cellline"][keep]]))
    screened = z["screened"][keep]
    lines = list(z["lines"])
    print(f"{len(genes):,} lncRNAs ({(~screened).sum():,} unscreened); seq {S.shape[1]}, "
          f"expression {E.shape[1]} dims; device {device}")

    val = rng.random(len(genes)) < 0.05
    tr = np.flatnonzero(~val)
    enc_s, enc_e = Encoder(S.shape[1]).to(device), Encoder(E.shape[1]).to(device)
    log_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([*enc_s.parameters(), *enc_e.parameters(), log_scale],
                            lr=args.lr, weight_decay=1e-4)
    St, Et = torch.from_numpy(S).to(device), torch.from_numpy(E).to(device)

    for epoch in range(1, args.epochs + 1):
        enc_s.train(); enc_e.train()
        perm, total = rng.permutation(tr), 0.0
        for i in range(0, len(perm) - args.batch + 1, args.batch):  # full batches only
            idx = torch.from_numpy(perm[i:i + args.batch]).to(device)
            a = enc_s(augment(St[idx], args.drop, args.noise))
            b = enc_e(augment(Et[idx], args.drop, args.noise))
            loss = clip_loss(a, b, log_scale)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                log_scale.clamp_(0, np.log(100))
            total += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            r1, r1pct = retrieval(embed(enc_s, S[val], device), embed(enc_e, E[val], device))
            print(f"  epoch {epoch:>3}  loss {total / (len(perm) // args.batch):.3f}  "
                  f"val retrieval top-1 {r1:.3f} top-1% {r1pct:.3f} "
                  f"(chance {1 / val.sum():.4f} / 0.01)  temp {1 / log_scale.exp().item():.3f}",
                  flush=True)

    zs, ze = embed(enc_s, S, device), embed(enc_e, E, device)
    np.savez_compressed(args.out, genes=genes, seq_emb=zs, expr_emb=ze)
    print(f"-> {args.out}")

    # ---- evaluation on the screened lncRNAs; labels are read only from here on ----
    s = screened
    y = {c: z[f"label_{c}"][keep][s].astype(int) for c in lines}
    gc, ln = z["gc"][keep][s], z["log10_length"][keep][s]
    cands = {
        "learned: sequence": zs[s],
        "learned: expression": ze[s],
        "learned: both": np.hstack([zs[s], ze[s]]),
        "raw: k-mer 3-5": S[s],
        "raw: atlas + cell-line expr": E[s],
        "control: GC + length": np.column_stack([gc, ln]),
    }
    report = {"args": vars(args), "z": {}, "gc_r2": {}}
    print(f"\nNeighbour separation z (vs 200 shuffles), {s.sum():,} screened lncRNAs")
    print(f"{'embedding':<30}" + "".join(f"{c:>12}" for c in lines) + f"{'GC R2':>8}")
    for name, X in cands.items():
        zrow = {c: round(float(separation_z(X, y[c], rng)), 1) for c in lines}
        r2 = (float(np.mean(cross_val_score(RidgeCV(alphas=np.logspace(-2, 3, 12)), X, gc,
                                            cv=5, scoring="r2")))
              if not name.startswith("control") else 1.0)
        report["z"][name], report["gc_r2"][name] = zrow, round(r2, 3)
        print(f"{name:<30}" + "".join(f"{zrow[c]:>12.1f}" for c in lines) + f"{r2:>8.2f}",
              flush=True)
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(f"-> {args.report}")


if __name__ == "__main__":
    main()
