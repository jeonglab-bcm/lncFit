"""SSL step 5: fine-tune a cell-line-conditioned head on the frozen tissue-aware embedding.

Input per (lncRNA, cell line): the frozen 128-d tissue-aware expression embedding from step 4
(trained without labels, and without the screened lncRNAs) plus a one-hot of the cell line.
A small MLP head maps it to a 64-d embedding, trained with

  supervised contrastive loss  rows in the same class -- (cell line, essential or not) -- are
                               positives, so lncRNAs essential in the same line pull together
  + BCE                        a linear classifier on the head embedding, for a score

Genes are held out: 5 folds over the 5,496 screened lncRNAs; each fold's head trains on the
other 80% (all five lines' rows) and is applied to its 20%, which it never saw in any line.
A map of the genes a head trained on would look separated by construction.

Compared against the frozen embedding on the same held-out genes:
  AUPRC  head's BCE score vs logistic regression on the frozen embedding (same folds)
  z      20-nearest-neighbour essential score among a fold's held-out genes, vs 200 label
         shuffles, averaged over folds

Writes the fold-0 held-out genes' fine-tuned embeddings, per cell line, for the maps.

Usage:
  uv run python contrastive/finetune.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from train_clip import separation_z  # noqa: E402

_VIEWS = "data/processed/ssl_views.npz"
_FROZEN = "data/processed/ssl_tissue_aware_seed0.npz"


class Head(nn.Module):
    def __init__(self, d_in: int, n_lines: int, d_hidden=256, d_out=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in + n_lines, d_hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(d_hidden, d_out))
        self.clf = nn.Linear(d_out, 1)

    def forward(self, x):
        z = F.normalize(self.net(x), dim=-1)
        return z, self.clf(z).squeeze(-1)


def supcon(z, cls, temp=0.1):
    """Supervised contrastive loss (Khosla et al. 2020); anchors without a positive skipped."""
    sim = z @ z.T / temp
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (cls[:, None] == cls[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n = pos.sum(1)
    keep = n > 0
    return -((logp * pos).sum(1)[keep] / n[keep]).mean()


def rows(X, lines_idx, n_lines):
    """Stack (gene, line) inputs: embedding + line one-hot, for every gene x line."""
    oh = np.eye(n_lines, dtype=np.float32)
    return np.vstack([np.hstack([X, np.repeat(oh[[l]], len(X), 0)]) for l in lines_idx])


def fit_head(Xtr, line_tr, y_tr, n_lines, seed, device, epochs=40, batch=1024, lam_bce=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = Head(Xtr.shape[1] - n_lines, n_lines).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    cls = torch.from_numpy(line_tr * 2 + y_tr).to(device)
    for _ in range(epochs):
        head.train()
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), batch):
            idx = torch.from_numpy(perm[i:i + batch]).to(device)
            z, logit = head(Xt[idx])
            loss = supcon(z, cls[idx]) + lam_bce * F.binary_cross_entropy_with_logits(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--report", default="contrastive/finetune_eval.json")
    ap.add_argument("--save", default="data/processed/ssl_finetuned_fold0.npz")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    v = np.load(_VIEWS)
    lines = list(v["lines"])
    screened = v["genes"][v["screened"]]
    lab = {c: v[f"label_{c}"][v["screened"]].astype(int) for c in lines}
    F0 = np.load(_FROZEN)
    pos = {g: i for i, g in enumerate(F0["genes"])}
    X = F0["expr_emb"][[pos[g] for g in screened]].astype(np.float32)
    n, L = len(screened), len(lines)
    print(f"{n:,} screened lncRNAs x {L} lines; frozen tissue-aware embedding {X.shape[1]} dims")

    rng = np.random.default_rng(0)
    oof = {c: np.zeros(n) for c in lines}
    oof_lr = {c: np.zeros(n) for c in lines}
    z_ft = {c: [] for c in lines}
    z_fr = {c: [] for c in lines}
    for f, (tr, te) in enumerate(KFold(args.folds, shuffle=True, random_state=0).split(screened)):
        Xtr = rows(X[tr], range(L), L)
        line_tr = np.repeat(np.arange(L), len(tr))
        y_tr = np.concatenate([lab[c][tr] for c in lines])
        head = fit_head(Xtr, line_tr, y_tr, L, f, device, args.epochs)
        saved = {}
        for li, c in enumerate(lines):
            Xte = rows(X[te], [li], L)
            with torch.no_grad():
                zt, logit = head(torch.from_numpy(Xte).to(device))
            zt, prob = zt.cpu().numpy(), torch.sigmoid(logit).cpu().numpy()
            oof[c][te] = prob
            oof_lr[c][te] = LogisticRegression(max_iter=2000).fit(X[tr], lab[c][tr]) \
                .predict_proba(X[te])[:, 1]
            z_ft[c].append(separation_z(zt, lab[c][te], rng))
            z_fr[c].append(separation_z(X[te], lab[c][te], rng))
            saved[c] = zt
        if f == 0:
            np.savez_compressed(args.save, genes=screened[te], lines=np.array(lines),
                                **{f"emb_{c}": saved[c] for c in lines})
        print(f"  fold {f} done", flush=True)

    rep = {}
    print(f"\nHeld-out genes only. AUPRC over all {n:,} (out-of-fold); z mean over folds")
    print(f"{'':<34}" + "".join(f"{c:>12}" for c in lines))
    for name, vals in [
        ("AUPRC: positive rate", {c: lab[c].mean() for c in lines}),
        ("AUPRC: LR on frozen embedding", {c: average_precision_score(lab[c], oof_lr[c]) for c in lines}),
        ("AUPRC: fine-tuned head", {c: average_precision_score(lab[c], oof[c]) for c in lines}),
        ("z: frozen embedding", {c: float(np.mean(z_fr[c])) for c in lines}),
        ("z: fine-tuned embedding", {c: float(np.mean(z_ft[c])) for c in lines}),
    ]:
        rep[name] = {c: round(float(vals[c]), 4) for c in lines}
        fmt = "{:>12.4f}" if name.startswith("AUPRC") else "{:>12.1f}"
        print(f"{name:<34}" + "".join(fmt.format(vals[c]) for c in lines))
    Path(args.report).write_text(json.dumps({"args": vars(args), "results": rep}, indent=1))
    print(f"-> {args.report}, {args.save}")


if __name__ == "__main__":
    main()
