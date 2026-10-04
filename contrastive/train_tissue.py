"""SSL step 3: contrastive learning against tissue specificity, with seeds and controls.

Two pairings, each trained over several seeds, each next to an untrained control:

  A  sequence vs expression  -- step 2's setup: k-mers vs developmental atlas (297) +
                                cell-line TPM (5), hard InfoNCE (only the same lncRNA is
                                a positive)
  B  sequence vs tissue      -- k-mers vs a tissue-specificity profile: log2 mean RPKM per
                                organ (7, pseudocount 0.05) + tau. SOFT positives: the target
                                for each sequence is a distribution over the batch weighted
                                by tissue-profile similarity, so lncRNAs with similar organ
                                patterns count as partial matches. EDA 2 found tissue
                                patterns form a continuum, so hard cluster labels would invent
                                structure that is not there.

  untrained  the same encoders at random initialisation, never trained: what the
             architecture gives for free. A contrastive gain has to beat this.

No essentiality label is read during training. Evaluation is the EDA-1 neighbour test on the
5,496 screened lncRNAs, per cell line, reported as mean (sd) over seeds.

Usage:
  uv run python contrastive/train_tissue.py
  uv run python contrastive/train_tissue.py --seeds 3 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_clip import Encoder, augment, clip_loss, embed, separation_z, standardise  # noqa: E402

_VIEWS = "data/processed/ssl_views.npz"
_ATLAS = "data/external/sarropoulos_all_lncrna_rpkm.tsv.gz"
_ORGANS = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Ovary", "Testis"]


def tissue_view(genes: np.ndarray) -> np.ndarray:
    atlas = pd.read_csv(_ATLAS, sep="\t").set_index("lncRNA")
    atlas.index = atlas.index.astype(str)
    atlas = atlas.reindex(genes).fillna(0.0)
    organ_of = np.array([c.split(".")[0] for c in atlas.columns])
    prof = np.column_stack([atlas.loc[:, organ_of == o].mean(axis=1) for o in _ORGANS])
    m = prof.max(1, keepdims=True)
    tau = (1 - prof / np.where(m > 0, m, 1)).sum(1) / (len(_ORGANS) - 1)
    return np.column_stack([np.log2(prof + 0.05), tau]).astype(np.float32)


def soft_loss(a, b, log_scale, target):
    """InfoNCE with soft targets: rows of `target` sum to 1 over the batch."""
    logits = a @ b.T * log_scale.exp()
    return (-(target * F.log_softmax(logits, 1)).sum(1).mean()
            - (target.T * F.log_softmax(logits.T, 1)).sum(1).mean()) / 2


def train(S, T, soft: bool, seed: int, epochs: int, device: str, batch=1024, lr=1e-3,
          drop=0.1, noise=0.1, target_temp=0.05):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc_s, enc_t = Encoder(S.shape[1]).to(device), Encoder(T.shape[1]).to(device)
    if epochs == 0:  # untrained control
        return enc_s, enc_t
    log_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([*enc_s.parameters(), *enc_t.parameters(), log_scale],
                            lr=lr, weight_decay=1e-4)
    St, Tt = torch.from_numpy(S).to(device), torch.from_numpy(T).to(device)
    Tn = F.normalize(Tt, dim=-1)  # for soft targets: cosine similarity of tissue profiles
    for _ in range(epochs):
        enc_s.train(); enc_t.train()
        perm = rng.permutation(len(S))
        for i in range(0, len(perm) - batch + 1, batch):
            idx = torch.from_numpy(perm[i:i + batch]).to(device)
            a = enc_s(augment(St[idx], drop, noise))
            b = enc_t(augment(Tt[idx], drop, noise))
            if soft:
                target = F.softmax(Tn[idx] @ Tn[idx].T / target_temp, dim=1)
                loss = soft_loss(a, b, log_scale, target)
            else:
                loss = clip_loss(a, b, log_scale)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                log_scale.clamp_(0, np.log(100))
    return enc_s, enc_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30,
                    help="step 2's held-out retrieval peaked around epoch 30")
    ap.add_argument("--report", default="contrastive/tissue_eval.json")
    ap.add_argument("--save", default="data/processed/ssl_tissue_seed0.npz")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    z = np.load(_VIEWS)
    keep = z["has_seq"] & z["has_cellline"] & z["has_atlas"]
    genes = z["genes"][keep]
    S = standardise(z["seq"][keep])
    views = {
        "A": standardise(np.hstack([z["atlas"][keep], z["cellline"][keep]])),
        "B": standardise(tissue_view(genes)),
    }
    s = z["screened"][keep]
    lines = list(z["lines"])
    y = {c: z[f"label_{c}"][keep][s].astype(int) for c in lines}
    print(f"{len(genes):,} lncRNAs; tissue view {views['B'].shape[1]} dims; device {device}")

    rng = np.random.default_rng(0)
    results: dict[str, dict[str, list[float]]] = {}

    def score(name: str, X: np.ndarray):
        for c in lines:
            results.setdefault(name, {}).setdefault(c, []).append(
                float(separation_z(X, y[c], rng)))

    score("raw: tissue profile", views["B"][s])
    for seed in range(args.seeds):
        for model, soft in [("A", False), ("B", True)]:
            for trained in (True, False):
                enc_s, enc_t = train(S, views[model], soft, seed,
                                     args.epochs if trained else 0, device)
                zs, zt = embed(enc_s, S, device), embed(enc_t, views[model], device)
                tag = f"{model}{'' if trained else ' untrained'}"
                score(f"{tag}: sequence", zs[s])
                score(f"{tag}: other view", zt[s])
                score(f"{tag}: both", np.hstack([zs[s], zt[s]]))
                if trained and model == "B" and seed == 0:
                    np.savez_compressed(args.save, genes=genes, seq_emb=zs, tissue_emb=zt)
        print(f"  seed {seed} done", flush=True)

    print(f"\nNeighbour separation z, mean (sd) over {args.seeds} seeds; "
          f"A = sequence vs expression, B = sequence vs tissue (soft)")
    print(f"{'embedding':<28}" + "".join(f"{c:>15}" for c in lines))
    for name, per in results.items():
        print(f"{name:<28}" + "".join(
            f"{np.mean(per[c]):>9.1f} ({np.std(per[c]):>3.1f})" for c in lines))
    Path(args.report).write_text(json.dumps(
        {"args": vars(args), "z": {n: {c: [round(v, 2) for v in per[c]] for c in lines}
                                   for n, per in results.items()}}, indent=1))
    print(f"-> {args.report}, {args.save}")


if __name__ == "__main__":
    main()
