"""Does adding the Celligner cell-line embedding (issue #78) help the k=5 kmer model,
and does the embedding's dimensionality matter?

Same k=5 kmer features, same stratified 90/10 split (seed=42), same chr1 held-out
test, and the same best-known xgboost hyperparameters from the feature x model
comparison (results/lncrna_rra_day14/README.md) -- the only thing that changes is
celligner_embedding_dim: 0 (off), 2 (2-D UMAP), 10 or 70 (pre-UMAP PCA columns).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.features import build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

K = 5
SEED = 42
EMBEDDING_DIMS = [0, 2, 10, 70]
XGB_PARAMS = {
    "learning_rate": 0.02, "max_depth": 9, "subsample": 0.5,
    "colsample_bytree": 0.9, "min_child_weight": 3, "reg_alpha": 3.188749808609341,
    "reg_lambda": 3.078336708769974e-06, "scale_pos_weight": 1.0,
    "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist", "random_state": SEED,
}


def main():
    train_records = load_jsonl("data/processed/train_lncrna_day14_chrom1.jsonl.gz", record_cls=LncRnaRecord)
    test_records = load_jsonl("data/processed/test_lncrna_day14_chrom1.jsonl.gz", record_cls=LncRnaRecord)
    with open("data/processed/body_sequences_transcript.json") as fh:
        raw = json.load(fh)
    transcript_sequences = {gid: seq for gid, (seq, _) in raw.items()}

    idx_train, idx_es = train_test_split(
        range(len(train_records)), test_size=0.1,
        stratify=[r.label for r in train_records], random_state=SEED,
    )
    final_train = [train_records[i] for i in idx_train]
    final_es = [train_records[i] for i in idx_es]

    train_targets = {r.target for r in final_train}
    vocab = fit_vocab([transcript_sequences[t] for t in train_targets if t in transcript_sequences], K)

    rows = []
    for dim in EMBEDDING_DIMS:
        X_tr, y_tr, _ = build_lncrna_features(
            final_train, transcript_sequences, k=K, vocab=vocab, celligner_embedding_dim=dim,
        )
        X_es, y_es, _ = build_lncrna_features(
            final_es, transcript_sequences, k=K, vocab=vocab, celligner_embedding_dim=dim,
        )
        X_test, y_test, _ = build_lncrna_features(
            test_records, transcript_sequences, k=K, vocab=vocab, celligner_embedding_dim=dim,
        )

        model = xgb.XGBClassifier(
            **XGB_PARAMS, n_estimators=2000,
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
        )
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        y_pred = model.predict_proba(X_test)[:, 1]
        metrics_rows = evaluate_lncrna_by_group(test_records, y_test, y_pred)
        overall = next(r for r in metrics_rows if r["split"] == "Overall")
        row = {"celligner_embedding_dim": dim, "n_features": X_tr.shape[1],
               "auroc": overall["auroc"], "auprc": overall["auprc"]}
        rows.append(row)
        print(f"celligner_embedding_dim={dim}  n_features={X_tr.shape[1]}  "
              f"AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}")
        for r in metrics_rows:
            if r["split"] != "Overall":
                print(f"    {r['split']:<12} AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}")

    out_dir = Path("results/lncrna_rra_day14/celligner_embedding_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "run_info.json", "w") as fh:
        json.dump({"k": K, "seed": SEED, "embedding_dims": EMBEDDING_DIMS,
                   "xgb_params": XGB_PARAMS, "git_commit": git_commit()}, fh, indent=2)
    print(f"\nSaved -> {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
