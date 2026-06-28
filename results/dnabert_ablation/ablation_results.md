# DNABERT-2 Feature Ablation (Issue #54)

Hyperparameters fixed from `data/model/xgboost_best_params_k3_mse.json` (k=3, no re-tuning per condition).

| Condition | CV ρ (mean) | CV ρ (std) | Δ vs baseline |
|---|---|---|---|
| k-mer only (baseline) | 0.1325 | 0.0141 | 0.0000 |
| k-mer + DNABERT-2 body (first) | 0.1661 | 0.0082 | +0.0336 |
| k-mer + DNABERT-2 body (last) | 0.1753 | 0.0091 | +0.0428 |
| k-mer + DNABERT-2 body (mean) | 0.1674 | 0.0098 | +0.0349 |
| k-mer + DNABERT-2 guide | 0.1214 | 0.0126 | -0.0111 |
| k-mer + body(mean) + guide | 0.1644 | 0.0085 | +0.0319 |
