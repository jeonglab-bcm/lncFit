# log2FC Clipping Quantile Sweep (issue #51)

Config: k=3, objective=reg:squarederror, body_sequences=data/processed/body_sequences_transcript.json

Fixed hyperparameters from: results/transcript_overlap/data/model/xgboost_best_params_k3_mse.json


| quantile | clip_limit | pct_clipped | cv_rho_mean | cv_rho_std |
|----------|-----------|-------------|-------------|------------|
| 1.000 | 10.2530 | 0.000% | 0.1728 | 0.0081 |
| 0.990 | 2.2250 | 1.000% | 0.1759 | 0.0079 |
| 0.975 | 1.2890 | 2.497% | 0.1776 | 0.0076 |
| 0.950 | 1.0000 | 4.671% | 0.1781 | 0.0074 |
| 0.900 | 1.0000 | 4.671% | 0.1781 | 0.0074 |
