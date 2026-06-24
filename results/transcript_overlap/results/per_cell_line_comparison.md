# Per-cell-line vs Pooled Comparison (issue #49)

Config: k=3, objective=reg:squarederror, day=both
Pooled baseline: final_eval_20260623_231609

| cell_line | n_test | pooled_rho | per_cell_line_rho | delta_rho | pooled_day7 | per_day7 | pooled_day14 | per_day14 | per_cell_line_run |
|---|---|---|---|---|---|---|---|---|---|
| HAP1 | 13888 | 0.0648 | 0.0699 | 0.0051 | 0.0151 | 0.0249 | 0.0978 | 0.1046 | final_eval_20260623_141441 |
| HEK293FT | 13728 | 0.1557 | 0.1685 | 0.0128 | 0.2114 | 0.2284 | 0.0762 | 0.0838 | final_eval_20260623_164938 |
| K562 | 13884 | 0.309 | 0.3189 | 0.0099 | 0.0042 | 0.0063 | 0.3074 | 0.319 | final_eval_20260623_182222 |
| MDA-MB-231 | 13896 | 0.1178 | 0.1403 | 0.0225 | 0.0883 | 0.11 | 0.143 | 0.169 | final_eval_20260623_150517 |
| THP1 | 13896 | 0.042 | 0.0379 | -0.0041 | 0.0391 | 0.0256 | 0.0448 | 0.049 | final_eval_20260623_141449 |

## Interpretation

- Positive `delta_rho` => per-cell-line model beats pooled (pooling was diluting).
- If K562 per-cell-line > pooled K562 => adopt per-cell-line models.
- If HAP1/THP1/MDA stay ~0 alone => low-reliability labels, not a modeling problem.

### How to read the K562 result (training-set confound)

- If per-cell-line K562 beats pooled K562 **despite 5x less data** -> specialization dominates, strong signal to adopt per-cell-line.
- If per-cell-line K562 is worse -> inconclusive (could be data volume, not pooling). A size-matched pooled control (210K random records across all 5 cell lines) would disentangle this.
