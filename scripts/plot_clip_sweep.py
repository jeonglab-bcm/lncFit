"""Plot log2FC clipping quantile sweep results (issue #51)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

SWEEP_CSV = Path("results/clip_sweep/clip_quantile_sweep.csv")
OUT_PATH   = Path("results/clip_sweep/clip_quantile_sweep.png")

df = pl.read_csv(SWEEP_CSV).sort("quantile")

quantiles   = df["quantile"].to_list()
rho         = df["cv_rho_mean"].to_list()
rho_std     = df["cv_rho_std"].to_list()
pct_clipped = df["pct_clipped"].to_list()
clip_limit  = df["clip_limit"].to_list()

labels = [f"{q:.3f}" for q in quantiles]
x = list(range(len(quantiles)))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("log₂FC Clipping Quantile Sweep (k=3, MSE, transcript body)", fontsize=11)

# ── Left: CV ρ vs quantile ───────────────────────────────────────────────────
ax = axes[0]
ax.errorbar(x, rho, yerr=rho_std, fmt="o-", color="#1a3a5c",
            capsize=4, linewidth=1.8, markersize=6)
ax.axhline(rho[0], color="#aaa", linestyle="--", linewidth=1, label="no clipping")
for xi, (q, r) in enumerate(zip(quantiles, rho)):
    ax.annotate(f"{r:.4f}", (xi, r), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="#1a3a5c")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel("Clipping quantile", fontsize=10)
ax.set_ylabel("CV Spearman ρ (mean ± std)", fontsize=10)
ax.set_title("CV performance vs clipping quantile", fontsize=10)
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
ax.grid(axis="y", alpha=0.3)

# ── Right: clip_limit and pct_clipped vs quantile ────────────────────────────
ax2 = axes[1]
color_limit = "#1a3a5c"
color_pct   = "#c0392b"

l1, = ax2.plot(x, clip_limit, "o-", color=color_limit, linewidth=1.8, markersize=6)
ax2.set_ylabel("Clip limit (log₂FC units)", fontsize=10, color=color_limit)
ax2.tick_params(axis="y", labelcolor=color_limit)

ax3 = ax2.twinx()
l2, = ax3.plot(x, pct_clipped, "s--", color=color_pct, linewidth=1.5, markersize=6)
ax3.set_ylabel("% of training values clipped", fontsize=10, color=color_pct)
ax3.tick_params(axis="y", labelcolor=color_pct)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_xlabel("Clipping quantile", fontsize=10)
ax2.set_title("Clip limit and % clipped vs quantile", fontsize=10)
ax2.legend([l1, l2], ["clip limit", "% clipped"], fontsize=8, loc="upper right")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_PATH}")
