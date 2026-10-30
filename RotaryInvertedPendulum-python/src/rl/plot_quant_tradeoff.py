"""Quantization tradeoff plots.  Usage: python plot_quant_tradeoff.py data.csv [last|best_model]
Swap METRIC (e.g. to 'conv_episodes' once you add it; set LOWER_IS_BETTER=True) to reuse everything."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

path = sys.argv[1]; WHICH = sys.argv[2] if len(sys.argv) > 2 else "last"
METRIC, LOWER_IS_BETTER = "mean_return", False
d = pd.read_csv(path); d = d[d.which == WHICH]
keys = ["obs_norm", "obs_bits", "weight_bits", "act_bits"]
g = d.groupby(keys)[METRIC].agg(["mean", "std", "count"]).reset_index()
g["bits"] = g.obs_bits + g.weight_bits + g.act_bits          # crude bit-budget proxy
norms, obits = ["orig", "raw"], sorted(g.obs_bits.unique(), reverse=True)
bits = [16, 8, 4]
cmap = "RdYlGn_r" if LOWER_IS_BETTER else "RdYlGn"
lo, hi = g["mean"].min(), g["mean"].max()

# ---- Fig 1: heatmap grid (weight bits x act bits), per obs bits / obs normalisation ----
fig, axs = plt.subplots(len(norms), len(obits), figsize=(4*len(obits)+1, 4*len(norms)), squeeze=False)
for i, n in enumerate(norms):
    for j, ob in enumerate(obits):
        ax = axs[i, j]; s = g[(g.obs_norm == n) & (g.obs_bits == ob)]
        M = np.full((3, 3), np.nan); S = M.copy()
        for _, r in s.iterrows():
            M[bits.index(r.weight_bits), bits.index(r.act_bits)] = r["mean"]
            S[bits.index(r.weight_bits), bits.index(r.act_bits)] = r["std"]
        im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi)
        for a in range(3):
            for b in range(3):
                if not np.isnan(M[a, b]):
                    ax.text(b, a, f"{M[a,b]:.0f}\n±{S[a,b]:.0f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), [f"a{b}" for b in bits]); ax.set_yticks(range(3), [f"w{b}" for b in bits])
        ax.set_title(f"obs_norm={n}, obs bits={ob}")
fig.colorbar(im, ax=axs, shrink=.8, label=f"{METRIC} ({WHICH}, mean over seeds)")
fig.suptitle("Return vs. weight/activation bits (cell: mean ± seed std)")
fig.savefig("tradeoff_heatmaps.png", dpi=150, bbox_inches="tight")

# ---- Fig 2: return vs bit budget, with Pareto frontier ----
fig, ax = plt.subplots(figsize=(9, 5.5))
for n, m in zip(norms, ["o", "s"]):
    s = g[g.obs_norm == n]
    sc = ax.scatter(s.bits + (0.15 if n == "raw" else -0.15), s["mean"], c=s.obs_bits.map({16: 2, 8: 1, 4: 0}),
                    cmap="viridis", marker=m, s=50, edgecolor="k", lw=.4, label=f"obs_norm={n}")
    ax.errorbar(s.bits + (0.15 if n == "raw" else -0.15), s["mean"], yerr=s["std"], fmt="none", ecolor="gray", alpha=.3)
best = g.sort_values("bits"); sign = -1 if LOWER_IS_BETTER else 1
front = best[sign*best["mean"] >= sign*best["mean"].cummax() if not LOWER_IS_BETTER else best["mean"] <= best["mean"].cummin()]
ax.step(front.bits, front["mean"], where="post", color="crimson", lw=2, label="Pareto frontier")
ax.set_xlabel("obs + weight + act bits (proxy for cost)"); ax.set_ylabel(f"{METRIC} ({WHICH})")
cb = fig.colorbar(sc, ticks=[0, 1, 2]); cb.ax.set_yticklabels(["obs 4b", "obs 8b", "obs 16b"])
ax.legend(loc="lower right"); ax.grid(alpha=.3); ax.set_title("Quantization tradeoff: return vs. bit budget (error bars = seed std)")
fig.savefig("tradeoff_pareto.png", dpi=150, bbox_inches="tight")
print(g.sort_values("mean", ascending=LOWER_IS_BETTER).head(8).to_string())
