# Figure ch03_chebotarev: Chebotarev frequencies and Chebyshev bias for x^3 - x - 1.
# Requires the output of ch03_chebotarev.py (path of cumulative class counts).
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
C = {"ink": "#1F3B63", "rust": "#9C3D1F", "teal": "#1B7F79", "gold": "#C99A2E"}
P = np.load("/tmp/claude-0/ch03_path.npy")
p = P[:, 0].astype(float); n = P[:, 1:].sum(1).astype(float)
names = [r"split, $\{1\}$", r"transposition", r"$3$ cycle"]
dens = [1/6, 1/2, 1/3]; roots = [4, 0, 1]
cols = [C["ink"], C["rust"], C["teal"]]
idx = np.unique(np.geomspace(20, len(p)-1, 1500).astype(int))
fig, ax = plt.subplots(1, 2, figsize=(4.4, 2.1))
for k in range(3):
    ax[0].semilogx(p[idx], P[idx, k+1]/n[idx], color=cols[k], lw=1.0, label=names[k])
    ax[0].axhline(dens[k], color="k", lw=0.5, ls=":")
    E = np.log(p)/np.sqrt(p)*(P[:, k+1]/dens[k] - n)
    ax[1].semilogx(p[idx], E[idx], color=cols[k], lw=0.8)
    ax[1].axhline(1 - roots[k], color=cols[k], lw=0.6, ls="--")
    print(names[k], "mean E over log scale (p>1e3):", np.mean(E[idx][p[idx] > 1e3]))
ax[0].set_xlabel(r"$x$"); ax[0].set_title("class frequencies", fontsize=9)
ax[0].set_ylim(0, 0.85); ax[0].legend(frameon=False, fontsize=6, loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8)
ax[1].set_xlabel(r"$x$"); ax[1].set_title(r"normalised error $E_C(x)$", fontsize=9)
fig.tight_layout()
fig.savefig("../work/figs/ch03_chebotarev.pdf")
