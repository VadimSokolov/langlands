# Chapter 12: exact trace densities for USp(4), SU(2)xSU(2), SU(2) diagonal (Weyl integration by quadrature),
# posterior probability paths for the genus 2 curves C1, C2, C3 (uniform prior), and the genus 2 histogram figure.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "dejavuserif"})
COL = ["#1F3B63", "#1B7F79", "#9C3D1F", "#C99A2E"]
DATA = "../work/"; OUT = "../work/figs/"
s = lambda x: np.sqrt(np.clip(4 - x * x, 0, None)) / (2 * np.pi)          # semicircle density
u, w = leggauss(200); th = (u + 1) * np.pi / 2; w = w * np.pi / 2
def conv(t, kern):                                                      # integral over the overlap of [-2,2] and t-[-2,2]
    a, b = max(-2, t - 2), min(2, t + 2)
    if b <= a: return 0.0
    x = (a + b) / 2 + (b - a) / 2 * np.cos(th)
    return np.sum(w * (b - a) / 2 * np.sin(th) * kern(x, t))
fU = np.vectorize(lambda t: conv(t, lambda x, t: 0.5 * (2 * x - t) ** 2 * s(x) * s(t - x)))   # USp(4)
fS = np.vectorize(lambda t: conv(t, lambda x, t: s(x) * s(t - x)))                          # SU(2)xSU(2)
fD = lambda t: np.sqrt(np.clip(16 - t * t, 0, None)) / (8 * np.pi)                          # SU(2) diagonal
g = np.linspace(-4, 4, 8001); dens = np.vstack([fU(g), fS(g), fD(g)])
names = ["USp(4)", r"SU(2)$\times$SU(2)", "SU(2) diagonal"]
def loglik(x): return np.array([np.clip(np.log(np.maximum(np.interp(x, g, d), 1e-300)), -70, None) for d in dens])
curves = [("C1", r"$C_1$"), ("C2", r"$C_2$"), ("C3", r"$C_3$")]
# posterior paths
fig, ax = plt.subplots(3, 1, figsize=(4.4, 2.7), sharex=True)
for a, (f, lab) in zip(ax, curves):
    x = np.load(DATA + f"g2_{f}.npy"); cum = np.cumsum(loglik(x), axis=1)
    post = np.exp(cum - cum.max(0)); post /= post.sum(0); n = np.arange(1, len(x) + 1)
    for k, (c, ls) in enumerate(zip(COL[:3], ["-", "--", ":"])): a.plot(n, post[k], color=c, ls=ls, lw=1.3, label=names[k])
    a.set_xscale("log"); a.set_ylim(-0.03, 1.03); a.set_ylabel("posterior"); a.text(1.2, 0.8, lab)
    for sp in ["top", "right"]: a.spines[sp].set_visible(False)
    print(f, [post[:, m - 1] for m in (10, 50, 100, 200, len(x))], cum[:, -1] - cum[0, -1])
ax[0].legend(frameon=False, fontsize=7, loc="center right"); ax[-1].set_xlabel("number of good primes")
fig.tight_layout(); fig.savefig(OUT + "ch12_posterior.pdf")
# histograms against the three candidate densities (C3: atom at 0 shown as a bar of its mass)
fig, ax = plt.subplots(1, 3, figsize=(4.4, 1.9), sharey=True)
for a, (f, lab) in zip(ax, curves):
    x = np.load(DATA + f"g2_{f}.npy"); nz = np.abs(x) > 1e-12
    a.hist(x[nz], bins=60, range=(-4, 4), weights=np.full(nz.sum(), 1 / (len(x) * 8 / 60)), color="#D9D9D9", edgecolor="none")
    if (~nz).mean() > 0.1: a.annotate(f"atom {(~nz).mean():.2f}", (0, 0.47), ha="center", fontsize=6.5, color=COL[3])
    for k in range(3): a.plot(g, dens[k], color=COL[k], lw=1, label=names[k])
    a.set_title(lab, fontsize=8); a.set_xlabel(r"$a_p/\sqrt{p}$", fontsize=7); a.set_xlim(-4, 4); a.set_ylim(0, 0.55)
    for sp in ["top", "right"]: a.spines[sp].set_visible(False)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=6.5, ncol=3, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT + "genus2.pdf")
# Kullback Leibler divergences (nats) and per prime rates against USp(4)
for i in range(3):
    print(names[i], [float(np.trapezoid(np.where(dens[i] > 1e-14, dens[i] * np.log(np.maximum(dens[i], 1e-300) / np.maximum(dens[j], 1e-300)), 0), g)) for j in range(3)])
for f, _ in curves:
    L = loglik(np.load(DATA + f"g2_{f}.npy")); print(f, "rates", [float(np.mean(np.clip(L[k] - L[0], -20, 20))) for k in range(3)])
