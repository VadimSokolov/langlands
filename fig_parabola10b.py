import numpy as np, math, cypari2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
C = {"ink": "#1F3B63", "rust": "#9C3D1F", "gold": "#C99A2E", "sky": "#8FB3DE"}
pari = cypari2.Pari()
pari("L=lfunmf(mfinit([1,12]),mfDelta());")
zD = [float(z) for z in pari("lfunzeros(L,30)") if float(z) > 1e-6]
print(zD[:6])
fig, ax = plt.subplots(figsize=(4.4, 2.6))
X0 = -720
y = np.linspace(-32, 32, 600)
for sg, col, lab in [(0.5, C["sky"], r"$P_{1/2}$ (unconditional)"), (0.25, C["gold"], r"$P_{1/4}$ (not known)")]:
    ax.fill_betweenx(y, X0, sg**2 - y**2 / (4 * sg**2), color=col, alpha=0.45, lw=0, label=lab)
g2 = [-g * g for g in zD if g * g < 700]
ax.plot(g2, [0] * len(g2), "o", color=C["ink"], ms=3.2, label=r"$s_k=-\gamma_k^2$ for $\Delta$")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlim(X0, 40); ax.set_ylim(-32, 32)
ax.set_xlabel(r"$\mathrm{Re}\,s$"); ax.set_ylabel(r"$\mathrm{Im}\,s$"); ax.legend(frameon=False, fontsize=7, loc="upper left")
plt.savefig("/home/claude/langlands/book/figs/thorin_parabola10b.pdf"); plt.close()
