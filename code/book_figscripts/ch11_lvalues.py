# Chapter 11: L(s,f) = E[Q(s, 2 pi M_f)] along the real axis for four multiplicative eta products.
import numpy as np, cypari2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pari = cypari2.Pari(); pari.allocatemem(10**9)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "dejavuserif"})
forms = [("[1,24]", r"$\Delta=\eta(\tau)^{24}$, $k=12$", "#1F3B63", "-"),
         ("[1,4;5,4]", r"$\eta(\tau)^4\eta(5\tau)^4$, $k=4$", "#1B7F79", "--"),
         ("[1,2;11,2]", r"$\eta(\tau)^2\eta(11\tau)^2$, $k=2$", "#9C3D1F", "-."),
         ("[1,1;23,1]", r"$\eta(\tau)\eta(23\tau)$, $k=1$", "#C99A2E", ":")]
s = np.linspace(0.02, 14, 300)
fig, ax = plt.subplots(figsize=(4.4, 2.7))
for q, lab, col, ls in forms:
    L = pari(f"lfunetaquo(Mat({q}))")
    vals = [float(pari.lfun(L, float(x))) for x in s]
    ax.plot(s, vals, color=col, ls=ls, lw=1.4, label=lab)
    k = float(pari.lfunparams(L)[1])
    ax.plot([k/2], [float(pari.lfun(L, k/2))], "o", color=col, ms=3.5)
ax.axhline(1, color="grey", lw=0.6)
ax.set_xlabel(r"$s$ (real)"); ax.set_ylabel(r"$L(s,f)=\Pr(G_s\geq 2\pi M_f)$")
ax.set_xlim(0, 14); ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("../work/figs/ch11_lvalues.pdf")
