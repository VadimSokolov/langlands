"""Colour figures for Chapter 22 (degree one): gl1_hecke, gl1_landau, gl1_cm, gl1_zeros."""
import numpy as np, json, matplotlib, math
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
INK, RUST, TEAL, GOLD, VIOLET = "#1F3B63", "#9C3D1F", "#2A7F7F", "#C8961E", "#6B4C9A"
d = np.load("gl1_data.npz"); theta = d["theta"]; P = d["P"]; ap = d["ap"]; P1 = d["P1"]; a = d["a"]; b = d["b"]
phi = (4*theta) % (2*np.pi)

# gl1_hecke: Gaussian primes (four associates) coloured by 4 theta; histogram; angle law of E
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
m = P1 < 3000
A = np.concatenate([a[m], -b[m], -a[m], b[m]]); B = np.concatenate([b[m], a[m], -b[m], -a[m]]); C = np.tile(phi[m], 4)
sc = ax[0].scatter(A, B, c=C, cmap="twilight", s=3, linewidths=0)
ax[0].set_aspect("equal"); ax[0].set_xlabel("$a$"); ax[0].set_ylabel("$b$"); ax[0].set_title("Gaussian primes, norm $<3000$", fontsize=8)
cb = plt.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.04); cb.set_ticks([0, np.pi, 2*np.pi]); cb.set_ticklabels(["0", "$\\pi$", "$2\\pi$"]); cb.ax.tick_params(labelsize=6)
ax[1].hist(phi, bins=40, density=True, color=INK, alpha=0.85, edgecolor="white", linewidth=0.3)
ax[1].axhline(1/(2*np.pi), color=RUST, lw=1.2)
ax[1].set_xlabel("$4\\theta_p$ mod $2\\pi$"); ax[1].set_ylabel("density"); ax[1].set_title("Hecke angles, $p<10^7$", fontsize=8); ax[1].set_ylim(0, 0.2)
thS = np.arccos(np.clip(ap/(2*np.sqrt(P1)), -1, 1)); nall = len(P1) + np.sum(P % 4 == 3)
ax[2].hist(thS, bins=60, weights=np.full(len(thS), 1.0/nall)/(np.pi/60), color=TEAL, alpha=0.85, edgecolor="white", linewidth=0.3)
ax[2].annotate("", xy=(np.pi/2, 1.5), xytext=(np.pi/2, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
ax[2].text(1.75, 1.35, "atom of mass $\\frac{1}{2}$", fontsize=7, color=INK)
ax[2].axhline(1/(2*np.pi), color=RUST, lw=1.2)
g = np.linspace(0, np.pi, 200); ax[2].plot(g, 2/np.pi*np.sin(g)**2, color=GOLD, lw=1.2, label="Sato Tate")
ax[2].set_xlabel("$\\theta_p$ for $y^2=x^3-x$"); ax[2].set_title("half uniform, half an atom", fontsize=8); ax[2].set_ylim(0, 1.6); ax[2].legend(fontsize=7, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig("../book/figs/gl1_hecke.pdf")

# gl1_landau: Mertens per class; Landau count; atoms
g4 = json.load(open("g4.json"))
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.4))
P1s = P[P % 4 == 1].astype(float); P3s = P[P % 4 == 3].astype(float)
for Ps, col, lab, th in [(P1s, INK, "$p\\equiv1$", g4["M1th"]), (P3s, RUST, "$p\\equiv3$", g4["M3th"])]:
    cs = np.cumsum(1/Ps) - 0.5*np.log(np.log(Ps))
    idx = np.unique(np.geomspace(1, len(Ps)-1, 400).astype(int))
    ax[0].semilogx(Ps[idx], cs[idx], color=col, lw=1, label=lab); ax[0].axhline(th, color=col, lw=0.8, ls="--")
ax[0].set_xlabel("$x$"); ax[0].set_ylabel(r"$\sum_{p\leq x}1/p-\frac{1}{2}\log\log x$"); ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("Mertens constants mod 4", fontsize=8); ax[0].set_ylim(-0.6, 0.3)
xs = [r[0] for r in g4["rows"]]; ratio = [r[3] for r in g4["rows"]]
ax[1].semilogx(xs, ratio, "o-", color=INK, ms=4); ax[1].axhline(1, color=RUST, lw=0.8)
ax[1].set_xlabel("$x$"); ax[1].set_ylabel("count$/(K'x/\\sqrt{\\log x})$"); ax[1].set_title("no prime factor $\\equiv3\\ (4)$", fontsize=8); ax[1].set_ylim(0.98, 1.05)
sig = sorted(float(k) for k in g4["atoms"]); ss = np.array(sig) - 1
a3 = [g4["atoms"][str(s)][0] for s in sig]; a1 = [g4["atoms"][str(s)][1] for s in sig]
ax[2].loglog(ss, a3, "o", color=RUST, ms=4, label="no $p\\equiv3$"); ax[2].loglog(ss, a1, "s", color=INK, ms=4, label="no $p\\equiv1$")
ax[2].loglog(ss, g4["eh3"]*np.sqrt(ss), color=RUST, lw=0.9); ax[2].loglog(ss, g4["eh1"]*np.sqrt(ss), color=INK, lw=0.9)
ax[2].set_xlabel(r"$\sigma-1$"); ax[2].set_ylabel("atom at zero"); ax[2].legend(fontsize=7, frameon=False); ax[2].set_title(r"$e^{-h_C(1)}(\sigma-1)^{1/2}$", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/gl1_landau.pdf")

# gl1_cm: Laplace transforms over Q and K; Thorin densities of the Frobenius functional
g3 = json.load(open("g3.json"))
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
lam = np.linspace(0, 8, 200)
cols = {"1.1": GOLD, "1.01": TEAL, "1.001": INK}
for s, rows in g3.items():
    L = np.array(rows)
    ax[0].plot(L[:, 0], L[:, 1], "o-", color=cols[s], ms=3.5, lw=0.8, label="over $\\mathbb{Q}$, $\\sigma=%s$" % s)
    ax[0].plot(L[:, 0], L[:, 3], "s--", color=cols[s], ms=3.5, lw=0.8, mfc="white", label="over $\\mathbb{Q}(i)$, $\\sigma=%s$" % s)
ax[0].plot(lam, 1/(1+lam), color=RUST, lw=1.4, label="$(1+\\lambda)^{-1}$"); ax[0].plot(lam, 1/(1+lam)**2, color=VIOLET, lw=1.4, label="$(1+\\lambda)^{-2}$")
ax[0].set_xlabel("$\\lambda$"); ax[0].set_ylabel(r"$\mathbb{E}\,e^{-\lambda(\sigma-1)X_\sigma}$"); ax[0].legend(fontsize=5.5, frameon=False, ncol=2); ax[0].set_title("Gamma shape 1 over $\\mathbb{Q}$, 2 over $\\mathbb{Q}(i)$", fontsize=8)
t = np.linspace(1.001, 12, 600)
ax[1].plot(t, 1/(2*np.pi*t*np.sqrt(t-1)), color=INK, lw=1.3, label="CM: $\\frac{1}{2\\pi t\\sqrt{t-1}}$, mass $\\frac{1}{2}$")
ax[1].plot(t, 2/np.pi*np.sqrt(t-1)/t**2, color=RUST, lw=1.3, label="Sato Tate: $\\frac{2\\sqrt{t-1}}{\\pi t^2}$, mass $1$")
ax[1].set_ylim(0, 0.6); ax[1].set_xlabel("$t$"); ax[1].set_ylabel("Thorin density"); ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("Thorin densities of $\\Xi_\\sigma(\\cos^2)$", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/gl1_cm.pdf")

# gl1_zeros: Xi ratio vs product; shifted Thorin density V_a
z = np.load("chi4_zeros.npy"); g5 = json.load(open("g5.json"))
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
rows = np.array(g5["rows"])
ax[0].semilogy(rows[:, 0], rows[:, 1], "o", color=INK, ms=5, label="$\\Xi_{\\chi_4}(\\sqrt{\\lambda})/\\Xi_{\\chi_4}(0)$")
ax[0].semilogy(rows[:, 0], rows[:, 2], "x", color=RUST, ms=6, label="$\\prod_k(1+\\lambda/\\gamma_k^2)$, 3130 zeros")
ax[0].set_xlabel("$\\lambda$"); ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("the Thorin product of $L(s,\\chi_4)$", fontsize=8)
zz = np.linspace(0.5, 400, 1500); y = np.sqrt(zz)
for aa, col in [(0.25, INK), (0.6, TEAL), (1.5, GOLD)]:
    V = np.zeros_like(y)
    for gk in z[:400]:
        V += aa/(aa*aa + (y - gk)**2) + aa/(aa*aa + (y + gk)**2)
    ax[1].plot(zz, V/np.pi, color=col, lw=1.1, label="$a=%g$" % aa)
for gk in z[:5]: ax[1].axvline(gk*gk, color="0.6", lw=0.6, ls=":")
ax[1].set_xlabel("$z$"); ax[1].set_ylabel("$2\\sqrt{z}\\,V_a(z)$"); ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("zeros plus Cauchy noise at $\\frac{1}{2}+a$", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/gl1_zeros.pdf")
print("ok")
