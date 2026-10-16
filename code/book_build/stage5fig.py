import numpy as np, json, math, cypari2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
D = "../work/figs/"
C = {"ink": "#1F3B63", "rust": "#9C3D1F", "teal": "#1B7F79", "gold": "#C99A2E", "plum": "#6B3F7A", "sky": "#8FB3DE", "sand": "#E2C9A6"}
pari = cypari2.Pari(); pari.allocatemem(2 * 10**9)
o = json.load(open("../work/stage5.json"))

# --- E redone: symmetric Jacobi form of the tree passage time generator
def jacobi(q, n):
    rho = 2 * math.sqrt(q) / (q + 1)
    off = np.full(n - 1, rho / 2); off[0] = 1 / math.sqrt(q + 1)
    return np.eye(n) - np.diag(off, 1) - np.diag(off, -1)
tree = {}
for q in (2, 3, 5):
    rho = 2 * math.sqrt(q) / (q + 1); res = {"rho": rho, "gap": 1 - rho, "limit_mean": (q + 1) / (q - 1), "limit_var": ((q + 1) / (q - 1)) ** 3}
    for n in (100, 400, 1600, 6400):
        lam = np.linalg.eigvalsh(jacobi(q, n))
        u = np.clip((1 - lam) / rho, -1, 1); ph = np.sort(np.arccos(u)) / math.pi
        res[n] = dict(min_rate=float(lam.min()), max_rate=float(lam.max()), ET_over_n=float(np.sum(1 / lam) / n), VT_over_n=float(np.sum(1 / lam**2) / n),
                      ks_arcsine=float(np.max(np.abs(ph - (np.arange(1, n + 1) - 0.5) / n))))
    tree[q] = res
o["tree_sym"] = tree
json.dump(o, open("../work/stage5.json", "w"), indent=1)
for q in tree: print(q, {n: (round(v["min_rate"], 5), round(v["ET_over_n"], 5), round(v["VT_over_n"], 4), round(v["ks_arcsine"], 5)) for n, v in tree[q].items() if isinstance(n, int)}, tree[q]["gap"])

# Fig 1: degenerate Thorin atom = rank / 2
fig, ax = plt.subplots(figsize=(4.4, 2.4))
curves = [("11a1", "[0,-1,1,-10,-20]", 0, C["ink"]), ("37a1", "[0,0,1,-1,0]", 1, C["teal"]), ("389a1", "[0,1,1,-2,0]", 2, C["gold"]), ("5077a1", "[0,0,1,-7,6]", 3, C["rust"])]
ss = np.logspace(-6, 2, 120)
for name, c, r, col in curves:
    pari(f"L=lfuncreate(ellinit({c}));")
    v = np.array([abs(float(pari(f"lfunlambda(L,1+{math.sqrt(s)})"))) for s in ss])
    ax.loglog(ss, v / v[-1], color=col, lw=1.3, label=f"{name}, rank {r}")
ax.set_xlabel(r"$s$"); ax.set_ylabel(r"$|\Lambda(1+\sqrt{s})|$, normalised"); ax.legend(frameon=False, fontsize=7)
plt.savefig(D + "thorin_rankatom.pdf"); plt.close()

# Fig 2: local Thorin combs versus global zeros for Delta
pari("L=lfunmf(mfinit([1,12]),mfDelta());")
zD = [float(z) for z in pari("lfunzeros(L,60)") if float(z) > 1e-6]
fig, ax = plt.subplots(figsize=(4.4, 2.2))
for i, (p, col) in enumerate([(2, C["rust"]), (3, C["gold"]), (5, C["teal"]), (7, C["plum"])]):
    x = float(pari(f"ramanujantau({p})")) / p ** 5.5; th = math.acos(x / 2); L = math.log(p)
    pts = sorted({(sg * th + 2 * math.pi * k) / L for k in range(-60, 61) for sg in (1, -1)})
    pts = [t for t in pts if 0 < t < 60]
    ax.plot(pts, [i + 1] * len(pts), "|", color=col, ms=9, mew=1.2)
ax.plot(zD, [0] * len(zD), "o", color=C["ink"], ms=3.2)
ax.set_yticks([0, 1, 2, 3, 4]); ax.set_yticklabels(["zeros of $\\Lambda$", "$p=2$", "$p=3$", "$p=5$", "$p=7$"]); ax.set_xlabel(r"$t$"); ax.set_xlim(0, 60)
plt.savefig(D + "thorin_combs.pdf"); plt.close()

# Fig 3: parabolic Thorin regions in the s plane
fig, ax = plt.subplots(figsize=(4.4, 2.6))
y = np.linspace(-6, 6, 400)
for sg, col, lab in [(0.5, C["sky"], r"$\sigma=1/2$ (critical strip)"), (7 / 64, C["gold"], r"$\sigma=7/64$ (Kim and Sarnak)")]:
    ax.fill_betweenx(y, -40, sg**2 - y**2 / (4 * sg**2), color=col, alpha=0.35, lw=0, label=lab)
g2 = [-g * g for g in zD if g * g < 40]
ax.plot(g2, [0] * len(g2), "o", color=C["ink"], ms=3.2, label=r"$-\gamma_k^2$ for $\Delta$")
ax.axhline(0, color="k", lw=0.6); ax.set_xlim(-40, 3); ax.set_ylim(-6, 6)
ax.set_xlabel(r"$\mathrm{Re}\,s$"); ax.set_ylabel(r"$\mathrm{Im}\,s$"); ax.legend(frameon=False, fontsize=7, loc="upper left")
plt.savefig(D + "thorin_parabola.pdf"); plt.close()

# Fig 4: tree passage time Thorin measure and law
q, n = 2, 1600; rho = 2 * math.sqrt(q) / (q + 1)
lam = np.linalg.eigvalsh(jacobi(q, n))
fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.1))
ax[0].hist(lam, bins=50, density=True, color=C["sky"], edgecolor="none")
xx = np.linspace(1 - rho + 1e-4, 1 + rho - 1e-4, 400)
ax[0].plot(xx, 1 / (math.pi * np.sqrt(rho**2 - (1 - xx) ** 2)), color=C["ink"], lw=1.2); ax[0].set_ylim(0, 3)
ax[0].set_title("Thorin rates, $q=2$, $n=1600$", fontsize=8); ax[0].set_xlabel("rate")
sim, gg = np.load("../work/tree_sim.npy")
ax[1].hist(sim, bins=60, density=True, color=C["sand"], edgecolor="none", label="simulated walk")
ax[1].hist(gg, bins=60, density=True, histtype="step", color=C["rust"], lw=1.1, label=r"$\sum E_j/\lambda_j$")
ax[1].set_title(r"$T_{100}$, $q=2$", fontsize=8); ax[1].legend(frameon=False, fontsize=6)
plt.tight_layout(); plt.savefig(D + "thorin_tree.pdf"); plt.close()

# Fig 5: exponential law on isogeny graphs
iso = o["isogeny_exponential"]
hs = np.array([r["h"] for r in iso]); er = np.array([r["max_sup_error"] for r in iso]); ab = np.array([r["max_aldous_brown_bound"] for r in iso]); rb = np.array([r["ramanujan_bound"] for r in iso])
fig, ax = plt.subplots(figsize=(4.0, 2.3))
ax.loglog(hs, rb, "s-", color=C["gold"], ms=4, lw=1, label="Ramanujan bound")
ax.loglog(hs, ab, "^-", color=C["teal"], ms=4, lw=1, label=r"$\tau_2/E_\pi\tau_v$")
ax.loglog(hs, er, "o-", color=C["ink"], ms=4, lw=1.2, label="observed sup error")
ax.set_xlabel("vertices $h$"); ax.legend(frameon=False, fontsize=7)
plt.savefig(D + "thorin_isogeny.pdf"); plt.close()
print("figs ok")
