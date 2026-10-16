import numpy as np, json, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 220})
D = "../work/figs/"; L = "../work/"
import sys
which = sys.argv[1] if len(sys.argv) > 1 else "all"

def F_st(x):
    t = np.arccos(np.clip(x / 2, -1, 1)); return 1 - (t - np.sin(2 * t) / 2) / np.pi
if which in ("all", "kl"):
    # Kloosterman: Katz histogram and paths
    K = np.load(L + "kloost_10007.npy"); xs = np.linspace(-2, 2, 400)
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.hist(K, bins=60, density=True, color="#E8A87C", edgecolor="none"); ax.plot(xs, np.sqrt(4 - xs**2) / (2 * np.pi), color="#6B2D5C", lw=1.6)
    ax.set_xlabel(r"$\mathrm{Kl}(a;p)/\sqrt{p}$, $p=10007$, all $a$"); plt.savefig(D + "kloosterman_katz.pdf"); plt.close()
    paths = np.load(L + "kloost_paths.npy"); rfs = np.load(L + "rfs_paths.npy")
    fig, axs = plt.subplots(2, 4, figsize=(6.4, 3.4))
    cm = plt.get_cmap("plasma")
    for k in range(4):
        for row, P, lab in ((0, paths[k], "Kloosterman path"), (1, rfs[k], "random Fourier series")):
            a = axs[row, k]; pts = np.column_stack([P.real, P.imag]); seg = np.stack([pts[:-1], pts[1:]], 1)
            a.add_collection(LineCollection(seg, colors=cm(np.linspace(0, 0.95, len(seg))), linewidths=0.9))
            a.plot([0], [0], "o", color="k", ms=2.5); a.plot([P[-1].real], [P[-1].imag], "s", color="k", ms=2.5)
            a.set_aspect("equal"); a.autoscale(); a.set_xticks([]); a.set_yticks([])
            for sp_ in a.spines.values(): sp_.set_visible(False)
            if k == 0: a.set_ylabel(lab, fontsize=7)
    fig.suptitle(r"top: $t\mapsto p^{-1/2}\sum_{x\leq tp}e((ax+\bar x)/p)$, $p=10007$; bottom: $tST_0+\sum_{h\neq0}\frac{e(ht)-1}{2\pi i h}ST_h$", fontsize=7.5)
    plt.tight_layout(); plt.savefig(D + "kloosterman_paths.pdf"); plt.close()

if which in ("all", "cl"):
    o = json.load(open(L + "stage9c.json"))
    cl = o["cohen_lenstra"]
    fig, ax = plt.subplots(1, 3, figsize=(6.2, 2.2), sharey=True)
    for a, p, col in zip(ax, ("3", "5", "7"), ("#2E86AB", "#F18F01", "#C73E1D")):
        ob = cl["imaginary"][p]["observed"][:3]; th = cl["imaginary"][p]["cohen_lenstra"][:3]
        x = np.arange(3); a.bar(x - 0.2, ob, 0.4, color=col, label="observed"); a.bar(x + 0.2, th, 0.4, color=col, alpha=0.4, hatch="//", label="Cohen Lenstra")
        a.set_yscale("log"); a.set_xticks(x); a.set_title(f"{p} rank of class group", fontsize=8); a.set_xlabel("rank")
    ax[0].legend(frameon=False, fontsize=6.5); plt.tight_layout(); plt.savefig(D + "cohen_lenstra.pdf"); plt.close()
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    for kind, col, target in (("imaginary", "#2E86AB", 2.0), ("real", "#C73E1D", 4 / 3)):
        w = cl[kind]["windows"]; ax.plot([v[0] for v in w], [v[1] for v in w], "o-", color=col, ms=3, label=f"{kind}")
        ax.axhline(target, color=col, ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel(r"$|D|$"); ax.set_ylabel(r"mean of $|\mathrm{Cl}(D)[3]|$"); ax.legend(frameon=False, fontsize=7)
    plt.savefig(D + "davenport_heilbronn.pdf"); plt.close()
    pr = o["poonen_rains"]
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.3))
    x = np.arange(5)
    ax[0].bar(x - 0.2, pr["sel2_observed"][:5], 0.4, color="#5C4D7D", label="6121 curves"); ax[0].bar(x + 0.2, pr["poonen_rains"][:5], 0.4, color="#E0A458", label="Poonen Rains")
    ax[0].set_xlabel(r"$\dim\,\mathrm{Sel}_2$"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("2 Selmer ranks", fontsize=8)
    ax[1].bar(x, pr["rank_dist"][:5], color=plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 5))); ax[1].set_xlabel("rank"); ax[1].set_title(f"ranks (mean {pr['mean_rank']:.2f})", fontsize=8)
    plt.tight_layout(); plt.savefig(D + "selmer_ranks.pdf"); plt.close()

if which in ("all", "rm"):
    o = json.load(open(L + "stage9b.json"))
    d = np.load(L + "brw_levels.npy"); tg = d[0]
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5))
    cm = plt.get_cmap("plasma")
    for k, row in enumerate(d[1:]): ax[0].plot(tg, row, color=cm(k / (len(d) - 1)), lw=0.8)
    ax[0].set_xlabel("$t$"); ax[0].set_title("Sato Tate random Euler product, primes up to $e^{e^{k/2}}$", fontsize=7.5)
    m = o["random_field_maxima"]; Xs = sorted(int(k) for k in m)
    ax[1].errorbar([math.log(math.log(X)) for X in Xs], [m[str(X)]["mean_max"] for X in Xs], yerr=[m[str(X)]["sd_max"] / math.sqrt(30) for X in Xs], fmt="o", color="#C73E1D", label="mean maximum")
    ll = np.linspace(1.8, 2.7, 50); ax[1].plot(ll, ll - 0.75 * np.log(ll), color="#2E86AB", label=r"$\log\log X-\frac{3}{4}\log\log\log X$")
    ax[1].set_xlabel(r"$\log\log X$"); ax[1].legend(frameon=False, fontsize=6.5); ax[1].set_title("maximum over $t\\in[0,1]$", fontsize=8)
    plt.tight_layout(); plt.savefig(D + "random_field.pdf"); plt.close()
    h = o["harper"]; xs = sorted(int(k) for k in h)
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.semilogx(xs, [h[str(x)]["E|S|/sqrt x"] for x in xs], "o-", color="#5C4D7D", label=r"$\mathbb{E}|S(x)|/\sqrt{x}$ (400 draws)")
    ax.semilogx(xs, [0.98 * h[str(x)]["loglog^-1/4"] for x in xs], "--", color="#E0A458", label=r"$0.98(\log\log x)^{-1/4}$")
    ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=7); plt.savefig(D + "harper.pdf"); plt.close()
    # multiplicative chaos picture: |random Euler product|^2 as a measure on [0,1]
    rng = np.random.default_rng(3)
    def primes_upto(n):
        s = np.ones(n + 1, bool); s[:2] = False
        for i in range(2, int(n ** .5) + 1):
            if s[i]: s[i*i::i] = False
        return np.nonzero(s)[0]
    P = primes_upto(200000); th = rng.uniform(0, 2 * np.pi, len(P)); tg = np.linspace(0, 30, 12001)
    logm = np.zeros_like(tg)
    for i in range(0, len(P), 3000):
        p = P[i:i + 3000]; logm += -2 * (np.log(np.abs(1 - np.exp(1j * (th[i:i + 3000][None, :] - np.outer(tg, np.log(p)))) / np.sqrt(p)[None, :]))).sum(1)
    dens = np.exp(logm); dens /= dens.mean()
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    ax.fill_between(tg, dens, 1e-3, color="#6B2D5C", lw=0); ax.set_yscale("log"); ax.set_ylim(1e-3, dens.max() * 2); ax.set_xlabel("$t$")
    ax.set_title(r"$|\prod_{p\leq 2\cdot10^5}(1-e^{i\theta_p}p^{-1/2-it})^{-1}|^2$ normalised, log scale: a multiplicative chaos picture", fontsize=7.5)
    ax.set_xlim(0, 30); plt.savefig(D + "chaos.pdf"); plt.close()

if which in ("all", "delta"):
    o = json.load(open(L + "stage9a.json"))
    ts, la = np.load(L + "delta_logabs.npy")
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    m = (ts > 1000) & (ts < 1040)
    ax[0].plot(ts[m], la[m], color="#2E86AB", lw=0.7); ax[0].set_xlabel("$t$"); ax[0].set_ylabel(r"$\log|L(\frac{1}{2}+it,\Delta)|$")
    tc, mx, pred = np.load(L + "delta_fhk.npy"); mm = (tc > 1000) & (tc < 1040)
    ax[0].plot(tc[mm], mx[mm], "v", color="#C73E1D", ms=3); ax[0].set_title("values on the critical line and maxima over unit windows", fontsize=7.5)
    z = (la - la.mean()) / la.std(); g = np.linspace(-4, 3, 200)
    ax[1].hist(z, bins=80, density=True, color="#9BC1BC", edgecolor="none"); ax[1].plot(g, np.exp(-g * g / 2) / math.sqrt(2 * math.pi), color="#5C4D7D", lw=1.4)
    ax[1].set_title("Selberg's central limit theorem", fontsize=8); ax[1].set_xlabel("standardised $\\log|L|$")
    plt.tight_layout(); plt.savefig(D + "selberg_clt.pdf"); plt.close()
    pc = o["pair_correlation"]; sp = np.load(L + "delta_spacings.npy")
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.3))
    ax[0].bar(pc["bins"], pc["empirical"], width=0.1, color="#F18F01", alpha=0.85, label="zeros of $L(s,\\Delta)$")
    u = np.linspace(0.01, 3, 300); ax[0].plot(u, 1 - (np.sin(np.pi * u) / (np.pi * u)) ** 2, color="#2E1F27", lw=1.4, label="Montgomery")
    ax[0].set_xlabel("$u$"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("pair correlation", fontsize=8)
    ax[1].hist(sp, bins=50, density=True, color="#A23B72", alpha=0.8, edgecolor="none"); s = np.linspace(0, 3, 300)
    ax[1].plot(s, 32 / np.pi**2 * s * s * np.exp(-4 * s * s / np.pi), color="#2E1F27", lw=1.4); ax[1].set_xlabel("normalised spacing"); ax[1].set_title("nearest neighbours vs GUE surmise", fontsize=8)
    plt.tight_layout(); plt.savefig(D + "delta_pair_correlation.pdf"); plt.close()
print("ok", which)
