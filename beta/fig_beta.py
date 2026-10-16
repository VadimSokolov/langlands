"""Figures beta:fig:thorin (figs/beta_thorin.pdf) and beta:fig:arg (figs/beta_argument.pdf) of Chapter 18.

Written for the book on 2026-10-13: the coauthor sent the two figures without a script, drawn with a sans serif
font and colours outside the book's palette, and Figure beta:fig:arg wrote sigma for the alpha of the text.
Run on Hopper through misc/tools/hopper_run.sh, never locally:
    python3 fig_beta.py compute   evaluates every curve with mpmath and writes fig_beta.json
    python3 fig_beta.py plot      reads fig_beta.json and draws beta_thorin.pdf and beta_argument.pdf
    python3 fig_beta.py           does both, and skips the plot if matplotlib is missing
Notation as in the chapter: Lambda(s) = (4/pi)^(s/2) Gamma((s+1)/2) beta(s), beta(s) = L(s, chi_-4);
c_a = Lambda'/Lambda(a); G_a(t) = (Re psi((a+1+it)/2) - psi((a+1)/2))/2; P_a(t) = Re Lambda'/Lambda(a+it) - c_a - G_a(t);
A_s(t) = log(4/pi)/2 + Re psi((s+1+it)/2)/2 and S(s) = -zeta'/zeta(s) - log 2/(2^s - 1) (equation beta:eq:AS);
the bound of Corollary beta:cor:lower is (a - 1/2)/(1/2 + delta) (A_{1+delta}(t) - S(1+delta)).
"""
import json
import sys
import time

import mpmath as mp
import numpy as np

mp.mp.dps = 20
CHI = [0, 1, 0, -1]          # chi_-4 as a list indexed by k mod 4
ALPHAS = ["1.0", "0.75", "0.55"]
T_MAX, STEP = 30.0, 0.005    # left panels
ALPHA_DEC, STEP_DEC = 1.5, 0.02   # right panel of beta_thorin
ALPHA_MIN, DELTA = 0.75, 2.15     # right panel of beta_argument


def L(s):
    return mp.dirichlet(s, CHI)


def dlogLambda(s):
    return 0.5 * mp.log(4 / mp.pi) + 0.5 * mp.digamma((s + 1) / 2) + mp.dirichlet(s, CHI, 1) / L(s)


def Lambda(s):
    return mp.power(4 / mp.pi, s / 2) * mp.gamma((s + 1) / 2) * L(s)


def A(sig, t):
    return 0.5 * mp.log(4 / mp.pi) + 0.5 * mp.re(mp.digamma((sig + 1 + 1j * t) / 2))


def S(sig):
    return -mp.zeta(sig, 1, 1) / mp.zeta(sig) - mp.log(2) / (mp.power(2, sig) - 1)


def zeros_on_line(tmax):
    # Lambda(1/2+it) is real; locate its sign changes and refine them
    f = lambda t: mp.re(Lambda(mp.mpf(0.5) + 1j * t))
    grid = np.arange(0.1, tmax + 0.05, 0.05)
    vals = [f(t) for t in grid]
    out = []
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            out.append(float(mp.findroot(f, (grid[i], grid[i + 1]), solver="anderson")))
    return out


def compute():
    t0 = time.time()
    D = {"meta": {"dps": mp.mp.dps, "t_max": T_MAX, "step": STEP, "mpmath": mp.__version__, "numpy": np.__version__}}
    t = np.round(np.arange(0.0, T_MAX + STEP / 2, STEP), 10)
    D["t"] = t.tolist()
    D["re"], D["mass"], D["mass_quad"] = {}, {}, {}
    for a in ALPHAS:
        al = mp.mpf(a)
        re, phb = [], []
        for x in t:
            s = al + 1j * mp.mpf(x)
            val = L(s)
            d = 0.5 * mp.log(4 / mp.pi) + 0.5 * mp.digamma((s + 1) / 2) + mp.dirichlet(s, CHI, 1) / val
            re.append(float(mp.re(d)))
            phb.append(float(mp.arg(val)))
        # arg Lambda(a+it) = t log(4/pi)/2 + Im log Gamma((a+1+it)/2) + arg beta(a+it), continuous from t = 0
        el = [float(0.5 * x * mp.log(4 / mp.pi) + mp.im(mp.loggamma((al + 1 + 1j * mp.mpf(x)) / 2))) for x in t]
        arg = np.array(el) + np.unwrap(np.array(phb))
        arg -= arg[0]
        re = np.array(re)
        quad = np.concatenate([[0.0], np.cumsum((re[1:] + re[:-1]) / 2 * np.diff(t))])  # integral of Re Lambda'/Lambda
        D["re"][a] = re.tolist()
        D["mass"][a] = (arg / np.pi).tolist()
        D["mass_quad"][a] = (quad / np.pi).tolist()
        print(f"alpha={a}: min Re {re.min():.6f}, mass(30) {arg[-1] / np.pi:.6f}, "
              f"max |arg - integral|/pi {np.abs(arg - quad).max() / np.pi:.2e}, {time.time() - t0:.0f} s", flush=True)
    D["zeros"] = zeros_on_line(T_MAX)
    print("zeros of beta on the line below 30:", [round(g, 6) for g in D["zeros"]], flush=True)

    al = mp.mpf(ALPHA_DEC)
    c = float(mp.re(dlogLambda(al)))
    td = np.round(np.arange(0.0, T_MAX + STEP_DEC / 2, STEP_DEC), 10)
    G = np.array([float(0.5 * (mp.re(mp.digamma((al + 1 + 1j * mp.mpf(x)) / 2)) - mp.digamma((al + 1) / 2))) for x in td])
    tot = np.array([float(mp.re(dlogLambda(al + 1j * mp.mpf(x)))) for x in td])
    P = tot - c - G
    D["dec"] = {"alpha": ALPHA_DEC, "t": td.tolist(), "c": c, "cG": (c + G).tolist(), "P": P.tolist(), "total": tot.tolist()}
    i = int(P.argmin())
    print(f"alpha=3/2: c={c:.6f}, min P={P[i]:.4f} at t={td[i]:.2f}, min G={G.min():.2e}", flush=True)

    al = mp.mpf(ALPHA_MIN)
    c75 = float(mp.re(dlogLambda(al)))
    sig = 1 + DELTA
    Ssig = S(mp.mpf(sig))
    k = (al - mp.mpf(0.5)) / (mp.mpf(0.5) + DELTA)
    bound = np.array([float(k * (A(sig, mp.mpf(x)) - Ssig)) for x in t])
    tcross = float(mp.findroot(lambda x: k * (A(sig, x) - Ssig) - c75, 2.4))
    D["min"] = {"alpha": ALPHA_MIN, "delta": DELTA, "c": c75, "bound": bound.tolist(), "t_cross": tcross}
    print(f"alpha=0.75: c={c75:.6f}, S(3.15)={float(Ssig):.6f}, bound exceeds c for t > {tcross:.4f}", flush=True)

    checks = {}
    for (a, x, want) in [(2, 1, 0.2353511), (2, 7, 0.7691193), (1.5, 4, 0.32628)]:
        got = float(mp.re(dlogLambda(mp.mpf(a) + 1j * mp.mpf(x))))
        checks[f"Re Lambda'/Lambda({a}+{x}i)"] = [got, want]
    for a, want in [(0.55, 0.15603), (0.75, 0.15591), (1.0, 0.15557), (1.5, 0.15422), (2.5, 0.14924)]:
        got = float(mp.re(dlogLambda(mp.mpf(a))) / (a - 0.5))
        checks[f"c_a/(a-1/2) at a={a}"] = [got, want]
    for name, (got, want) in checks.items():
        print(f"check {name}: {got:.7f} (text {want})", flush=True)
    D["checks"] = checks
    json.dump(D, open("fig_beta.json", "w"))
    print(f"wrote fig_beta.json in {time.time() - t0:.0f} s", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    INK, RUST, TEAL, GOLD, PLUM = "#1F3B63", "#9C3D1F", "#1B7F79", "#C99A2E", "#6B3F7A"
    COL = dict(zip(ALPHAS, [INK, RUST, TEAL]))
    LAB = {"1.0": r"$\alpha=1$", "0.75": r"$\alpha=0.75$", "0.55": r"$\alpha=0.55$"}
    D = json.load(open("fig_beta.json"))
    t = np.array(D["t"])

    def top_legend(ax, ncol):
        leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False, fontsize=7,
                        handlelength=1.6, columnspacing=1.2)
        for h in leg.get_lines():
            h.set_linewidth(1.6)

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for g in D["zeros"]:
        ax[0].axvline(g, color="0.82", lw=0.5, zorder=0)
    for a in ALPHAS:
        ax[0].plot(t, D["re"][a], color=COL[a], lw=1.0, label=LAB[a])
    ax[0].set_xlim(0, 30); ax[0].set_ylim(0, 3); ax[0].set_xlabel(r"$t$")
    ax[0].set_title(r"$\mathrm{Re}\,(\Lambda'/\Lambda)(\alpha+it)$", fontsize=8, pad=18)
    top_legend(ax[0], 3)
    dec = D["dec"]; td = np.array(dec["t"])
    ax[1].axhline(0, color="0.6", lw=0.5)
    ax[1].plot(td, dec["cG"], color=INK, lw=1.2, label=r"$c_\alpha+G_\alpha(t)$")
    ax[1].plot(td, dec["P"], color=RUST, lw=1.0, label=r"$P_\alpha(t)$")
    ax[1].plot(td, dec["total"], color=TEAL, lw=1.0, label="total")
    ax[1].set_xlim(0, 30); ax[1].set_xlabel(r"$t$")
    ax[1].set_title(r"Decomposition at $\alpha=3/2$", fontsize=8, pad=18)
    top_legend(ax[1], 3)
    plt.tight_layout(); plt.savefig("beta_thorin.pdf"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for a in ALPHAS:
        ax[0].plot(t, D["mass"][a], color=COL[a], lw=1.0, label=LAB[a])
    z = D["zeros"]
    ax[0].step([0] + z + [30], list(range(len(z) + 1)) + [len(z)], where="post", color=PLUM, lw=0.9, label=r"$N(t)$")
    ax[0].set_xlim(0, 30); ax[0].set_xlabel(r"$t$")
    ax[0].set_title(r"Thorin mass $\frac{1}{\pi}\arg\Lambda(\alpha+it)$", fontsize=8, pad=18)
    top_legend(ax[0], 4)
    m = D["min"]
    ax[1].plot(t, D["re"]["0.75"], color=RUST, lw=1.0, label=r"$\mathrm{Re}\,\Lambda'/\Lambda(0.75+it)$")
    ax[1].axhline(m["c"], color="0.35", lw=0.8, ls=":", label=r"$c_{0.75}$")
    ax[1].plot(t, m["bound"], color=GOLD, lw=1.2, label=r"lower bound, $\delta=2.15$")
    ax[1].set_xlim(0, 30); ax[1].set_ylim(0, 0.6); ax[1].set_xlabel(r"$t$")
    ax[1].set_title(r"Minimum on the real axis, $\alpha=0.75$", fontsize=8, pad=18)
    top_legend(ax[1], 3)
    plt.tight_layout(); plt.savefig("beta_argument.pdf"); plt.close()
    print("wrote beta_thorin.pdf and beta_argument.pdf with matplotlib", matplotlib.__version__, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("compute", "all"):
        compute()
    if mode in ("plot", "all"):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("matplotlib is not installed here; fig_beta.json is written, run `fig_beta.py plot` where it is")
            sys.exit(0 if mode == "all" else 1)
        plot()
