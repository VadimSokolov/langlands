"""Colour figures for Chapter 24: cal_parity, cal_cm, cal_g2."""
import numpy as np, json, matplotlib, math
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpmath import mp, mpf, quad, exp, cosh, pi, sqrt, inf
from scipy.integrate import quad as sq
mp.dps = 15
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
INK, RUST, TEAL, GOLD, VIOLET = "#1F3B63", "#9C3D1F", "#2A7F7F", "#C8961E", "#6B4C9A"
cols = {0: INK, 1: TEAL, 3: GOLD, 11: RUST}

# cal_parity
c1 = json.load(open("c1.json"))
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.4))
ts = np.linspace(-4, 4, 400)
def f_w(t, w):
    if w == 0: return 1/np.cosh(np.pi*t)
    a = w/2
    return sq(lambda s: (1/np.cosh(np.pi*(t - s)))*(a/np.pi)/(a*a + s*s), -np.inf, np.inf)[0]
for w in [0, 1, 3, 11]:
    ax[0].plot(ts, [f_w(t, w) for t in ts], color=cols[w], lw=1.2, label="$w=%d$" % w)
ax[0].set_xlabel("$t$"); ax[0].set_ylabel("density of $C_w$"); ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("secant law plus Cauchy of scale $w/2$", fontsize=8)
xs = np.geomspace(0.01, 50, 80)
def delta(x, w):
    return float(quad(lambda u: exp(-mpf(w)*u/2)/cosh(u/2)*exp(-u*u/(4*mpf(x))), [0, inf])/(4*sqrt(pi*mpf(x))))
for w in [0, 1, 3, 11]:
    ax[1].loglog(xs, [delta(x, w) for x in xs], color=cols[w], lw=1.2, label="$w=%d$" % w)
ax[1].set_xlabel("$x$"); ax[1].set_ylabel(r"$\Delta_w(x)$"); ax[1].legend(fontsize=7, frameon=False); ax[1].set_title(r"$\frac{1}{4}\mathbb{E}\,e^{-xC_w^2}$", fontsize=8)
H = np.array(c1["heat"])
ax[2].semilogx(H[:, 0], H[:, 1], "o-", color=INK, ms=3.5, lw=1, label="$\\Gamma_{\\mathbb{C}}(s+\\frac{1}{2})$")
ax[2].semilogx(H[:, 0], H[:, 2], "s-", color=RUST, ms=3.5, lw=1, label="$\\Gamma_{\\mathbb{R}}(s+\\frac{1}{2})^2$")
ax[2].axhline(0, color="0.5", lw=0.6); ax[2].set_xlabel("$x$"); ax[2].set_ylabel("heat trace of $11a1$"); ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("odd data, two gamma factors", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/cal_parity.pdf")

# cal_cm
d = np.load("ec_qi.npz"); x = d["ap"]/np.sqrt(d["norms"]); deg = d["deg"]
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.4))
g = np.linspace(-2, 2, 400); st = np.sqrt(np.clip(1 - g*g/4, 0, None))/np.pi
ax[0].hist(x, bins=50, density=True, color=INK, alpha=0.8, edgecolor="white", linewidth=0.3); ax[0].plot(g, st, color=RUST, lw=1.3)
ax[0].set_xlabel(r"$a_P/\sqrt{NP}$"); ax[0].set_ylabel("density"); ax[0].set_title("all %d primes, $NP\\leq10^7$" % len(x), fontsize=8)
ax[1].hist(x[deg == 2], bins=15, density=True, color=TEAL, alpha=0.8, edgecolor="white", linewidth=0.3); ax[1].plot(g, st, color=RUST, lw=1.3)
ax[1].set_xlabel(r"$a_P/\sqrt{NP}$"); ax[1].set_title("the %d inert primes" % int(np.sum(deg == 2)), fontsize=8)
t = np.linspace(1.0001, 12, 500); ax[2].plot(t, 2/np.pi*np.sqrt(t - 1)/t**2, color=INK, lw=1.3); ax[2].fill_between(t, 0, 2/np.pi*np.sqrt(t - 1)/t**2, color=INK, alpha=0.15)
ax[2].set_xlabel("$t$"); ax[2].set_ylabel("Thorin density"); ax[2].set_title(r"$\frac{2\sqrt{t-1}}{\pi t^2}$, mass $1$", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/cal_cm.pdf")

# cal_g2
c3 = json.load(open("c3.json")); z = np.load("zeros249.npy"); xs2 = np.load("g2_x.npy")
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.4))
ax[0].vlines(z, 0, 1, color=INK, lw=0.7); ax[0].set_yticks([]); ax[0].set_xlabel("$\\gamma$"); ax[0].set_title("zeros $1+i\\gamma$, conductor 249", fontsize=8)
R = np.array(c3["rows"])
ax[1].semilogy(R[:, 0], R[:, 1], "o", color=INK, ms=5, label="$\\Xi_C(\\sqrt{\\lambda})/\\Xi_C(0)$"); ax[1].semilogy(R[:, 0], R[:, 2], "x", color=RUST, ms=6, label="product over %d zeros" % c3["nzeros"])
ax[1].set_xlabel("$\\lambda$"); ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("the Thorin product", fontsize=8)
# USp(4) trace density by Monte Carlo from the Weyl density
rng = np.random.default_rng(3); n = 4000000
th = rng.uniform(0, np.pi, (n, 2)); wgt = (np.cos(th[:, 0]) - np.cos(th[:, 1]))**2*np.sin(th[:, 0])**2*np.sin(th[:, 1])**2
tr = 2*np.cos(th[:, 0]) + 2*np.cos(th[:, 1])
hh, edges = np.histogram(tr, bins=80, range=(-4, 4), weights=wgt, density=True)
mid = (edges[1:] + edges[:-1])/2
ax[2].hist(xs2, bins=40, density=True, color=INK, alpha=0.8, edgecolor="white", linewidth=0.3, label="$a_p/\sqrt{p}$, first 20000 primes")
ax[2].plot(mid, hh, color=RUST, lw=1.3, label="$\\mathrm{USp}(4)$ trace law")
ax[2].set_xlabel("$a_p/\\sqrt{p}$"); ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("Sato Tate for genus two", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/cal_g2.pdf")
print("ok")
