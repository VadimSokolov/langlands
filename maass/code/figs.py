"""Colour figures for Chapter 23: maass_bm, maass_geodesics, maass_zeta, maass_vertical, maass_spectrum, maass_heat."""
import numpy as np, json, matplotlib, math
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpmath import li
from scipy.integrate import quad
import lmfdb_r
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
INK, RUST, TEAL, GOLD, VIOLET = "#1F3B63", "#9C3D1F", "#2A7F7F", "#C8961E", "#6B4C9A"
rng = np.random.default_rng(7)
def reduce(x, y):
    for _ in range(80):
        x = x - np.round(x); r2 = x*x + y*y; m = r2 < 1
        if not m.any(): break
        x = np.where(m, -x/r2, x); y = np.where(m, y/r2, y)
    return x, y
th = np.linspace(np.pi/3, 2*np.pi/3, 100)
def domain(ax, top):
    ax.plot(np.cos(th), np.sin(th), color="0.3", lw=0.8); ax.plot([-0.5, -0.5], [np.sqrt(3)/2, top], color="0.3", lw=0.8); ax.plot([0.5, 0.5], [np.sqrt(3)/2, top], color="0.3", lw=0.8)

# maass_bm: path in H from i to time 1.5; samples in F coloured by time; height law
d = np.load("bm_traj.npz"); traj = d["traj"]; ys = d["ys"]
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1]))
dt = 1e-4; n = int(1.5/dt); x = 0.0; ly = 0.0; path = np.empty((n, 2))
for i in range(n):
    yv = math.exp(ly); x += math.sqrt(2)*yv*rng.standard_normal()*math.sqrt(dt); ly += math.sqrt(2)*rng.standard_normal()*math.sqrt(dt) - dt
    path[i] = (x, math.exp(ly))
sc0 = ax[0].scatter(path[::10, 0], path[::10, 1], c=np.arange(0, n, 10)*dt, cmap="viridis", s=1.5, linewidths=0)
ax[0].plot(0, 1, "o", color=RUST, ms=4); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$"); ax[0].set_title("on $\\mathbb{H}$ from $i$, time $1.5$", fontsize=8)
t = traj[:1500]
domain(ax[1], 4)
sc = ax[1].scatter(t[:, 0], t[:, 1], c=np.arange(len(t)), cmap="viridis", s=3, linewidths=0)
ax[1].set_xlim(-0.6, 0.6); ax[1].set_ylim(0.8, 4); ax[1].set_xlabel("$x$"); ax[1].set_title("on $X$, samples in $F$", fontsize=8)
bins = np.geomspace(1, 32, 25); h, _ = np.histogram(ys[ys >= 1], bins=bins)
dens = h/(len(ys)*np.diff(bins)); mid = np.sqrt(bins[1:]*bins[:-1])
ax[2].loglog(mid, dens, "o", color=INK, ms=3.5, label="simulation"); ax[2].loglog(mid, 3/(np.pi*mid**2), color=RUST, lw=1.2, label="$3/(\\pi y^2)$")
ax[2].set_xlabel("$y$"); ax[2].set_ylabel("density"); ax[2].set_title("law of the height", fontsize=8); ax[2].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig("../book/figs/maass_bm.pdf")

# maass_geodesics: prime geodesic theorem; geodesics per unit length; Thorin densities at three basepoints
g = np.load("geodesics.npz"); N = g["norms"]; m = g["mults"]; L = np.log(N)
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.4))
xs = np.geomspace(30, N.max(), 300); cum = np.cumsum(m)
ratio = [cum[np.searchsorted(N, x, side="right")-1]/float(li(x)) for x in xs]
ax[0].semilogx(xs, ratio, color=INK, lw=0.9); ax[0].axhline(1, color=RUST, lw=0.9)
ax[0].set_xlabel("$x$"); ax[0].set_ylabel(r"$\pi_\Gamma(x)/\mathrm{li}(x)$"); ax[0].set_title("prime geodesic theorem", fontsize=8); ax[0].set_ylim(0.7, 1.1)
edges = np.arange(1.5, np.log(1e6) + 0.01, 0.5); hh, _ = np.histogram(L, bins=edges, weights=m)
mids = (edges[1:] + edges[:-1])/2
ax[1].semilogy(mids, hh/0.5, "o", color=INK, ms=3.5, label="geodesics per unit length"); ax[1].semilogy(mids, np.exp(mids)/mids, color=RUST, lw=1.1, label="$e^\\ell/\\ell$")
ax[1].set_xlabel("length $\\ell$"); ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("in the length variable", fontsize=8)
def U(z, alpha):
    r = np.sqrt(z); tot = 0.0
    for k in range(1, 40): tot += np.sum(m*L*N**(-k*alpha)*2*np.sin(k*L*r/2)**2)
    return tot/(2*np.pi*r)
zz = np.linspace(0.05, 60, 300)
for alpha, col in [(1.5, GOLD), (2, TEAL), (3, INK)]:
    ax[2].plot(zz, [U(z, alpha) for z in zz], color=col, lw=1.1, label="$\\alpha=%g$" % alpha)
ax[2].set_xlabel("$z$"); ax[2].set_ylabel("$U(z)$"); ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("Thorin density of $Z(s+1)/Z(s)$", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/maass_geodesics.pdf")

# maass_zeta: error term, periodogram, identity term and geodesic sum
d7 = np.load("m7.npz"); u = d7["u"]; f = d7["f"]; rs = d7["rs"]; per = d7["per"]; rj = d7["rj"]
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.4))
ax[0].plot(u, f - f.mean(), color=INK, lw=0.5); ax[0].set_xlabel("$u=\\log x$"); ax[0].set_ylabel(r"$(\psi_\Gamma(x)-x)/\sqrt{x}$"); ax[0].set_title("error term, centred", fontsize=8)
ax[1].plot(rs, per, color=INK, lw=0.9)
for r in rj[:6]: ax[1].axvline(r, color=RUST, lw=0.7, ls=":")
ax[1].set_xlabel("$r$"); ax[1].set_ylabel("periodogram"); ax[1].set_title("peaks at the $r_j$", fontsize=8); ax[1].set_xlim(2, 20)
ts = np.geomspace(0.05, 3, 80)
ident = np.array([quad(lambda r: np.exp(-t*(0.25 + r*r))*r*np.tanh(np.pi*r), 0, np.inf)[0]*2/12 for t in ts])
geo = []
for t in ts:
    s = 0.0
    for k in range(1, 30):
        s += np.sum(m*L/(2*np.sinh(k*L/2))*np.exp(-t/4)*np.exp(-(k*L)**2/(4*t)))/np.sqrt(4*np.pi*t)
    geo.append(s)
ax[2].loglog(ts, ident, color=RUST, lw=1.2, label="identity term"); ax[2].loglog(ts, geo, color=TEAL, lw=1.2, label="geodesic sum")
ax[2].loglog(ts, 1/(12*ts), color=GOLD, lw=0.9, ls="--", label="$1/(12t)$")
ax[2].set_xlabel("$t$"); ax[2].legend(fontsize=6.5, frameon=False); ax[2].set_title("terms of the heat trace formula", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/maass_zeta.pdf")
json.dump(dict(ts=ts.tolist(), ident=ident.tolist(), geo=[float(v) for v in geo]), open("zeta_terms.json", "w"))

# maass_vertical: vertical Sato Tate; moments m = 2, 4 with running averages
xs = np.load("vertical_p2.npy"); p = 2
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
ax[0].hist(xs, bins=40, density=True, color=INK, alpha=0.8, edgecolor="white", linewidth=0.3, label="eigenforms, $k\\leq400$")
gg = np.linspace(-2, 2, 400)
mu = (p+1)/np.pi*np.sqrt(np.clip(1-gg*gg/4, 0, None))/((np.sqrt(p)+1/np.sqrt(p))**2-gg*gg)
ax[0].plot(gg, mu, color=RUST, lw=1.3, label="Plancherel $\\mu_2$"); ax[0].plot(gg, np.sqrt(np.clip(1-gg*gg/4, 0, None))/np.pi, color=GOLD, lw=1.1, ls="--", label="Sato Tate")
ax[0].set_xlabel(r"$a_2(f)/2^{(k-1)/2}$"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=6, loc="upper center", framealpha=0.9); ax[0].set_title("vertical Sato Tate at $p=2$", fontsize=8); ax[0].set_ylim(0, 0.42)
rows = np.array(json.load(open("m6.json"))); k = rows[:, 0]
for j, (col, lab, lim) in enumerate([(INK, "$m=2$", 0.5), (TEAL, "$m=4$", 0.25)]):
    v = rows[:, j+1]
    ax[1].plot(k, v, ".", color=col, ms=3, alpha=0.6); ax[1].plot(k, np.cumsum(v)/np.arange(1, len(v)+1), color=col, lw=1.3, label=lab + ", running average")
    ax[1].axhline(lim, color=col, lw=0.7, ls=":")
ax[1].set_xlabel("weight $k$"); ax[1].set_ylabel(r"$\frac{1}{d_k}\sum_f U_m(\cos\theta_2(f))$"); ax[1].set_ylim(-1.5, 1.6); ax[1].legend(fontsize=6.5, frameon=False, loc="upper right"); ax[1].set_title("Eichler Selberg moments $\\to 2^{-m/2}$", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/maass_vertical.pdf")

# maass_spectrum: Weyl's law and the first 100 spectral parameters
R = np.array(lmfdb_r.R); par = np.array(list(lmfdb_r.parity))
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
T = np.linspace(0, 46, 500); Ncount = np.searchsorted(R, T, side="right")
ax[0].step(T, Ncount, where="post", color=INK, lw=1.1, label="$N(T)$, LMFDB")
ax[0].plot(T, T**2/12, color=GOLD, lw=1, ls="--", label="$T^2/12$")
c = (len(R) - 46**2/12 + 2*46/np.pi*np.log(46))/46
ax[0].plot(T, T**2/12 - 2*T/np.pi*np.log(np.maximum(T, 1e-9)) + c*T, color=RUST, lw=1, label="$T^2/12-\\frac{2T}{\\pi}\\log T+cT$, $c=%.2f$" % c)
ax[0].set_xlabel("$T$"); ax[0].set_ylabel("count of $r_j\\leq T$"); ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title("Weyl's law for $\\mathrm{SL}_2(\\mathbb{Z})$", fontsize=8)
ev = R[par == "e"]; od = R[par == "o"]
ax[1].vlines(od, 0, 1, color=INK, lw=0.9, label="odd (%d)" % len(od)); ax[1].vlines(ev, 1.15, 2.15, color=RUST, lw=0.9, label="even (%d)" % len(ev))
ax[1].set_yticks([]); ax[1].set_xlabel("$r_j$"); ax[1].set_xlim(0, 47); ax[1].legend(fontsize=7, frameon=False, loc="upper left"); ax[1].set_title("the first 100 spectral parameters", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/maass_spectrum.pdf")

# maass_heat: McKean's heat kernel
def pt(rho, t):
    fn = lambda s: s*np.exp(-s*s/(4*t))/np.sqrt(np.cosh(s) - np.cosh(rho))
    v = quad(fn, rho, rho + 60*np.sqrt(t) + 5, limit=200)[0]
    return np.sqrt(2)*np.exp(-t/4)/(4*np.pi*t)**1.5*v
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
rhos = np.linspace(0.01, 6, 120)
for t, col in [(0.1, INK), (0.3, TEAL), (1, GOLD), (3, RUST)]:
    ax[0].semilogy(rhos, [pt(r, t) for r in rhos], color=col, lw=1.2, label="$t=%g$" % t)
ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"$p_t(\rho)$"); ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("McKean's heat kernel on $\\mathbb{H}$", fontsize=8)
tt = np.geomspace(0.02, 30, 60); p0 = [pt(1e-4, t) for t in tt]
spec = [quad(lambda r: np.exp(-t*(0.25+r*r))*r*np.tanh(np.pi*r), 0, np.inf)[0]/(2*np.pi) for t in tt]
ax[1].loglog(tt, p0, color=INK, lw=1.6, label="$p_t(0)$ from \\eqref{eq:msmckean}" if False else "$p_t(0)$, McKean's integral")
ax[1].loglog(tt, spec, color=TEAL, lw=0.9, ls="-.", label="spectral integral")
ax[1].loglog(tt, 1/(4*np.pi*tt), color=GOLD, ls="--", lw=1, label="$1/(4\\pi t)$")
ax[1].loglog(tt, np.sqrt(np.pi)*np.exp(-tt/4)/(8*tt**1.5), color=RUST, ls=":", lw=1, label="$\\sqrt{\\pi}e^{-t/4}/(8t^{3/2})$")
ax[1].set_xlabel("$t$"); ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title("return density", fontsize=8)
plt.tight_layout(); plt.savefig("../book/figs/maass_heat.pdf")
print("ok", max(abs(np.array(p0)-np.array(spec))/np.array(spec)))
