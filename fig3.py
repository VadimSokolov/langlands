import numpy as np, json, math, cypari2, mpmath as mp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LogNorm, TwoSlopeNorm
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 220})
D = "/home/claude/langlands/book/figs/"
C = {"ink": "#1F3B63", "rust": "#9C3D1F", "teal": "#1B7F79", "gold": "#C99A2E", "plum": "#6B3F7A", "sky": "#8FB3DE"}
pari = cypari2.Pari(); pari.allocatemem(3 * 10**9); pari.set_real_precision(19)

def domain_rgb(Z):
    H = (np.angle(Z) / (2 * np.pi)) % 1.0
    lm = np.log(np.abs(Z) + 1e-300)
    bands = 0.5 + 0.5 * ((lm / math.log(2)) % 1.0)
    V = np.clip(0.55 + 0.45 * bands, 0, 1)
    S = np.full_like(H, 0.85)
    return hsv_to_rgb(np.dstack([H, S, V]))

# 1. domain colouring of Lambda(6+z)/Lambda(6) for Delta, z = x + i t
pari("L=lfunmf(mfinit([1,12]),mfDelta()); Li=lfuninit(L,[6,12.5,40]);")
L6 = float(pari("lfunlambda(Li,6)"))
xs = np.linspace(-3, 3, 181); ts = np.linspace(0, 32, 481)
Z = np.zeros((len(ts), len(xs)), complex)
for i, t in enumerate(ts):
    for j, x in enumerate(xs):
        Z[i, j] = complex(pari(f"lfunlambda(Li,{6 + x}+I*{t})")) / L6
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.imshow(domain_rgb(Z), origin="lower", extent=[xs[0], xs[-1], ts[0], ts[-1]], aspect="auto")
ax.axvline(0, color="white", lw=0.6, ls=":")
ax.set_xlabel(r"$\mathrm{Re}\,z$"); ax.set_ylabel(r"$\mathrm{Im}\,z$"); ax.set_title(r"$\Lambda(6+z,\Delta)/\Lambda(6,\Delta)$", fontsize=9)
plt.savefig(D + "domain_delta.pdf"); plt.savefig(D + "domain_delta.png"); plt.close()
np.save("/home/claude/langlands/domain_delta.npy", Z)

# 2. domain colouring of the Thorin reciprocal 1/F(s) = Lambda(6)/Lambda(6+sqrt s) in the s plane
sr = np.linspace(-700, 120, 411); si = np.linspace(-120, 120, 241)
W = np.zeros((len(si), len(sr)), complex)
for i, b in enumerate(si):
    for j, a in enumerate(sr):
        s = complex(a, b); z = np.sqrt(s)
        if z.real < 0: z = -z
        if abs(z.imag) > 39: W[i, j] = np.nan; continue
        W[i, j] = L6 / complex(pari(f"lfunlambda(Li,{6 + z.real}+I*{z.imag})"))
Wf = np.where(np.isnan(W), 1.0, W)
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.imshow(domain_rgb(Wf), origin="lower", extent=[sr[0], sr[-1], si[0], si[-1]], aspect="auto")
zD = [float(z) for z in pari("lfunzeros(L,30)") if float(z) > 1e-6]
ax.plot([-g * g for g in zD if g * g < 700], [0] * len([g for g in zD if g * g < 700]), "o", ms=3, mfc="none", mec="white", mew=0.8)
ax.set_xlabel(r"$\mathrm{Re}\,s$"); ax.set_ylabel(r"$\mathrm{Im}\,s$"); ax.set_title(r"$\Lambda(6)/\Lambda(6+\sqrt{s})$: poles only at $-\gamma_k^2$", fontsize=9)
plt.savefig(D + "domain_thorin.pdf"); plt.savefig(D + "domain_thorin.png"); plt.close()
print("domain ok", flush=True)

# 3. modular tessellation coloured by hyperbolic area (left Haar measure of the Borel group)
fig, ax = plt.subplots(figsize=(4.6, 2.9)); ax.set_facecolor("#0B1026")
cmap = plt.get_cmap("plasma")
def arc(p, q, **kw):
    if np.isinf(q):
        ax.plot([p, p], [0, 3], **kw); return
    c = (p + q) / 2; r = abs(q - p) / 2; th = np.linspace(0, np.pi, 200)
    ax.plot(c + r * np.cos(th), r * np.sin(th), **kw)
from fractions import Fraction
fr = set()
for den in range(1, 26):
    for num in range(-2 * den, 2 * den + 1):
        fr.add(Fraction(num, den))
fr = sorted(fr)
for a in fr:
    for b in fr:
        if a < b and abs(a.numerator * b.denominator - a.denominator * b.numerator) == 1:
            d = max(a.denominator, b.denominator)
            arc(float(a), float(b), color=cmap(min(1, math.log(d) / math.log(26))), lw=max(0.15, 1.2 / d))
for k in range(-2, 3): arc(k, np.inf, color=cmap(0.0), lw=1.2)
th = np.linspace(np.pi / 3, 2 * np.pi / 3, 100)
ax.fill_between(np.cos(th), np.sin(th), 3, color="#F4E3B1", alpha=0.35, lw=0)
ax.set_xlim(-2, 2); ax.set_ylim(0, 2.2); ax.set_aspect("equal"); ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([])
ax.set_title(r"Farey tessellation of $\mathbb{H}$; shaded: fundamental domain of area $\pi/3$", fontsize=8)
plt.savefig(D + "modular_tessellation.pdf"); plt.close()

# 4. Heegner points in the fundamental domain
P = np.load("/home/claude/langlands/heegner.npy")
fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
hb = ax[0].hexbin(P[:, 0], P[:, 1], gridsize=38, extent=(-0.5, 0.5, 0.85, 3.0), cmap="viridis", mincnt=1, bins="log", linewidths=0)
th = np.linspace(np.pi / 3, 2 * np.pi / 3, 100); ax[0].plot(np.cos(th), np.sin(th), color="white", lw=1)
ax[0].set_xlim(-0.5, 0.5); ax[0].set_ylim(0.85, 3.0); ax[0].set_title("Heegner points, $D\\approx -10^6$", fontsize=8); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")
tt = np.logspace(0, 1.5, 60); emp = [np.mean(P[:, 1] > t) for t in tt]
ax[1].loglog(tt, emp, color=C["teal"], lw=2, label="28251 points"); ax[1].loglog(tt, 3 / (np.pi * tt), "--", color=C["rust"], lw=1.2, label=r"$3/(\pi t)$")
ax[1].set_xlabel("$t$"); ax[1].set_title(r"$\Pr(\mathrm{Im}\,z>t)$", fontsize=8); ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(D + "haar_heegner.pdf"); plt.close()

# 5. right versus left Haar: posterior for one data set and coverage
us, ph = np.load("/home/claude/langlands/phi.npy")
lf = lambda z: np.log(np.maximum(np.interp(z, us, ph, left=0, right=0), 1e-300))
rng = np.random.default_rng(12); cdf = np.cumsum(ph); cdf /= cdf[-1]
yv = 1.3 + 0.7 * np.interp(rng.uniform(size=5), cdf, us)
mu = np.linspace(0.3, 2.3, 300); lsg = np.linspace(-3, 1.2, 300); MU, LS = np.meshgrid(mu, lsg); SG = np.exp(LS)
ll = sum(lf((y - MU) / SG) for y in yv) - 5 * LS
fig, ax = plt.subplots(1, 3, figsize=(6.0, 2.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for a, pw, title, cm in [(ax[0], 1, r"right Haar $d\mu\,d\sigma/\sigma$", "magma"), (ax[1], 2, r"left Haar $d\mu\,d\sigma/\sigma^2$", "viridis")]:
    lp = ll - (pw - 1) * LS; w = np.exp(lp - lp.max())
    a.contourf(MU, LS, w, levels=14, cmap=cm); a.plot(1.3, math.log(0.7), "w+", ms=8, mew=1.5)
    a.set_title(title, fontsize=8); a.set_xlabel(r"$\mu$")
ax[0].set_ylabel(r"$\log\sigma$")
o = json.load(open("/home/claude/langlands/stage8.json"))["coverage"]
xs_ = np.arange(2); wdt = 0.36; se = o["mc_se"]
ax[2].bar(xs_ - wdt / 2, [o["right"]["mu"], o["right"]["sigma"]], wdt, color="#B5446E", yerr=2 * se, label="right")
ax[2].bar(xs_ + wdt / 2, [o["left"]["mu"], o["left"]["sigma"]], wdt, color="#3B8E8A", yerr=2 * se, label="left")
ax[2].axhline(0.9, color="k", lw=0.8, ls="--"); ax[2].set_ylim(0.8, 0.93); ax[2].set_xticks(xs_); ax[2].set_xticklabels([r"$\mu$", r"$\sigma$"])
ax[2].set_title("coverage of 90% sets", fontsize=8); ax[2].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(D + "haar_posterior.pdf"); plt.close()
print("haar ok", flush=True)

# 6. Patie sign heat maps for xi and 11a
def patie_table(ctor, zeta, NT=40, K=30):
    pari.set_real_precision(350); pari(f"L2={ctor};"); c = float(pari("lfunparams(L2)")[1]) / 2
    ser = pari(f"lfunlambda(L2, {c} + x + O(x^{2*NT+2}))" + (" * (x^2 - 1/4)" if zeta else ""))
    mp.mp.dps = 350
    a = [mp.mpf(str(pari.polcoef(ser, 2 * n)).replace(" E", "E")) for n in range(NT + 1)]
    psi = [a[n - 1] / a[n] for n in range(1, NT + 1)]
    cur = [psi[i + 2] - 2 * psi[i + 1] + psi[i] for i in range(len(psi) - 2)]
    M = np.full((K + 1, len(cur)), np.nan)
    for k in range(K + 1):
        for i, v in enumerate(cur): M[k, i] = float((-1) ** k * v)
        cur = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
    return M
fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.6), sharey=True)
for a, (name, ctor, z) in zip(ax, [(r"$\xi$", "lfuncreate(1)", True), ("11a1", "lfuncreate(ellinit([0,-1,1,-10,-20]))", False)]):
    M = patie_table(ctor, z)
    S = np.sign(M) * np.log10(1 + np.abs(M) * 1e14)
    im = a.imshow(S, origin="lower", aspect="auto", cmap="RdYlBu", norm=TwoSlopeNorm(0, vmin=-np.nanmax(np.abs(S)), vmax=np.nanmax(np.abs(S))))
    a.set_title(name, fontsize=9); a.set_xlabel("$n$")
ax[0].set_ylabel("order $k$")
cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("signed size of $(-1)^k\\Delta^k\\delta_n$", fontsize=7)
plt.savefig(D + "patie_heatmap.pdf"); plt.close()
pari.set_real_precision(19)
print("patie ok", flush=True)

# 7. heat trace W(x) and derivatives from the zeros, with prime side points
pari("L=lfunmf(mfinit([1,12]),mfDelta());")
zz = np.array([float(z) for z in pari("lfunzeros(L,300)") if float(z) > 1e-6])
xg = np.logspace(-2.5, 0, 200)
fig, ax = plt.subplots(figsize=(4.4, 2.6))
cols = ["#1F3B63", "#1B7F79", "#C99A2E", "#9C3D1F"]
s6 = {0.02: [0.2061515276408161, 20.450633094275233, 2391.2851645202795, 359738.32402760646],
      0.05: [0.014290332953786858, 1.222317455581476, 105.30019215184348, 9216.990197017574],
      0.1: [0.00020241135580126968, 0.01721597898460342, 1.4643417218750119, 124.56174145637718]}
for n, c in enumerate(cols):
    ax.loglog(xg, [np.sum(zz ** (2 * n) * np.exp(-x * zz ** 2)) for x in xg], color=c, lw=1.4, label=f"$(-1)^{n}W^{{({n})}}$")
    ax.loglog(list(s6), [v[n] for v in s6.values()], "o", ms=5, mfc="white", mec=c, mew=1.3)
ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=7, ncol=2); ax.set_title("heat trace of $\\Delta$: curves from zeros, circles from primes", fontsize=8)
ax.set_ylim(1e-8, 1e9)
plt.savefig(D + "heattrace_primes.pdf"); plt.close()

# 8. zeta first passage law
r7 = json.load(open("/home/claude/langlands/stage7.json"))["zeta_first_passage"]["rows"]
m = [r[0] for r in r7]; sim = [r[1] for r in r7]; fm = [r[2] for r in r7]
fig, ax = plt.subplots(figsize=(4.0, 2.3))
xx = np.arange(len(m)); ax.bar(xx - 0.2, sim, 0.4, color="#8FB3DE", label="simulation"); ax.bar(xx + 0.2, fm, 0.4, color="#9C3D1F", label="ballot formula")
ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(m); ax.set_xlabel("$m$"); ax.set_ylabel(r"$\Pr(M_x=m)$"); ax.legend(frameon=False, fontsize=7)
plt.savefig(D + "zeta_passage.pdf"); plt.close()

# 9. Frobenius angle roses for E1 (no CM) and E2 (CM)
def tr(c, X):
    pari(f"E=ellinit({c}); P=primes([3,{X}]); V=vector(#P,i,ellap(E,P[i]));")
    return np.array([int(z) for z in pari("P")]), np.array([int(z) for z in pari("V")])
P1, V1 = tr("[0,0,0,1,1]", 400000); P2, V2 = tr("[0,0,0,-1,0]", 400000)
th1 = np.arccos(np.clip(V1 / (2 * np.sqrt(P1)), -1, 1)); th2 = np.arccos(np.clip(V2 / (2 * np.sqrt(P2)), -1, 1))
fig = plt.figure(figsize=(5.0, 2.6))
for k, (th, title, col) in enumerate([(th1, r"$y^2=x^3+x+1$ (Sato Tate)", "#3B5BA9"), (th2, r"$y^2=x^3-x$ (CM)", "#C2502B")]):
    a = fig.add_subplot(1, 2, k + 1, projection="polar")
    full = np.concatenate([th, -th]); bins = np.linspace(-np.pi, np.pi, 73)
    h, e = np.histogram(full, bins=bins, density=True)
    a.bar((e[:-1] + e[1:]) / 2, h, width=np.diff(e), color=plt.get_cmap("twilight")((e[:-1] + np.pi) / (2 * np.pi)), edgecolor="none", alpha=0.95)
    g = np.linspace(-np.pi, np.pi, 400)
    if k == 0: a.plot(g, np.sin(g) ** 2 / np.pi, color="k", lw=1)
    a.set_title(title, fontsize=8); a.set_yticklabels([]); a.set_xticklabels([])
plt.savefig(D + "frobenius_roses.pdf"); plt.close()
print("all ok", flush=True)
