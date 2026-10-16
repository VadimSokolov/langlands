import numpy as np, json, math, cypari2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 220})
D = "../work/figs/"
pari = cypari2.Pari(); pari.allocatemem(2 * 10**9); pari.set_real_precision(19)

# 1. enhanced phase portrait of Lambda(6+z) normalised column by column
Z = np.load("../work/domain_delta.npy")
xs = np.linspace(-3, 3, 181); ts = np.linspace(0, 32, 481)
Zn = Z / np.abs(Z).max(axis=1, keepdims=True)
H = (np.angle(Zn) / (2 * np.pi)) % 1.0
lm = np.log(np.abs(Zn) + 1e-300)
V = 0.62 + 0.38 * ((lm / math.log(2)) % 1.0)
V *= np.clip((np.abs(Zn) / (np.abs(Zn) + 0.03)) ** 0.35, 0, 1)
rgb = hsv_to_rgb(np.dstack([H, np.full_like(H, 0.9), V]))
pari("L=lfunmf(mfinit([1,12]),mfDelta());")
zD = [float(z) for z in pari("lfunzeros(L,32)") if float(z) > 1e-6]
fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.imshow(rgb, origin="lower", extent=[xs[0], xs[-1], ts[0], ts[-1]], aspect="auto")
ax.plot([0] * len(zD), zD, "o", ms=5, mfc="none", mec="white", mew=1.0)
ax.set_xlabel(r"$\mathrm{Re}\,z$"); ax.set_ylabel(r"$\mathrm{Im}\,z$")
ax.set_title(r"phase portrait of $\Lambda(6+z,\Delta)$; zeros (circles) on $\mathrm{Re}\,z=0$", fontsize=8)
plt.savefig(D + "domain_delta.pdf"); plt.close()

# 2. SL2(Z) tessellation by reduction depth
nx, ny = 1400, 760
x = np.linspace(-1.6, 1.6, nx); y = np.linspace(0.004, 1.75, ny)
X, Y = np.meshgrid(x, y); z = X + 1j * Y
depth = np.zeros(z.shape); parity = np.zeros(z.shape)
for it in range(80):
    k = np.round(z.real); z = z - k; parity += k
    inside = np.abs(z) < 1 - 1e-12
    if not inside.any(): break
    z = np.where(inside, -1 / z, z); depth += inside
img = plt.get_cmap("twilight_shifted")((np.mod(depth * 0.137 + 0.11 * np.mod(parity, 7), 1.0)))
shade = np.clip(0.55 + 0.45 * np.tanh(Y * 3), 0, 1)[..., None]
img[..., :3] *= shade
fig, ax = plt.subplots(figsize=(5.2, 2.9))
ax.imshow(img, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="equal")
th = np.linspace(np.pi / 3, 2 * np.pi / 3, 100)
ax.plot(np.cos(th), np.sin(th), color="white", lw=1.2); ax.plot([-0.5, -0.5], [math.sqrt(3) / 2, 1.75], color="white", lw=1.2); ax.plot([0.5, 0.5], [math.sqrt(3) / 2, 1.75], color="white", lw=1.2)
ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([0, 0.5, 1, 1.5])
ax.set_title(r"$\mathrm{SL}_2(\mathbb{Z})$ tiles of $\mathbb{H}$, each of hyperbolic area $\pi/3$; white: the fundamental domain", fontsize=7.5)
plt.savefig(D + "modular_tessellation.pdf"); plt.close()

# 3. zoomed posteriors under right and left Haar
us, ph = np.load("../work/phi.npy")
lf = lambda q: np.log(np.maximum(np.interp(q, us, ph, left=0, right=0), 1e-300))
rng = np.random.default_rng(12); cdf = np.cumsum(ph); cdf /= cdf[-1]
yv = 1.3 + 0.7 * np.interp(rng.uniform(size=5), cdf, us)
mu = np.linspace(0.6, 2.0, 400); lsg = np.linspace(-2.4, 0.9, 400); MU, LS = np.meshgrid(mu, lsg)
ll = sum(lf((v - MU) / np.exp(LS)) for v in yv) - 5 * LS
o = json.load(open("../work/stage8.json"))["coverage"]
fig, ax = plt.subplots(1, 3, figsize=(6.2, 2.3), gridspec_kw={"width_ratios": [1, 1, 0.85]})
for a, pw, title, cm in [(ax[0], 1, r"right Haar $d\mu\,d\sigma/\sigma$", "magma"), (ax[1], 2, r"left Haar $d\mu\,d\sigma/\sigma^2$", "viridis")]:
    lp = ll - (pw - 1) * LS; w = np.exp(lp - lp.max())
    a.contourf(MU, LS, w, levels=16, cmap=cm); a.contour(MU, LS, w / w.max(), levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="white", linewidths=0.4)
    a.plot(1.3, math.log(0.7), "+", color="cyan", ms=9, mew=1.6); 
    a.set_title(title, fontsize=8); a.set_xlabel(r"$\mu$")
ax[0].set_ylabel(r"$\log\sigma$")
xx = np.arange(2); wd = 0.36; se = o["mc_se"]
ax[2].bar(xx - wd / 2, [o["right"]["mu"], o["right"]["sigma"]], wd, color="#B5446E", yerr=2 * se, capsize=2, label="right Haar")
ax[2].bar(xx + wd / 2, [o["left"]["mu"], o["left"]["sigma"]], wd, color="#3B8E8A", yerr=2 * se, capsize=2, label="left Haar")
ax[2].axhline(0.9, color="k", lw=0.8, ls="--"); ax[2].set_ylim(0.82, 0.93); ax[2].set_xticks(xx); ax[2].set_xticklabels([r"$\mu$", r"$\sigma$"])
ax[2].set_title("coverage of 90% sets, 4000 runs", fontsize=8); ax[2].legend(frameon=False, fontsize=6.5, loc="lower right")
plt.tight_layout(); plt.savefig(D + "haar_posterior.pdf"); plt.close()

# 4. Frobenius roses with the CM atom handled separately
def tr(c, X):
    pari(f"E=ellinit({c}); P=primes([3,{X}]); V=vector(#P,i,ellap(E,P[i]));")
    return np.array([int(q) for q in pari("P")]), np.array([int(q) for q in pari("V")])
P1, V1 = tr("[0,0,0,1,1]", 400000); P2, V2 = tr("[0,0,0,-1,0]", 400000)
th1 = np.arccos(np.clip(V1 / (2 * np.sqrt(P1)), -1, 1)); m2 = V2 != 0; th2 = np.arccos(np.clip(V2[m2] / (2 * np.sqrt(P2[m2])), -1, 1))
fig = plt.figure(figsize=(5.2, 2.7)); bins = np.linspace(-np.pi, np.pi, 73); g = np.linspace(-np.pi, np.pi, 400)
for k, (th, title, curve) in enumerate([(th1, r"$y^2=x^3+x+1$: Sato Tate", np.sin(g) ** 2 / np.pi), (th2, r"$y^2=x^3-x$: CM, split primes", np.full_like(g, 1 / (2 * np.pi)))]):
    a = fig.add_subplot(1, 2, k + 1, projection="polar")
    full = np.concatenate([th, -th]); h, e = np.histogram(full, bins=bins, density=True)
    a.bar((e[:-1] + e[1:]) / 2, h, width=np.diff(e), color=plt.get_cmap("twilight")((e[:-1] + np.pi) / (2 * np.pi)), edgecolor="none")
    a.plot(g, curve, color="k", lw=1); a.set_yticklabels([]); a.set_xticklabels([]); a.set_title(title, fontsize=8)
plt.savefig(D + "frobenius_roses.pdf"); plt.close()
print("ok")
