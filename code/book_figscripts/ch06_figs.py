# Figures for Chapter 6: Katz Sarnak one level densities, and CUE spacings vs Wigner surmise.
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
C = {"ink": "#1F3B63", "rust": "#9C3D1F", "teal": "#1B7F79", "gold": "#C99A2E", "plum": "#6B3F7A"}
D = "../work/figs/"

# one level densities
x = np.linspace(-3, 3, 1201)
sinc = np.sinc(2*x)          # sin(2 pi x)/(2 pi x)
W = {"U": np.ones_like(x),
     "USp": 1 - sinc,
     "SO(even)": 1 + sinc,
     "SO(odd)": 1 - sinc,
     "O": 1 + 0.0*x}
fig, ax = plt.subplots(figsize=(4.4, 2.3))
ax.plot(x, W["U"], color="k", lw=1.0, ls=":", label=r"$\mathrm{U}$")
ax.plot(x, W["USp"], color=C["ink"], lw=1.3, label=r"$\mathrm{USp}$")
ax.plot(x, W["SO(even)"], color=C["rust"], lw=1.3, label=r"$\mathrm{SO(even)}$")
# SO(odd) and O include a delta at 0: draw the continuous part and an arrow
ax.plot(x, W["SO(odd)"], color=C["teal"], lw=1.0, ls="--", label=r"$\mathrm{SO(odd)}$, plus $\delta_0$")
ax.plot(x, 1 + 0.5*0*x, color=C["gold"], lw=0.0)
ax.annotate("", xy=(0, 2.25), xytext=(0, 0.0), arrowprops=dict(arrowstyle="->", color=C["teal"], lw=1.0))
ax.set_xlabel(r"$x$ (zeros in units of mean spacing)"); ax.set_ylabel(r"$W(G)(x)$")
ax.set_ylim(0, 2.3); ax.legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(D + "ch06_onelevel.pdf"); plt.close(fig)

# CUE spacings
rng = np.random.default_rng(2026)
def haar_unitary(n):
    z = (rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))/np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r); return q * (d/abs(d))
N, M = 200, 400
sp = []; tr1 = []
for _ in range(M):
    U = haar_unitary(N)
    th = np.sort(np.angle(np.linalg.eigvals(U)))
    s = np.diff(np.concatenate([th, [th[0] + 2*np.pi]]))*N/(2*np.pi)
    sp.append(s)
sp = np.concatenate(sp)
print("spacings", sp.size, "mean", sp.mean(), "var", sp.var())
s = np.linspace(0, 3.5, 400)
wig2 = 32/np.pi**2*s**2*np.exp(-4*s**2/np.pi)
fig, ax = plt.subplots(figsize=(4.4, 2.2))
ax.hist(sp, bins=70, range=(0, 3.5), density=True, color=C["ink"], alpha=0.35, edgecolor="none",
        label=r"$\mathrm{CUE}$, $N=200$, $400$ samples")
ax.plot(s, wig2, color=C["rust"], lw=1.4, label=r"Wigner surmise $\frac{32}{\pi^2}s^2e^{-4s^2/\pi}$")
ax.plot(s, np.exp(-s), color="k", lw=0.8, ls=":", label=r"Poisson $e^{-s}$")
ax.set_xlabel(r"normalised spacing $s$"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(D + "ch06_spacing.pdf"); plt.close(fig)

# Diaconis Shahshahani check: traces of powers for N = 200 (used in text)
N, M = 50, 4000
T = np.zeros((M, 3), complex)
for m in range(M):
    ev = np.linalg.eigvals(haar_unitary(N))
    T[m] = [np.sum(ev**j) for j in (1, 2, 3)]
for j in range(3):
    print("j", j+1, "E|tr U^j|^2 / j =", np.mean(abs(T[:, j])**2)/(j+1), "E|.|^4/(2 j^2)", np.mean(abs(T[:, j])**4)/(2*(j+1)**2))
print("corr |tr U|^2 vs |tr U^2|^2", np.corrcoef(abs(T[:, 0])**2, abs(T[:, 1])**2)[0, 1])
