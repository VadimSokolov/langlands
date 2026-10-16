import cypari2, numpy as np, json, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import kolmogorov
pari = cypari2.Pari(); pari.allocatemem(3 * 10**9)
D = "../work/figs/"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})
C = {"ink": "#1F3B63", "rust": "#9C3D1F", "teal": "#1B7F79", "gold": "#C99A2E", "plum": "#6B3F7A", "sky": "#8FB3DE", "sand": "#E2C9A6"}
W1, H1 = 4.4, 2.6
def arr(expr):
    t = str(pari(f"Vecsmall({expr})")); return np.array(t[t.index("[")+1:t.rindex("]")].split(","), dtype=np.int64)
def tr(c, X):
    pari(f"E=ellinit({c}); P=primes([3,{X}]); V=vector(#P,i,ellap(E,P[i]));"); return arr("P"), arr("V")
xs = np.linspace(-2, 2, 801); st = np.sqrt(np.clip(4 - xs**2, 0, None)) / (2*np.pi)
arc = 1/(np.pi*np.sqrt(np.clip(4 - xs**2, 1e-9, None)))
P1, V1 = tr("[0,0,0,1,1]", 10**7); g = P1 != 31; x1 = V1[g]/np.sqrt(P1[g])
P2, V2 = tr("[0,0,0,-1,0]", 10**7); x2 = V2/np.sqrt(P2)

fig, ax = plt.subplots(1, 2, figsize=(W1, 2.2))
ax[0].hist(x1, bins=80, density=True, color=C["sky"], edgecolor="none"); ax[0].plot(xs, st, color=C["ink"], lw=1.4)
ax[0].set_title(r"$y^2=x^3+x+1$", fontsize=9); ax[0].set_xlabel(r"$a_p/\sqrt{p}$")
nz = x2[V2 != 0]; edges = np.linspace(-2, 2, 81)  # mass len(nz)/len(x2), about 1/2, to match .5*arc (fixed 2026-10-09)
ax[1].hist(nz, bins=edges, weights=np.full(len(nz), 1/(len(x2)*(edges[1]-edges[0]))), color=C["sand"], edgecolor="none")
ax[1].plot(xs, .5*arc, color=C["rust"], lw=1.4); ax[1].set_ylim(0, 1.0); ax[1].set_title(r"$y^2=x^3-x$ (CM)", fontsize=9)
ax[1].set_xlabel(r"$a_p/\sqrt{p}$"); plt.tight_layout(); plt.savefig(D+"satotate.pdf"); plt.close()

xt = np.load("xt.npy")
fig, ax = plt.subplots(figsize=(W1*0.75, 2.2))
ax.hist(xt, bins=80, density=True, color="#B9D7C9", edgecolor="none"); ax.plot(xs, st, color=C["teal"], lw=1.4)
ax.set_xlabel(r"$\tau(p)/p^{11/2}$"); plt.savefig(D+"tau.pdf"); plt.close()

# Weyl averages
def Un(n, x):
    t = np.arccos(np.clip(x/2, -1, 1)); s = np.sin(t); return np.sin((n+1)*t)/np.where(s > 1e-12, s, 1e-12)
ns = np.arange(1, 17)
fig, ax = plt.subplots(figsize=(W1, 2.1))
ax.bar(ns-0.2, [np.mean(Un(n, x1)) for n in ns], 0.4, color=C["ink"], label="no CM")
ax.bar(ns+0.2, [np.mean(Un(n, x2)) for n in ns], 0.4, color=C["rust"], label="CM")
ax.set_xlabel(r"$n$"); ax.set_ylabel(r"mean of $U_n(x_p/2)$"); ax.legend(frameon=False, fontsize=8); ax.set_xticks(ns)
plt.savefig(D+"weyl.pdf"); plt.close()

# discrepancy path
def F_st(x):
    t = np.arccos(np.clip(x/2, -1, 1)); return 1 - (t - np.sin(2*t)/2)/np.pi
ks = np.unique(np.logspace(1.5, math.log10(len(x1)), 60).astype(int)); dd = []
for k in ks:
    s = np.sort(x1[:k]); e = np.arange(1, k+1)/k
    dd.append(max(np.max(np.abs(e - F_st(s))), np.max(np.abs(e - 1/k - F_st(s)))))
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.2))
ax[0].loglog(ks, dd, color=C["ink"], lw=1.2); ax[0].loglog(ks, 0.87/np.sqrt(ks), "--", color=C["gold"], lw=1)
ax[0].set_xlabel(r"$n$"); ax[0].set_title(r"$D_n$ and $0.87/\sqrt{n}$", fontsize=9)
cross = np.load("cross.npy"); zz = np.linspace(0.2, 2, 300)
dk = np.gradient(1 - kolmogorov(zz), zz)
ax[1].hist(cross, bins=20, density=True, color=C["sky"], edgecolor="white"); ax[1].plot(zz, dk, color=C["ink"], lw=1.3)
ax[1].set_title(r"$\sqrt{n}\,D_n$ over 200 curves", fontsize=9)
plt.tight_layout(); plt.savefig(D+"discrepancy.pdf"); plt.close()

# rank bias
s1 = json.load(open("stage1.json"))
fig, ax = plt.subplots(figsize=(W1, 2.5))
cols = [C["ink"], C["teal"], C["gold"], C["rust"]]
for (name, v), c in zip(s1["bias"].items(), cols):
    a = np.array(v["curve"]); r = v["rank"]
    ax.semilogx(a[:, 0], a[:, 1], color=c, lw=1.1, label=f"{name}, rank {r}")
    ax.semilogx(a[:, 0], (0.5 - r)*np.log(a[:, 0]), ":", color=c, lw=1)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\sum_{p\leq x} a_p\log p/p$"); ax.legend(frameon=False, fontsize=7)
plt.savefig("/tmp/rb.pdf"); plt.close()

# Lang Trotter counting
z0 = P1[g][V1[g] == 0]
from scipy.integrate import quad
Xs = np.unique(np.logspace(2, 7, 80).astype(int))
pih = np.array([quad(lambda t: 1/(2*math.sqrt(t)*math.log(t)), 2, X, limit=200)[0] for X in Xs])
cnt = np.array([np.sum(z0 <= X) for X in Xs])
fig, ax = plt.subplots(figsize=(W1*0.8, 2.2))
ax.loglog(Xs, np.maximum(cnt, 0.8), drawstyle="steps-post", color=C["plum"], label=r"$\#\{p\leq x: a_p=0\}$")
ax.loglog(Xs, 0.9*pih, "--", color=C["gold"], label=r"$0.9\,\pi_{1/2}(x)$"); ax.set_xlabel(r"$x$"); ax.legend(frameon=False, fontsize=7)
plt.savefig("/tmp/lt.pdf"); plt.close()

# Kesten McKay
def km(q, x): return (q+1)/((math.sqrt(q)+1/math.sqrt(q))**2 - x**2)*np.sqrt(np.clip(4 - x**2, 0, None))/(2*np.pi)
fig, ax = plt.subplots(figsize=(W1, 2.3))
for q, c in zip([2, 3, 5, 11, 47], [C["rust"], C["gold"], C["teal"], C["sky"], C["plum"]]):
    ax.plot(xs, km(q, xs), color=c, lw=1.2, label=f"$p={q}$")
ax.plot(xs, st, "k--", lw=1, label="Sato Tate"); ax.legend(frameon=False, fontsize=7, ncol=2); ax.set_xlabel(r"$x$")
plt.savefig(D+"kestenmckay.pdf"); plt.close()

# random walk return probabilities on trees
def ret(q, T):
    d = np.zeros(T+2); d[0] = 1; r = []
    for t in range(1, T+1):
        nd = np.zeros_like(d); nd[1] += d[0]; nd[2:] += d[1:-1]*q/(q+1); nd[:-1] += d[1:]/(q+1); d = nd; r.append(d[0])
    return np.array(r)
fig, ax = plt.subplots(figsize=(W1*0.8, 2.2))
for q, c in zip([2, 3, 5], [C["rust"], C["gold"], C["teal"]]):
    r = ret(q, 200); t = np.arange(2, 201, 2)
    ax.semilogy(t, r[1::2], color=c, lw=1.2, label=f"$q={q}$"); ax.semilogy(t, (2*math.sqrt(q)/(q+1))**t, ":", color=c)
ax.set_xlabel("time $n$"); ax.set_ylabel(r"$\Pr(S_n=o)$"); ax.legend(frameon=False, fontsize=7)
plt.savefig(D+"treereturn.pdf"); plt.close()

# Delta: kernel, characteristic function, zeros, GGC, M*
us, ph = np.load("phi.npy")
var = np.sum(us**2*ph)*(us[1]-us[0])
fig, ax = plt.subplots(figsize=(W1*0.8, 2.2))
ax.plot(us, ph, color=C["teal"], lw=1.4, label=r"$\Phi_\Delta(u)$")
ax.plot(us, np.exp(-us**2/(2*var))/np.sqrt(2*np.pi*var), "--", color=C["gold"], lw=1, label="normal, same variance")
ax.set_xlim(-1.6, 1.6); ax.legend(frameon=False, fontsize=7); ax.set_xlabel(r"$u$")
plt.savefig(D+"deltakernel.pdf"); plt.close()

pari("mf=mfinit([1,12]); L=lfunmf(mf,mfDelta());")
L6 = float(pari("lfunlambda(L,6)"))
ts = np.linspace(0, 40, 800)
Zt = np.array([float(pari(f"real(lfunlambda(L,6+I*{t}))"))/L6 for t in ts])
zer = [float(t) for t in pari("lfunzeros(L,400)")]
fig, ax = plt.subplots(figsize=(W1, 2.2))
ax.plot(ts, np.sign(Zt)*np.abs(Zt)**0.25, color=C["ink"], lw=1.2)
ax.axhline(0, color="gray", lw=.5); ax.plot([z for z in zer if z < 40], [0]*len([z for z in zer if z < 40]), "o", color=C["rust"], ms=3)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\Lambda(6+it)/\Lambda(6)$, fourth root")
plt.savefig(D+"deltazeros.pdf"); plt.close()

tail = lambda T: (math.log(T/(2*math.pi)) + 1)/(math.pi*T)
ss = np.linspace(0, 400, 120)
lhs = np.array([L6/float(pari(f"lfunlambda(L,{6+math.sqrt(s)})")) for s in ss])
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.2))
for T, c in [(20, C["gold"]), (60, C["teal"]), (400, C["ink"])]:
    zz_ = [z for z in zer if z <= T]
    rhs = np.array([math.exp(-sum(math.log1p(s/z**2) for z in zz_) - s*tail(T)) for s in ss])
    ax[0].semilogy(ss, rhs, color=c, lw=1, label=f"zeros up to {T}")
    ax[1].plot(ss, rhs/lhs - 1, color=c, lw=1)
ax[0].semilogy(ss, lhs, "k:", lw=1.5, label="exact"); ax[0].legend(frameon=False, fontsize=6); ax[0].set_xlabel(r"$s$")
ax[0].set_title(r"$\Lambda(6)/\Lambda(6+\sqrt{s})$", fontsize=9); ax[1].set_title("relative error", fontsize=9); ax[1].set_xlabel(r"$s$")
plt.tight_layout(); plt.savefig(D+"ggc.pdf"); plt.close()

# Thorin measure atoms and Levy density k(x) = sum exp(-gamma^2 x)
fig, ax = plt.subplots(figsize=(W1*0.8, 2.2))
gx = np.logspace(-4, 0, 300); kx = np.array([sum(math.exp(-z*z*x) for z in zer) for x in gx])
ax.loglog(gx, kx, color=C["plum"], lw=1.3); ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$x\,\nu(x)=\sum_k e^{-\gamma_k^2 x}$")
plt.savefig(D+"levydelta.pdf"); plt.close()

# law of M*
yy = np.linspace(0.05, 2.5, 400)
F = np.array([float(pari(f"real(eta(I*{y},1)^24)"))*math.exp(2*math.pi*y) for y in yy])
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.1))
ax[0].plot(yy, F, color=C["rust"], lw=1.3); ax[0].set_title(r"$F(y)=\Pr(M^*\leq y)$", fontsize=9); ax[0].set_xlabel(r"$y$")
ax[1].plot(yy, np.gradient(F, yy), color=C["rust"], lw=1.3); ax[1].set_title("density of $M^*$", fontsize=9); ax[1].set_xlabel(r"$y$")
plt.tight_layout(); plt.savefig(D+"mstar.pdf"); plt.close()

# Bayes factor paths for elliptic curves
def bf(x, z, eps=1e-3):
    lst = np.where(z, np.log(eps), np.log(1-eps) + np.log(np.sqrt(np.clip(4 - x**2, 1e-12, None))/(2*np.pi)))
    lcm = np.where(z, np.log(.5), np.log(.5/(np.pi*np.sqrt(np.clip(4 - x**2, 1e-12, None)))))
    return np.cumsum(np.clip(lcm - lst, -20, 20))
b1 = bf(x1[:2000], V1[g][:2000] == 0); b2 = bf(x2[:2000], V2[:2000] == 0)
fig, ax = plt.subplots(figsize=(W1*0.8, 2.2))
ax.plot(np.arange(1, 301), b1[:300], color=C["ink"], label="no CM"); ax.plot(np.arange(1, 301), b2[:300], color=C["rust"], label="CM")
ax.plot(np.arange(1, 301), -np.arange(1, 301), ":", color=C["ink"], lw=.8); ax.plot(np.arange(1, 301), 3.108*np.arange(1, 301), ":", color=C["rust"], lw=.8)
ax.axhline(0, color="gray", lw=.5); ax.set_xlabel("good primes"); ax.set_ylabel(r"$\log B$"); ax.legend(frameon=False, fontsize=7)
plt.savefig(D+"bayes.pdf"); plt.close()
print("done")
