import cypari2, numpy as np, json, math, mpmath as mp, time
pari = cypari2.Pari(); pari.allocatemem(4 * 10**9); pari.set_real_precision(60)
mp.mp.dps = 50
out = {}; t0 = time.time()

# ---------------- A. global Thorin catalogue ----------------
cat = {
 "zeta (xi)":            ("lfuncreate(1)", "zeta"),
 "L(s,chi_-4)":          ("lfuncreate(-4)", None),
 "L(s,chi_-3)":          ("lfuncreate(-3)", None),
 "L(s,chi_5)":           ("lfuncreate(5)", None),
 "Delta":                ("lfunmf(mfinit([1,12]),mfDelta())", None),
 "11a1 (rank 0)":        ("lfuncreate(ellinit([0,-1,1,-10,-20]))", None),
 "37a1 (rank 1)":        ("lfuncreate(ellinit([0,0,1,-1,0]))", None),
 "389a1 (rank 2)":       ("lfuncreate(ellinit([0,1,1,-2,0]))", None),
 "5077a1 (rank 3)":      ("lfuncreate(ellinit([0,0,1,-7,6]))", None),
 "Artin rho, x^3-x-1":   ("lfundiv(lfuncreate(x^3-x-1),lfuncreate(1))", None),
 "Sym^2 of 11a1":        ("lfunsympow(ellinit([0,-1,1,-10,-20]),2)", None),
}
T = 300
NS = 8
glob = {}
for name, (ctor, special) in cat.items():
    pari(f"L={ctor};")
    par = pari("lfunparams(L)"); N = float(par[0]); k = float(par[1]); d = len(par[2])
    c = k / 2
    r = 0 if special == "zeta" else int(pari("lfunorderzero(L)"))
    zs = [float(z) for z in pari(f"lfunzeros(L,{T})") if float(z) > 1e-6]
    tail = (math.log(N) + d * math.log(T / (2 * math.pi)) + d) / (2 * math.pi * T)
    # Taylor series of Xi(z) = Lambda(c+z)/z^r  (zeta: times (z^2-1/4))
    M = 2 * NS + r + 2
    if special == "zeta":
        ser = pari(f"lfunlambda(L, 1/2 + x + O(x^{M})) * (x^2 - 1/4)")
    else:
        pari(f"s0=lfunlambda(L, {c} + x + O(x^{M}));")
        ser = pari(f"sum(j={r},{M-1}, polcoef(s0,j)*x^(j-{r})) + O(x^{M-r})")
    lg = pari.log(ser / pari.polcoef(ser, 0))
    S_taylor = [abs(float(pari.polcoef(lg, 2 * n))) * n for n in range(1, NS + 1)]
    signs = [int(np.sign(float(pari.polcoef(lg, 2 * n)))) for n in range(1, NS + 1)]
    S_zeros = [sum(g ** (-2 * n) for g in zs) for n in range(1, NS + 1)]
    S_zeros[0] += tail
    # product check
    X0 = float(pari.polcoef(ser, 0))
    def Xi(s):
        z = math.sqrt(s)
        if special == "zeta":
            v = float(pari(f"lfunlambda(L, 1/2 + {z})")) * (z * z - 0.25)
        else:
            v = float(pari(f"lfunlambda(L, {c} + {z})")) / z ** r
        return v / X0
    prod = []
    for s in (4.0, 25.0, 100.0):
        lhs = Xi(s); rhs = math.exp(sum(math.log1p(s / g ** 2) for g in zs) + s * tail)
        prod.append((s, lhs, rhs, rhs / lhs - 1))
    # degenerate atom: slope of log Lambda(c + sqrt s) near s = 0 equals r/2
    if special != "zeta":
        s1, s2 = 1e-6, 1e-4
        v1 = abs(float(pari(f"lfunlambda(L, {c} + {math.sqrt(s1)})"))); v2 = abs(float(pari(f"lfunlambda(L, {c} + {math.sqrt(s2)})")))
        slope = (math.log(v2) - math.log(v1)) / (math.log(s2) - math.log(s1))
    else:
        slope = 0.0
    glob[name] = dict(N=N, k=k, degree=d, center=c, rank=r, n_zeros=len(zs), first_zero=zs[0] if zs else None,
                      product_check=prod, S_taylor=S_taylor, S_zeros=S_zeros, taylor_signs=signs, atom_slope=slope)
    print(name, r, len(zs), [f"{p[3]:.1e}" for p in prod], f"{S_taylor[0]:.6g} {S_zeros[0]:.6g}", f"slope {slope:.3f}", flush=True)
out["global"] = glob

# ---------------- B. Thorin Hankel (Stieltjes) test from central Taylor coefficients ----------------
def hankel(S, shift, m):
    A = mp.matrix(m, m)
    for i in range(m):
        for j in range(m): A[i, j] = mp.mpf(S[i + j + shift])
    return mp.det(A)
hk = {}
for name in glob:
    S = glob[name]["S_taylor"]
    hk[name] = {"H0": [float(hankel(S, 0, m)) for m in (1, 2, 3, 4)], "H1": [float(hankel(S, 1, m)) for m in (1, 2, 3)]}
out["hankel"] = hk

# ---------------- C. kernel cumulants for Delta (third route) ----------------
us, ph = np.load("phi.npy"); du = us[1] - us[0]
m2 = np.sum(us**2 * ph) * du; m4 = np.sum(us**4 * ph) * du; m6 = np.sum(us**6 * ph) * du
k4 = m4 - 3 * m2**2; k6 = m6 - 15 * m4 * m2 + 30 * m2**3
S = glob["Delta"]["S_taylor"]
out["delta_cumulants"] = {"kappa2": [m2, 2 * S[0]], "kappa4": [k4, -12 * S[1]], "kappa6": [k6, 240 * S[2]]}

# ---------------- D. explicit formula for Delta with Gaussian test function ----------------
sig = 4.5
pari("L=lfunmf(mfinit([1,12]),mfDelta());")
zD = [float(z) for z in pari("lfunzeros(L,120)") if float(z) > 1e-6]
h = lambda t: math.exp(-t * t / (2 * sig * sig))
Hx = lambda x: sig * math.sqrt(2 * math.pi) * math.exp(-sig * sig * x * x / 2)
zero_side = 2 * sum(h(g) for g in zD)
arch = mp.quad(lambda t: mp.exp(-t * t / (2 * sig * sig)) * 2 * mp.re(-mp.log(2 * mp.pi) + mp.digamma(6 + 1j * t)), [-60, 0, 60]) / (2 * mp.pi)
prime_terms = {}; comb_terms = {}
psum = 0.0
for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
    x = float(pari(f"ramanujantau({p})")) / p ** 5.5
    cs = [2.0, x]
    for m in range(2, 12): cs.append(x * cs[-1] - cs[-2])
    tp = -(1 / math.pi) * sum(cs[m] * math.log(p) * p ** (-m / 2) * Hx(m * math.log(p)) for m in range(1, 11))
    cp = (math.log(p) / math.pi) * sum(cs[m] * Hx(m * math.log(p)) for m in range(1, 11))
    prime_terms[p] = tp; comb_terms[p] = cp; psum += tp
out["explicit_formula"] = {"sigma": sig, "zero_side": zero_side, "arch": float(arch), "prime_side": psum,
                           "rhs": float(arch) + psum, "difference": zero_side - (float(arch) + psum),
                           "prime_terms": prime_terms, "comb_oscillatory_terms": comb_terms}
print("explicit", out["explicit_formula"]["zero_side"], out["explicit_formula"]["rhs"], flush=True)

# ---------------- E. hitting times on the tree: Keilson GGC ----------------
def bd_generator(q, n):
    Q = np.zeros((n, n))
    up = q / (q + 1); dn = 1 / (q + 1)
    Q[0, 0] = -1; Q[0, 1] = 1 if n > 1 else 0
    for k in range(1, n):
        Q[k, k] = -1; Q[k, k - 1] = dn
        if k + 1 < n: Q[k, k + 1] = up
    return Q
tree = {}
for q in (2, 3, 5):
    rho = 2 * math.sqrt(q) / (q + 1); res = {"rho": rho, "gap": 1 - rho}
    for n in (100, 400, 1600):
        Q = bd_generator(q, n)
        # symmetrize: D^{1/2} Q D^{-1/2} with detailed balance weights
        lam = np.sort(np.real(np.linalg.eigvals(-Q)))
        ET = float(np.sum(1 / lam)); VT = float(np.sum(1 / lam**2))
        # exact mean passage time by first step recursion
        e = [1.0]  # E[time 0->1] = 1
        up = q / (q + 1); dn = 1 / (q + 1)
        for kk in range(1, n):
            e.append((1 + dn * e[-1]) / up)   # E[k -> k+1] = (1 + dn*E[k-1 -> k]) / up  (holding mean 1)
        ET_direct = float(sum(e))
        # arcsine comparison of the empirical Thorin measure on the Kesten band
        u = np.clip((1 - lam) / rho, -1, 1); phis = np.sort(np.arccos(u)) / math.pi
        ks_arc = float(np.max(np.abs(phis - (np.arange(1, n + 1) - 0.5) / n)))
        res[n] = dict(min_rate=float(lam[0]), max_rate=float(lam[-1]), ET_over_n=ET / n, ET_direct_over_n=ET_direct / n,
                      VT_over_n=VT / n, ks_arcsine=ks_arc)
    res["limit_mean"] = (q + 1) / (q - 1); res["limit_var"] = ((q + 1) / (q - 1)) ** 3
    tree[q] = res
# simulation check of the GGC law for q = 2, n = 100
rng = np.random.default_rng(5); q, n, R = 2, 100, 20000
pos = np.zeros(R, int); tt = np.zeros(R); alive = np.ones(R, bool)
while alive.any():
    idx = np.nonzero(alive)[0]
    tt[idx] += rng.exponential(size=len(idx))
    p0 = pos[idx]; uu = rng.uniform(size=len(idx))
    step = np.where(p0 == 0, 1, np.where(uu < q / (q + 1), 1, -1))
    pos[idx] = p0 + step; alive[idx] = pos[idx] < n
lam = np.sort(np.real(np.linalg.eigvals(-bd_generator(q, n))))
ggc = (rng.exponential(size=(R, n)) / lam).sum(1)
from scipy.stats import ks_2samp
tree["sim_q2_n100"] = {"sim_mean": float(tt.mean()), "ggc_mean": float(ggc.mean()), "sim_var": float(tt.var()), "ggc_var": float(ggc.var()),
                       "ks_p": float(ks_2samp(tt, ggc).pvalue)}
np.save("tree_sim.npy", np.vstack([np.sort(tt), np.sort(ggc)]))
out["tree"] = tree
print("tree", json.dumps(tree["sim_q2_n100"]), flush=True)

# ---------------- F. exponential law of hitting times on isogeny graphs ----------------
pari('read("ss_graph.gp")')
def graph(p, l):
    pari(f"A=ssgraph({p},{l})"); n = int(pari("#A"))
    return np.array([[int(pari(f"A[{i},{j}]")) for j in range(1, n + 1)] for i in range(1, n + 1)], dtype=float)
iso = []
for p in (1009, 2017, 4129, 7993):
    if p % 12 != 1: continue
    A = graph(p, 2); hN = len(A); P = A / 3; ev = np.sort(np.linalg.eigvalsh(A))
    gap = 1 - ev[-2] / 3; tau2 = 1 / gap; rho = 2 * math.sqrt(2) / 3
    errs = []; ratios = []; mu2s = []; muhs = []
    rng = np.random.default_rng(p)
    for v in rng.choice(hN, size=min(6, hN), replace=False):
        idx = [i for i in range(hN) if i != v]
        Qv = (P - np.eye(hN))[np.ix_(idx, idx)]
        mu, V = np.linalg.eigh(-Qv)
        w = (V.T @ np.ones(hN - 1)) ** 2 / hN
        Et = float(np.sum(w / mu))
        ts = np.linspace(0, 6 * Et, 3000)
        surv = (w[None, :] * np.exp(-np.outer(ts, mu))).sum(1)
        err = float(np.max(np.abs(surv - np.exp(-ts / Et))))
        errs.append(err); ratios.append(tau2 / Et); mu2s.append(float(mu[1])); muhs.append(float(mu[-1]))
    iso.append(dict(N=p, h=hN, gap=gap, ramanujan_gap=1 - rho, tau2=tau2, max_sup_error=max(errs), max_aldous_brown_bound=max(ratios),
                    ramanujan_bound=(3 / (math.sqrt(2) - 1) ** 2) * 2 / (hN - 1), min_mu2=min(mu2s), kesten_lower=1 - rho, max_muh=max(muhs), kesten_upper=1 + rho))
    print(iso[-1], flush=True)
out["isogeny_exponential"] = iso

# ---------------- G. Weil: a genus 2 curve over F_101 as an unconditional Thorin comb ----------------
p = 101
cp = pari(f"hyperellcharpoly(Mod(1,{p})*(x^5-x+1))")
roots = [complex(z) for z in pari.polroots(cp)]
thetas = sorted({round(abs(math.atan2(z.imag, z.real)), 12) for z in roots})
out["weil"] = {"p": p, "charpoly": str(cp), "abs_roots_over_sqrtp": [abs(z) / math.sqrt(p) for z in roots], "thetas": thetas}
# product identity check: cosh w - cos th = (1 - cos th) prod_k (1 + w^2/(th + 2 pi k)^2)
chk = []
for th in thetas:
    for w in (0.7, 2.3):
        lhs = math.cosh(w) - math.cos(th)
        rhs = (1 - math.cos(th)) * math.prod(1 + w * w / (th + 2 * math.pi * kk) ** 2 for kk in range(-20000, 20001))
        chk.append(rhs / lhs - 1)
out["weil"]["product_identity_max_relerr"] = max(abs(c) for c in chk)
out["time"] = time.time() - t0
json.dump(out, open("stage5.json", "w"), indent=1)
print("done", out["time"])
