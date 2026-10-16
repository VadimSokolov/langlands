import cypari2, numpy as np, json, math, mpmath as mp
pari = cypari2.Pari(); pari.allocatemem(4 * 10**9); pari.set_real_precision(200)
mp.mp.dps = 120
out = {}

# ---------------- (a) Patie class test on Taylor coefficients ----------------
cat = {
 "xi":        ("lfuncreate(1)", "zeta"),
 "chi_-4":    ("lfuncreate(-4)", None),
 "Delta":     ("lfunmf(mfinit([1,12]),mfDelta())", None),
 "11a1":      ("lfuncreate(ellinit([0,-1,1,-10,-20]))", None),
 "Artin":     ("lfundiv(lfuncreate(x^3-x-1),lfuncreate(1))", None),
 "Sym2 11a1": ("lfunsympow(ellinit([0,-1,1,-10,-20]),2)", None),
}
NT = 26
patie = {}
coeffs_all = {}
for name, (ctor, sp) in cat.items():
    pari(f"L={ctor};"); k = float(pari("lfunparams(L)")[1]); c = k / 2
    M = 2 * NT + 2
    if sp == "zeta":
        ser = pari(f"lfunlambda(L, 1/2 + x + O(x^{M})) * (x^2 - 1/4)")
    else:
        ser = pari(f"lfunlambda(L, {c} + x + O(x^{M}))")
    a = [mp.mpf(str(pari.polcoef(ser, 2 * n)).replace(" E", "E")) for n in range(NT + 1)]
    odd = max(abs(float(pari.polcoef(ser, 2 * n + 1))) for n in range(NT)) / float(a[0])
    psi = [a[n - 1] / a[n] for n in range(1, NT + 1)]          # implied Psi(n), n = 1..NT
    d2 = [psi[i + 2] - 2 * psi[i + 1] + psi[i] for i in range(len(psi) - 2)]   # Delta^2 Psi(n)
    # completely monotone sequence test: (-1)^k Delta^k d2 >= 0
    table = []; cur = d2[:]
    first_violation = None
    for kk in range(0, 9):
        signs = [(-1) ** kk * v for v in cur]
        neg = [i for i, v in enumerate(signs) if v < 0]
        table.append({"k": kk, "min": float(min(signs)), "n_negative": len(neg), "first_negative_index": (neg[0] + 1) if neg else None})
        if neg and first_violation is None: first_violation = (kk, neg[0] + 1)
        cur = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
    # growth: psi(n)/n^2 and fit psi(n) ~ c n^2 / log(n)^p
    ratio = [float(psi[n - 1] / n**2) for n in (1, 2, 5, 10, 15, 20, 25)]
    patie[name] = {"positive_coeffs": all(v > 0 for v in a), "odd_coeff_rel": odd,
                   "psi_first": [float(v) for v in psi[:8]], "psi_over_n2": ratio,
                   "psi_convex": all(v > 0 for v in d2), "cm_table": table, "first_violation": first_violation}
    coeffs_all[name] = [str(v) for v in a]
    print(name, "psi:", [round(float(v), 4) for v in psi[:6]], "violation:", first_violation, flush=True)
out["patie"] = patie

# ---------------- (b) van Dantzig test for Delta: density of V with ch.f. Lambda(6)/Lambda(6+it) ----------------
pari.set_real_precision(40)
pari("L=lfunmf(mfinit([1,12]),mfDelta());")
L6 = float(pari("lfunlambda(L,6)"))
ts = np.linspace(0, 60, 1201); dt = ts[1] - ts[0]
g = np.array([L6 / float(pari(f"real(lfunlambda(L,6+{t}))")) for t in ts])   # = prod (1+t^2/gamma^2)^{-1}
xs = np.linspace(0, 1.5, 301)
w = np.ones_like(ts); w[0] = w[-1] = 0.5
dens = np.array([(w * g * np.cos(ts * x)).sum() * dt / math.pi for x in xs])
out["vandantzig"] = {"g_at_60": float(g[-1]), "density_at_0": float(dens[0]), "min_density": float(dens.min()),
                     "argmin": float(xs[np.argmin(dens)]), "total_mass": float(np.trapezoid(dens, xs) * 2),
                     "logconcave": bool(np.all(np.diff(np.log(np.maximum(dens[:200], 1e-300)), 2) <= 1e-9))}
np.save("vdp_density.npy", np.vstack([xs, dens]))
print("vdp", out["vandantzig"], flush=True)

# ---------------- (c) heat trace W(x) = sum_{gamma>0} exp(-x gamma^2) for Delta: zeros vs primes ----------------
zD = [float(z) for z in pari("lfunzeros(L,400)") if float(z) > 1e-6]
P = [int(p) for p in pari("primes([2,200000])")]
pari("T=mfcoefs(mfDelta(),200000);")
tau = {p: float(pari(f"T[{p}+1]")) for p in P}
def W_zero(x, n=0): return sum(g2 ** n * math.exp(-x * g2) for g2 in (z * z for z in zD))
def W_prime(x, n=0):
    # explicit formula with h(t) = t^{2n} e^{-x t^2}; H(u) = int h(t) e^{-iut} dt
    # base Gaussian G(u) = sqrt(pi/x) exp(-u^2/(4x)); H = (-1)^n d^{2n}/du^{2n} G
    def H(u):
        uu = mp.mpf(u); X = mp.mpf(x)
        return float((-1) ** n * mp.diff(lambda v: mp.sqrt(mp.pi / X) * mp.e ** (-v * v / (4 * X)), uu, 2 * n))
    arch = mp.quad(lambda t: t ** (2 * n) * mp.e ** (-x * t * t) * 2 * mp.re(-mp.log(2 * mp.pi) + mp.digamma(6 + 1j * t)), [-80, 0, 80]) / (2 * mp.pi)
    umax = math.sqrt(4 * x * 60) + 1
    pr = 0.0
    for p in P:
        Lp = math.log(p)
        if Lp > umax: break
        xp = tau[p] / p ** 5.5; cs = [2.0, xp]
        for m in range(2, 40): cs.append(xp * cs[-1] - cs[-2])
        m = 1
        while m * Lp <= umax:
            pr += cs[m] * Lp * p ** (-m / 2) * H(m * Lp); m += 1
    total = float(arch) - pr / math.pi           # sum over all zeros (both signs) of h(gamma)
    return total / 2                              # positive ordinates only
heat = []
for x in (0.02, 0.05, 0.1):
    row = {"x": x}
    for n in (0, 1, 2, 3):
        row[f"zero_n{n}"] = W_zero(x, n); row[f"prime_n{n}"] = W_prime(x, n)
    heat.append(row); print(row, flush=True)
out["heat_trace"] = heat

# ---------------- (d) zeros from primes: smoothed Thorin density by sliding bump ----------------
wid = 1.0
def bump_prime(t0):
    H = lambda u: 2 * wid * math.sqrt(2 * math.pi) * math.exp(-wid * wid * u * u / 2) * math.cos(t0 * u)
    arch = mp.quad(lambda t: (mp.e ** (-(t - t0) ** 2 / (2 * wid * wid)) + mp.e ** (-(t + t0) ** 2 / (2 * wid * wid))) * 2 * mp.re(-mp.log(2 * mp.pi) + mp.digamma(6 + 1j * t)), [-80, -t0, 0, t0, 80]) / (2 * mp.pi)
    umax = math.sqrt(2 * 60) / wid + 1; pr = 0.0
    for p in P:
        Lp = math.log(p)
        if Lp > umax: break
        xp = tau[p] / p ** 5.5; cs = [2.0, xp]
        for m in range(2, 40): cs.append(xp * cs[-1] - cs[-2])
        m = 1
        while m * Lp <= umax:
            pr += cs[m] * Lp * p ** (-m / 2) * H(m * Lp); m += 1
    return float(arch) - pr / math.pi
def bump_zero(t0): return sum(math.exp(-(g - t0) ** 2 / (2 * wid * wid)) + math.exp(-(g + t0) ** 2 / (2 * wid * wid)) for g in zD) * 2 / 2
grid = np.linspace(0.5, 40, 160)
bp = np.array([bump_prime(t) for t in grid]); bz = np.array([bump_zero(t) for t in grid])
np.save("bump.npy", np.vstack([grid, bp, bz]))
out["bump"] = {"width": wid, "max_abs_diff": float(np.max(np.abs(bp - bz))), "max_value": float(bz.max())}
print("bump", out["bump"], flush=True)

# ---------------- (e) 2 sin^2 transport check: cosine Levy measure -> Thorin measure ----------------
# U(z) = -(1/(2 pi sqrt z)) int 2 sin^2(x sqrt z / 2) mu(dx), mu = 2 sum cos(gamma x) dx, smoothed by e^{-eps x}
eps = 0.05
def U_transport(z):
    r = math.sqrt(z)
    # int_0^inf (1 - cos(r x)) cos(g x) e^{-eps x} dx = eps/(eps^2+g^2) - (1/2)[eps/(eps^2+(r-g)^2) + eps/(eps^2+(r+g)^2)]
    tot = 0.0
    for gma in zD[:60]:
        tot += 2 * (eps / (eps**2 + gma**2) - 0.5 * (eps / (eps**2 + (r - gma)**2) + eps / (eps**2 + (r + gma)**2)))
    return -tot / (2 * math.pi * r)
zz = np.linspace(4, 900, 3000)
U = np.array([U_transport(z) for z in zz])
# mass near the first atom gamma_1^2 should be ~1
g1 = zD[0] ** 2; mask = np.abs(zz - g1) < 30
out["transport"] = {"eps": eps, "mass_near_first_atom": float(np.trapezoid(U[mask], zz[mask])), "gamma1_sq": g1,
                    "min_U": float(U.min())}
np.save("transport.npy", np.vstack([zz, U]))
print("transport", out["transport"], flush=True)
json.dump(out, open("stage6.json", "w"), indent=1)
json.dump(coeffs_all, open("stage6_coeffs.json", "w"))
print("done")
