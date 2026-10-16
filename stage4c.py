"""Genus 2 numbers of Chapter 19 (book/chapters/ch12.tex), recomputed from scratch. Reads no files.

Supersedes stage4.json for Chapter 19. That file (written by stage4.py, then by stage4b.py) keeps valid traces and
moments, but its trace densities were histograms of a 2001 by 2001 grid of sums whose values fall exactly on the
bin edges, so its Kullback Leibler block, posteriors and rates are aliased and disagree with the chapter.
Here the densities are evaluated by quadrature.

Computes
  1. The Frobenius traces a_p of C1: y^2 = x^5 - x + 1, C2: y^2 = x^6 + 3x^4 + x^2 + 1 and C3: y^2 = x^6 + x^3 + 1
     at every odd prime p < 10^5 not dividing the discriminant (9589, 9590, 9590 primes), by a numpy point count,
     checked against PARI hyperellcharpoly at every good p < 3000 and at 5 seeded larger primes per curve. Also
     a_p(C2) = a_p(E) + a_p(E') for the two elliptic curves of the chapter (j invariants, a_5), the Galois group of
     x^5 - x + 1, and the zeros of a_p for C3 by residue class mod 3.
  2. The trace densities of USp(4), SU(2)xSU(2) and the diagonal SU(2) on [-4, 4] by quadrature: a midpoint rule in
     an angle that removes the endpoint singularities, checked against scipy.integrate.quad and against elliptic
     integrals. Normalisations 8 pi^2 and 4 pi^2, masses, moments 1, 3, 14 / 2, 10, 70 / 4, 32, 320, the values f(0)
     and the edge exponents (4 - |t|)^4 and (4 - |t|)^(1/2).
  3. The Kullback Leibler table (graded Gauss Legendre panels), checked by Monte Carlo (seed 20261009).
  4. For each curve: even moments with standard errors, posteriors under a uniform prior after 10, 50, 100, ...
     primes, per prime log Bayes factors against USp(4), with and without clipping at 20.
  5. The misspecified case C3: the atom at 0, the moments off the atom, predicted and observed rates, the mean log
     likelihood of the winner against its expectation.
  6. A table of every genus 2 number printed in ch12.tex against the value computed here.

Chapter 19 numbers backed: Table tab:12kl; the constants of the section "Genus two: three connected candidates";
the "about 120 primes" of "Sequential testing"; the moments, posteriors, rates and C2, C3 facts of "Three curves"
and the captions of Figures fig:12genus2 and fig:12posterior; the values f(0), the predicted and observed rates and
the log likelihoods of "Misspecification".

Run: python3 stage4c.py [--quick]    (needs numpy, scipy and cypari2 with PARI 2.15; --quick uses p < 3000 and 2e5
Monte Carlo draws and does not reproduce the chapter's counts). On the cluster:
misc/tools/hopper_run.sh -t 120 -c 4 stage4c.py. Writes stage4c.json to the working directory: every number above,
the good primes and traces of the three curves (keys primes and ap), the chapter check table and the internal checks.
"""
import json, math, os, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np, scipy, cypari2
from scipy.integrate import quad
from scipy.special import ellipk, ellipe

SEED = 20261009
QUICK = "--quick" in sys.argv
LIMIT = 3000 if QUICK else 10**5      # odd primes p < LIMIT
HYPER_ALL = 1500 if QUICK else 3000   # hyperellcharpoly at every good prime below this ...
HYPER_BIG = 2 if QUICK else 5         # ... and at this many seeded larger primes per curve
MC = 200000 if QUICK else 4000000     # Monte Carlo draws per group
NODES, NODES_MC = 4096, 256           # midpoint nodes on [0, pi] in the density integrals
M = 20.0                              # clipping level of the log Bayes factors
ALPHA = 1e-6                          # error level of the sequential test in the chapter
NS = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000]   # posterior checkpoints
CURVES = {"C1": ("y^2 = x^5 - x + 1", [1, 0, 0, 0, -1, 1]),
          "C2": ("y^2 = x^6 + 3x^4 + x^2 + 1", [1, 0, 3, 0, 1, 0, 1]),
          "C3": ("y^2 = x^6 + x^3 + 1", [1, 0, 0, 1, 0, 0, 1])}
GROUPS = ["USp(4)", "SU(2)xSU(2)", "SU(2) diagonal"]


def poly(c):
    return "+".join(f"({a})*x^{len(c) - 1 - i}" for i, a in enumerate(c))


def ap_count(c, p):
    """a_p = p + 1 - #C(F_p) for y^2 = c[0] x^d + ... + c[d], p odd, p not dividing the discriminant."""
    x = np.arange(p, dtype=np.int64); v = np.zeros(p, dtype=np.int64)
    for a in c: v = (v * x + a) % p
    sq = np.zeros(p, np.int8); sq[(x * x) % p] = 1
    chi = np.where(v == 0, 0, np.where(sq[v] == 1, 1, -1))
    inf = 1 if len(c) % 2 == 0 else (2 if sq[c[0] % p] else 0)
    return 1 - int(chi.sum()) - inf


_pari = None


def init_worker():
    global _pari
    _pari = cypari2.Pari(); _pari.allocatemem(10**9)


def ap_hyper(job):
    f, p = job
    return -int(_pari.polcoef(_pari(f"hyperellcharpoly(Mod(1,{p})*({f}))"), 3))


def raw_dens(t, n=NODES):
    """(8 pi^2 f_USp4(t), 4 pi^2 f_SU2xSU2(t)) for |t| < 4. With x = t/2 + w cos(phi), w = 2 - t/2, A = 2 + t/2:
    sqrt((4 - x^2)(4 - (t - x)^2)) dx = w^2 sin^2(phi) sqrt(4t + w^2 sin^2(phi)) dphi, and (2x - t)^2 = 4 w^2 cos^2(phi)."""
    t = np.abs(np.asarray(t, float)).ravel()
    phi = np.pi * (np.arange(n) + 0.5) / n
    s2 = np.sin(phi) ** 2; c2s2 = np.cos(phi) ** 2 * s2
    iu = np.empty(t.size); i_s = np.empty(t.size); ch = max(1, 2 * 10**6 // n)
    for i in range(0, t.size, ch):
        tt = t[i:i + ch]; w = 2 - tt / 2
        r = np.sqrt(4 * tt[:, None] + (w ** 2)[:, None] * s2[None, :])
        i_s[i:i + ch] = w ** 2 * (r @ s2) * (np.pi / n)
        iu[i:i + ch] = 4 * w ** 4 * (r @ c2s2) * (np.pi / n)
    return iu, i_s


def dens(t, n=NODES):
    """Trace densities (USp(4), SU(2)xSU(2), SU(2) diagonal) at t, even in t."""
    ta = np.abs(np.asarray(t, float)).ravel()
    iu, i_s = raw_dens(ta, n)
    return iu / (8 * np.pi ** 2), i_s / (4 * np.pi ** 2), np.sqrt(np.maximum((4 - ta) * (4 + ta), 0.0)) / (8 * np.pi)


def logs(fs):
    return np.log(np.maximum(np.stack(fs), 1e-300))


def panels(order=32, t0=1e-4, kmax=33):
    """Gauss Legendre panels on [t0, 4), geometrically graded towards 0 and towards 4."""
    br = sorted(set([t0 * 2.0 ** k for k in range(15)] + [4.0 - 2.0 * 2.0 ** (-k) for k in range(kmax + 1)]))
    xg, wg = np.polynomial.legendre.leggauss(order)
    t = np.concatenate([0.5 * (b - a) * xg + 0.5 * (b + a) for a, b in zip(br[:-1], br[1:])])
    w = np.concatenate([0.5 * (b - a) * wg for a, b in zip(br[:-1], br[1:])])
    return t, w, t0


def kl_table(order, kmax):
    """KL(P || Q) in nats for P, Q in the three trace laws, on graded Gauss Legendre panels of the given order."""
    t, w, t0 = panels(order, kmax=kmax)
    f = dens(t); f0 = [float(v[0]) for v in dens([0.0])]
    lg = [np.log(np.maximum(v, 1e-300)) for v in f]; l0 = [math.log(v) for v in f0]
    return [[2.0 * (float(w @ (f[a] * (lg[a] - lg[b]))) + t0 * f0[a] * (l0[a] - l0[b])) if a != b else 0.0
             for b in range(3)] for a in range(3)]


def cat(m):
    return math.comb(2 * m, m) // (m + 1)


def sc(n):
    return cat(n // 2) if n % 2 == 0 else 0   # E x^n for x semicircular on [-2, 2]


def mom_exact(g, n):
    """E t^n for the trace t = x1 + x2 of USp(4) (g = 0), of SU(2)xSU(2) (g = 1), t = 2x of the diagonal SU(2) (g = 2)."""
    if g == 0: return sum(math.comb(n, k) * (sc(k + 2) * sc(n - k) - sc(k + 1) * sc(n - k + 1)) for k in range(n + 1))
    if g == 1: return sum(math.comb(n, k) * sc(k) * sc(n - k) for k in range(n + 1))
    return 2 ** n * sc(n)


def s_raw(x):
    return math.sqrt(max(4.0 - x * x, 0.0))


def quad_raw(t):
    """(8 pi^2 f_USp4(t), 4 pi^2 f_SU2xSU2(t)) straight from the convolution integrals, by scipy.integrate.quad."""
    lo, hi = max(-2.0, t - 2.0), min(2.0, t + 2.0)
    kw = dict(limit=500, epsabs=0.0, epsrel=1e-11)   # relative tolerance only: the integrals shrink like (4 - t)^4 near 4
    return (quad(lambda x: (2 * x - t) ** 2 * s_raw(x) * s_raw(t - x), lo, hi, **kw)[0],
            quad(lambda x: s_raw(x) * s_raw(t - x), lo, hi, **kw)[0])


def fs_elliptic(t):
    """f_SU2xSU2(t) = A^3 ((1 + k^2) E(k) - (1 - k^2) K(k)) / (6 pi^2), A = 2 + t/2, k = (4 - t)/(4 + t), 0 < t < 4."""
    k2 = ((4.0 - t) / (4.0 + t)) ** 2
    return (2 + t / 2) ** 3 * ((1 + k2) * ellipe(k2) - (1 - k2) * ellipk(k2)) / (6 * np.pi ** 2)


def semicircle(rng, n):   # density sqrt(4 - x^2) / (2 pi): first coordinate of a uniform point of the disc of radius 2
    return 2.0 * np.sqrt(rng.random(n)) * np.cos(2.0 * np.pi * rng.random(n))


def sample_t(rng, g, n):
    if g == 2: return 2.0 * semicircle(rng, n)
    chunk, got, res = min(4 * 10**6, 10 * n), 0, []
    while got < n:
        x1, x2 = semicircle(rng, chunk), semicircle(rng, chunk)
        t = x1 + x2
        if g == 0: t = t[rng.random(chunk) * 16.0 < (x1 - x2) ** 2]   # joint density (x1 - x2)^2 s(x1) s(x2) / 2
        res.append(t); got += t.size
    return np.concatenate(res)[:n]


def posterior(L, ns, clip):
    """Uniform prior; L[g] = log f_g(x_k). Returns the posterior at each n in ns and the full path of group 1."""
    d = L[1:] - L[0]
    if clip: d = np.clip(d, -M, M)
    H = np.vstack([np.zeros(L.shape[1]), np.cumsum(d, axis=1)])    # log Bayes factors against USp(4) after n primes
    P = np.exp(H - H.max(0)); P /= P.sum(0)
    return {str(n): [float(v) for v in P[:, n - 1]] for n in ns}, P, H


def dumps(o, ind=0):
    if isinstance(o, np.ndarray): o = o.tolist()
    if isinstance(o, np.generic): o = o.item()
    if isinstance(o, dict):
        if not o: return "{}"
        pad = " " * (ind + 1)
        return "{\n" + ",\n".join(f"{pad}{json.dumps(str(k))}: {dumps(v, ind + 1)}" for k, v in o.items()) + "\n" + " " * ind + "}"
    if isinstance(o, (list, tuple)):
        o = [v.item() if isinstance(v, np.generic) else v for v in o]
        if all(not isinstance(v, (dict, list, tuple, np.ndarray)) for v in o): return json.dumps(o)
        pad = " " * (ind + 1)
        return "[\n" + ",\n".join(pad + dumps(v, ind + 1) for v in o) + "\n" + " " * ind + "]"
    return json.dumps(o)


def main():
    t0 = time.time(); stamps = []

    def log(*a):
        print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
        stamps.append([round(time.time() - t0, 1), " ".join(str(v) for v in a)])

    r_pick, r_check, r_mc = (np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(3))
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
    pari = cypari2.Pari(); pari.allocatemem(10**9)
    out = {"meta": {"script": "stage4c.py", "seed": SEED, "quick": QUICK, "prime_limit": LIMIT, "mc_draws": MC,
                    "midpoint_nodes": NODES, "clip": M, "python": sys.version.split()[0], "numpy": np.__version__,
                    "scipy": scipy.__version__, "pari": [int(v) for v in pari.version()], "workers": nproc}}
    checks = {}

    def check(name, value, limit, ok=None):
        checks[name] = {"value": float(value), "limit": limit, "ok": bool(value <= limit if ok is None else ok)}

    # ---- 1. traces ---------------------------------------------------------------------------------------------
    log("1. point counts (numpy) and the PARI cross-check (hyperellcharpoly)")
    primes = [int(q) for q in pari(f"primes([3,{LIMIT - 1}])")]
    good, ap, info = {}, {}, {}
    for name, (eq, c) in CURVES.items():
        D = int(pari(f"poldisc({poly(c)})"))
        good[name] = [p for p in primes if D % p]
        info[name] = {"equation": eq, "coefficients_high_to_low": c, "discriminant": D,
                      "bad_odd_primes": [p for p in primes if D % p == 0]}
    picks = {}
    for name in CURVES:
        big = [p for p in good[name] if p >= HYPER_ALL]
        picks[name] = sorted(big[int(i)] for i in r_pick.choice(len(big), HYPER_BIG, replace=False))
    jobs = sorted(((n, p) for n in CURVES for p in [q for q in good[n] if q < HYPER_ALL] + picks[n]), key=lambda j: -j[1])
    with ProcessPoolExecutor(nproc, mp_context=mp.get_context("spawn"), initializer=init_worker) as ex:
        futs = {j: ex.submit(ap_hyper, (poly(CURVES[j[0]][1]), j[1])) for j in jobs}
        for name, (eq, c) in CURVES.items():
            ap[name] = np.array([ap_count(c, p) for p in good[name]], dtype=np.int64)
            log(f"   {name}: {len(good[name])} good primes counted")
        hyp = {j: f.result() for j, f in futs.items()}
    pos = {n: {p: i for i, p in enumerate(good[n])} for n in CURVES}
    mism = [[n, p, h, int(ap[n][pos[n][p]])] for (n, p), h in sorted(hyp.items()) if h != ap[n][pos[n][p]]]
    out["trace_check"] = {"hyperellcharpoly_every_good_prime_below": HYPER_ALL, "larger_primes_checked": picks,
                          "compared": len(hyp), "mismatches": mism}
    check("hyperellcharpoly against point count, mismatches", len(mism), 0)
    log(f"   hyperellcharpoly agrees with the point count at {len(hyp) - len(mism)} of {len(hyp)} primes")

    pari(f"P=[{','.join(map(str, good['C2']))}]; E=ellinit([0,0,0,-2,2]); F=ellinit([0,1,0,3,1]);")
    split = np.array([int(v) for v in pari("vector(#P,i,ellap(E,P[i])+ellap(F,P[i]))")])
    out["C1_galois_group"] = str(pari("polgalois(x^5-x+1)"))
    out["C2_split"] = {"E": "y^2 = x^3 - 2x + 2", "Eprime": "y^2 = x^3 + x^2 + 3x + 1",
                       "j_E": str(pari("E.j")), "j_Eprime": str(pari("F.j")),
                       "a5_E": int(pari("ellap(E,5)")), "a5_Eprime": int(pari("ellap(F,5)")),
                       "primes_compared": len(split), "mismatches_ap_C2_vs_ap_E_plus_ap_Eprime": int(np.sum(split != ap["C2"]))}
    check("a_p(C2) = a_p(E) + a_p(E'), mismatches", out["C2_split"]["mismatches_ap_C2_vs_ap_E_plus_ap_Eprime"], 0)

    # ---- 2. densities ------------------------------------------------------------------------------------------
    log("2. trace densities by quadrature")
    TQ, WQ, T0 = panels()

    def integ(g, g0=0.0):   # integral over [-4, 4] of an even function: panels on [T0, 4) plus the constant g0 on [0, T0]
        return 2.0 * (float(WQ @ g) + T0 * g0)

    FQ = dens(TQ); F0 = [float(f[0]) for f in dens([0.0])]
    LQ = [np.log(np.maximum(f, 1e-300)) for f in FQ]; L0 = [math.log(f) for f in F0]
    IUq, ISq = raw_dens(TQ); IU0, IS0 = raw_dens([0.0])
    norm = [integ(IUq, IU0[0]), integ(ISq, IS0[0])]
    mass = [integ(FQ[g], F0[g]) for g in range(3)]
    mom_q = {k: [integ(TQ ** k * FQ[g]) for g in range(3)] for k in (2, 4, 6)}
    mom_x = {k: [float(mom_exact(g, k)) for g in range(3)] for k in (2, 4, 6)}
    f0_exact = [64 / (15 * math.pi ** 2), 8 / (3 * math.pi ** 2), 1 / (2 * math.pi)]
    u1, u2 = 1e-4, 2e-4
    fe = dens([4 - u1, 4 - u2])
    slope = [math.log(f[0] / f[1]) / math.log(u1 / u2) for f in fe]
    th = np.pi * (np.arange(64) + 0.5) / 64; cth = np.cos(th); sth2 = np.sin(th) ** 2
    weyl_mass = 8 / np.pi ** 2 * (np.pi / 64) ** 2 * float(np.sum((cth[:, None] - cth[None, :]) ** 2 * sth2[:, None] * sth2[None, :]))
    tt = np.concatenate([[0.003, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 3.5, 3.9, 3.99, 3.999], r_check.uniform(0.0, 4.0, 29)])
    iut, ist = raw_dens(tt); qq = np.array([quad_raw(float(t)) for t in tt])
    rel_u, rel_s = np.abs(iut / qq[:, 0] - 1), np.abs(ist / qq[:, 1] - 1)
    rel_quad = float(max(rel_u.max(), rel_s.max()))
    sel = tt <= 3.9
    rel_ell = float(np.max(np.abs(ist[sel] / (4 * np.pi ** 2) / fs_elliptic(tt[sel]) - 1)))
    out["densities"] = {
        "normalisation_quadrature": {"USp4_integral_of_8pi2_f": norm[0], "8pi2": 8 * math.pi ** 2,
                                     "SU2xSU2_integral_of_4pi2_f": norm[1], "4pi2": 4 * math.pi ** 2},
        "mass": mass, "even_moments_quadrature": {str(k): v for k, v in mom_q.items()},
        "even_moments_exact": {str(k): v for k, v in mom_x.items()},
        "f0_quadrature": F0, "f0_exact": f0_exact, "f0_names": ["64/(15 pi^2)", "8/(3 pi^2)", "1/(2 pi)"],
        "edge_exponent_at_4": slope, "edge_exponent_expected": [4, 2, 0.5],
        "edge_constants": {"f_USp4_over_u4": float(fe[0][0] / u1 ** 4), "1/(64 pi)": 1 / (64 * math.pi)},
        "weyl_density_mass": weyl_mass,
        "scipy_quad_check": {"t": tt, "relative_difference_USp4": rel_u, "relative_difference_SU2xSU2": rel_s, "max": rel_quad},
        "max_relative_difference_to_elliptic_integrals": rel_ell}
    check("quadrature against scipy quad, max relative difference", rel_quad, 1e-8)
    check("f_SU2xSU2 against elliptic integrals, max relative difference", rel_ell, 1e-8)
    check("normalisation 8 pi^2, relative error", abs(norm[0] / (8 * math.pi ** 2) - 1), 1e-10)
    check("normalisation 4 pi^2, relative error", abs(norm[1] / (4 * math.pi ** 2) - 1), 1e-10)
    check("masses equal 1, max error", max(abs(m - 1) for m in mass), 1e-10)
    check("even moments against exact values, max relative error", max(abs(a / b - 1) for k in mom_q for a, b in zip(mom_q[k], mom_x[k])), 1e-9)
    check("f(0) against closed forms, max relative error", max(abs(a / b - 1) for a, b in zip(F0, f0_exact)), 1e-10)
    check("edge exponents, max error", max(abs(s - e) for s, e in zip(slope, (4, 2, 0.5))), 1e-3)
    check("Weyl density of USp(4) has mass 1, error", abs(weyl_mass - 1), 1e-12)

    # ---- 3. Kullback Leibler table -----------------------------------------------------------------------------
    log("3. Kullback Leibler table by quadrature and by Monte Carlo")
    KL = [[integ(FQ[a] * (LQ[a] - LQ[b]), F0[a] * (L0[a] - L0[b])) if a != b else 0.0 for b in range(3)] for a in range(3)]
    KLc = [[integ(FQ[a] * np.clip(LQ[a] - LQ[b], -M, M), F0[a] * min(M, max(-M, L0[a] - L0[b]))) if a != b else 0.0
            for b in range(3)] for a in range(3)]
    T = [sample_t(r_mc, g, MC) for g in range(3)]
    mc, mcm = {}, {}
    for a in range(3):
        La = logs(dens(T[a], NODES_MC))
        for b in range(3):
            if a != b:
                d = La[a] - La[b]; est, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(MC))
                mc[f"{GROUPS[a]}||{GROUPS[b]}"] = {"estimate": est, "se": se, "z": (est - KL[a][b]) / se}
        mcm[GROUPS[a]] = {"estimate": [float(np.mean(T[a] ** k)) for k in (2, 4, 6)],
                          "se": [float(np.std(T[a] ** k, ddof=1) / math.sqrt(MC)) for k in (2, 4, 6)],
                          "exact": [mom_x[k][a] for k in (2, 4, 6)]}
    out["kl"] = {"rows_P_columns_Q": GROUPS, "table": KL, "table_clipped_at_20": KLc,
                 "pairs": {f"{GROUPS[a]}||{GROUPS[b]}": KL[a][b] for a in range(3) for b in range(3) if a != b},
                 "monte_carlo": mc, "monte_carlo_even_moments_2_4_6": mcm}
    check("Monte Carlo against quadrature, max |z| over the six divergences", max(abs(v["z"]) for v in mc.values()), 4.0)
    check("Monte Carlo even moments against exact values, max |z|",
          max(abs(e - x) / s for v in mcm.values() for e, s, x in zip(v["estimate"], v["se"], v["exact"])), 4.0)
    KL2 = kl_table(48, 30)
    out["kl"]["table_second_panel_set_order_48"] = KL2
    check("KL table, second panel set, max difference", max(abs(KL[a][b] - KL2[a][b]) for a in range(3) for b in range(3)), 1e-9)
    log("   KL table", [[round(v, 4) for v in r] for r in KL])
    seq = {"alpha": ALPHA, "log(1/alpha)": math.log(1 / ALPHA),
           "smallest_divergence": min(KL[a][b] for a in range(3) for b in range(3) if a != b),
           "primes_for_smallest_divergence": math.log(1 / ALPHA) / min(KL[a][b] for a in range(3) for b in range(3) if a != b),
           "exercise_primes_SU2xSU2_against_USp4": math.log(1 / ALPHA) / KL[1][0]}
    out["sequential"] = seq

    # ---- 4. the three curves -----------------------------------------------------------------------------------
    log("4. moments, posteriors and rates of the three curves")
    out["curves"] = {}; X = {}; LL = {}
    for name in CURVES:
        a = ap[name]; p = np.array(good[name], dtype=float); n = len(a); x = a / np.sqrt(p); X[name] = x
        ux, inv = np.unique(np.abs(x), return_inverse=True)
        L = logs(dens(ux))[:, inv.ravel()]; LL[name] = L
        m = [float(np.mean(x ** k)) for k in (2, 4, 6)]
        z = [[(m[j] - mom_exact(g, 2 * (j + 1))) / math.sqrt((mom_exact(g, 4 * (j + 1)) - mom_exact(g, 2 * (j + 1)) ** 2) / n)
              for j in range(3)] for g in range(3)]
        ns = sorted(set([k for k in NS if k <= n] + [n]))
        post_c, Pc, Hc = posterior(L, ns, True); post_u, Pu, Hu = posterior(L, ns, False)
        win = int(np.argmax(Pc[:, -1]))
        bad = np.nonzero(Pc[win] < 0.99)[0]
        d = L[1:] - L[0]
        out["curves"][name] = dict(info[name], n=n, primes=good[name], ap=a, even_moments=m, moment_z_scores_by_group=z,
                                   posterior_clipped=post_c, posterior_unclipped=post_u, winner=GROUPS[win],
                                   n_after_which_winner_stays_above_0_99=int(bad[-1] + 2) if bad.size else 1,
                                   rate_vs_USp4_clipped=[float(v) for v in Hc[:, -1] / n],
                                   rate_vs_USp4_unclipped=[0.0] + [float(v) for v in d.mean(1)],
                                   terms_clipped_at_20=int(np.sum(np.abs(d) > M)), max_abs_log_ratio=float(np.max(np.abs(d))),
                                   mean_loglik=[float(v) for v in L.mean(1)])
        log(f"   {name}: n = {n}, moments {[round(v, 3) for v in m]}, posterior at 10/50/100 {[round(post_c[str(k)][win], 5) for k in (10, 50, 100)]}")
    check("clipping at 20 never active, terms clipped", sum(out["curves"][c]["terms_clipped_at_20"] for c in CURVES), 0)
    check("posterior at 10, 50, 100: clipped against unclipped, max difference",
          max(abs(out["curves"][c]["posterior_clipped"][k][g] - out["curves"][c]["posterior_unclipped"][k][g])
              for c in CURVES for k in ("10", "50", "100") for g in range(3)), 1e-9)

    # ---- 5. C3: atom, moments off the atom, misspecification ---------------------------------------------------
    log("5. the misspecified curve C3")
    a3 = ap["C3"]; p3 = np.array(good["C3"]); x3 = X["C3"]; zero = a3 == 0
    nz = ~zero
    c3 = {"n": int(len(a3)), "atom_mass": float(zero.mean()), "zeros": int(zero.sum()),
          "good_primes_2_mod_3": int(np.sum(p3 % 3 == 2)), "zeros_among_2_mod_3": int(np.sum(zero & (p3 % 3 == 2))),
          "good_primes_1_mod_3": int(np.sum(p3 % 3 == 1)), "zeros_among_1_mod_3": int(np.sum(zero & (p3 % 3 == 1))),
          "even_moments_off_the_atom": [float(np.mean(x3[nz] ** k)) for k in (2, 4, 6)],
          "half_of_diagonal_moments": [0.5 * mom_x[k][2] for k in (2, 4, 6)]}
    score = [0.5 * L0[g] + 0.5 * integ(FQ[2] * LQ[g], F0[2] * L0[g]) for g in range(3)]
    c3["limit_law_score_by_group"] = score
    c3["predicted_rate_vs_USp4"] = [s - score[0] for s in score]
    L3 = LL["C3"]; d3 = L3[1:] - L3[0]
    c3["observed_rate_vs_USp4_unclipped"] = [0.0] + [float(v) for v in d3.mean(1)]
    win = int(np.argmax(score))
    ent = integ(FQ[win] * LQ[win], F0[win] * L0[win])             # E log f under the winner's own law
    var = integ(FQ[win] * LQ[win] ** 2, F0[win] * L0[win] ** 2) - ent ** 2
    se = math.sqrt(var / len(a3)); obs = float(L3[win].mean())
    c3.update({"winner_by_score": GROUPS[win], "mean_loglik_observed_under_winner": obs, "expected_loglik_under_winner": ent,
               "sd_of_log_f_under_winner": math.sqrt(var), "standard_error": se, "gap_in_standard_errors": (obs - ent) / se})
    out["C3"] = c3
    check("a_p = 0 at every good p = 2 mod 3, exceptions", c3["good_primes_2_mod_3"] - c3["zeros_among_2_mod_3"], 0)
    check("a_p = 0 at no good p = 1 mod 3, exceptions", c3["zeros_among_1_mod_3"], 0)

    # ---- 6. chapter against script -----------------------------------------------------------------------------
    rows = []

    def row(item, where, printed, value, mode="round", ref=None, tol=None):
        """One printed number. round: value rounds to the printed digits; eq, in: strings; gt, ge, le: value against ref;
        near: |value - ref| <= tol (for phrases such as 'about 120')."""
        if mode == "round":
            d = len(printed.split(".")[1]) if "." in printed else 0
            ok = f"{value:.{d}f}" == printed
        elif mode == "eq": ok = str(value) == printed
        elif mode == "in": ok = printed in str(value)
        elif mode == "gt": ok = value > ref
        elif mode == "ge": ok = value >= ref
        elif mode == "le": ok = value <= ref
        elif mode == "near": ok = abs(value - ref) <= tol
        else: raise ValueError(mode)
        rows.append({"item": item, "where": where, "chapter": printed, "script": value if isinstance(value, str) else float(value), "ok": bool(ok)})

    cv = out["curves"]; U, S, D = 0, 1, 2
    W = "Three curves"
    for c, n in zip(CURVES, ("9589", "9590", "9590")): row(f"number of primes, {c}", W, n, cv[c]["n"])
    for g, (a, b, c) in enumerate((("1", "3", "14"), ("2", "10", "70"), ("4", "32", "320"))):
        for k, s in zip((2, 4, 6), (a, b, c)): row(f"moment {k} of {GROUPS[g]}", "Genus two", s, mom_q[k][g])
    for g, s in enumerate(("0.432", "0.270", "0.159")): row(f"f(0) of {GROUPS[g]}", "Misspecification", s, F0[g])
    row("edge exponent of the USp(4) density", "Genus two", "4", slope[0])
    row("edge exponent of the diagonal density", "Genus two", "0.5", slope[2])
    for (a, b), s in (((0, 1), "0.117"), ((0, 2), "0.456"), ((1, 0), "0.168"), ((1, 2), "0.161"), ((2, 0), "1.061"), ((2, 1), "0.262")):
        row(f"KL({GROUPS[a]} || {GROUPS[b]})", "tab:12kl", s, KL[a][b])
    row("KL(diagonal || USp(4)) over KL(USp(4) || diagonal), more than twice", "Genus two", "more than 2", KL[2][0] / KL[0][2], "gt", ref=2.0)
    row("primes needed at the smallest divergence, about 120", "Sequential testing", "about 120", seq["primes_for_smallest_divergence"], "near", ref=120.0, tol=5.0)
    c1, c2 = cv["C1"], cv["C2"]
    for k, s in zip(range(3), ("1.020", "3.166", "15.40")): row(f"C1 even moment {2 * (k + 1)}", W + ", C1", s, c1["even_moments"][k])
    row("C1 moments within about two standard errors of USp(4), max |z|", W + ", C1", "about 2", max(abs(v) for v in c1["moment_z_scores_by_group"][U]), "le", ref=2.5)
    for k, s in zip(("10", "50", "100"), ("0.426", "0.984", "0.99998")): row(f"C1 posterior of USp(4) after {k} primes", W + ", C1", s, c1["posterior_clipped"][k][U])
    row("C1 rate of SU(2)xSU(2) against USp(4)", W + ", C1", "-0.111", c1["rate_vs_USp4_clipped"][S])
    row("C1 theoretical rate -KL(USp(4) || SU(2)xSU(2))", W + ", C1", "-0.117", -KL[U][S])
    row("C1 rate of the diagonal group against USp(4)", W + ", C1", "-0.447", c1["rate_vs_USp4_clipped"][D])
    row("C1 theoretical rate -KL(USp(4) || diagonal)", W + ", C1", "-0.456", -KL[U][D])
    row("Galois group of x^5 - x + 1", W + ", C1", "S5", out["C1_galois_group"], "in")
    sp = out["C2_split"]
    row("j invariant of E", W + ", C2", "-13824/19", sp["j_E"], "eq")
    row("j invariant of E'", W + ", C2", "32768/19", sp["j_Eprime"], "eq")
    row("a_5(E)", W + ", C2", "1", sp["a5_E"]); row("a_5(E')", W + ", C2", "-3", sp["a5_Eprime"])
    row("a_p(C2) = a_p(E) + a_p(E'), mismatches over all good primes", W + ", C2", "0", sp["mismatches_ap_C2_vs_ap_E_plus_ap_Eprime"])
    for k, s in zip(range(3), ("1.999", "9.91", "68.3")): row(f"C2 even moment {2 * (k + 1)}", W + ", C2", s, c2["even_moments"][k])
    row("C2 posterior of SU(2)xSU(2) after 50 primes", W + ", C2", "0.998", c2["posterior_clipped"]["50"][S])
    row("C2 rate of SU(2)xSU(2) against USp(4)", W + ", C2", "0.167", c2["rate_vs_USp4_clipped"][S])
    row("C2 theoretical rate KL(SU(2)xSU(2) || USp(4))", W + ", C2", "0.168", KL[S][U])
    c3r = cv["C3"]
    for k, s in zip(range(3), ("1.98", "15.8", "158.8")): row(f"C3 even moment {2 * (k + 1)}", W + ", C3", s, c3r["even_moments"][k])
    row("C3 mass of the atom at 0", W + ", C3", "0.501", c3["atom_mass"])
    row("C3 exceptions to a_p = 0 for p = 2 mod 3", W + ", C3", "0", c3["good_primes_2_mod_3"] - c3["zeros_among_2_mod_3"])
    row("C3 zeros of a_p at primes p = 1 mod 3", W + ", C3", "0", c3["zeros_among_1_mod_3"])
    for k, s in zip(range(3), ("3.96", "31.7", "318")): row(f"C3 even moment {2 * (k + 1)} off the atom", W + ", C3", s, c3["even_moments_off_the_atom"][k])
    for k, s in zip(range(3), ("2", "16", "160")): row(f"half of the diagonal moment {2 * (k + 1)}", W + ", C3", s, c3["half_of_diagonal_moments"][k])
    row("C3 posterior of SU(2)xSU(2) after 100 primes, exceeds", "Misspecification", "more than 0.99", c3r["posterior_clipped"]["100"][S], "gt", ref=0.99)
    row("C3 winner", "Misspecification", GROUPS[S], c3["winner_by_score"], "eq")
    row("C3 winner by posterior", "fig:12posterior", GROUPS[S], c3r["winner"], "eq")
    row("C3 posterior stays above 0.99 from about 100 primes", "fig:12posterior", "about 100", c3r["n_after_which_winner_stays_above_0_99"], "near", ref=100.0, tol=30.0)
    row("C3 predicted rate of SU(2)xSU(2) against USp(4)", "Misspecification", "0.164", c3["predicted_rate_vs_USp4"][S])
    row("C3 predicted rate of the diagonal group against USp(4)", "Misspecification", "0.031", c3["predicted_rate_vs_USp4"][D])
    row("C3 observed rate of SU(2)xSU(2) against USp(4)", "Misspecification", "0.158", c3["observed_rate_vs_USp4_unclipped"][S])
    row("C3 observed rate of the diagonal group against USp(4)", "Misspecification", "0.021", c3["observed_rate_vs_USp4_unclipped"][D])
    row("C3 observed rates after clipping at 20, SU(2)xSU(2)", "Misspecification", "0.158", c3r["rate_vs_USp4_clipped"][S])
    row("C3 observed rates after clipping at 20, diagonal", "Misspecification", "0.021", c3r["rate_vs_USp4_clipped"][D])
    row("C3 mean log likelihood of the winner", "Misspecification", "-1.80", c3["mean_loglik_observed_under_winner"])
    row("C3 expected log likelihood under the winning law", "Misspecification", "-1.75", c3["expected_loglik_under_winner"])
    row("C3 gap in standard errors, about seven", "Misspecification", "about 7", abs(c3["gap_in_standard_errors"]), "near", ref=7.0, tol=0.5)
    zz = [max(abs(v) for v in c3r["moment_z_scores_by_group"][g]) for g in range(3)]
    row("C3 moments off every candidate, smallest of the largest |z|, dozens", "Misspecification", "dozens", min(zz), "ge", ref=24.0)
    out["chapter_check"] = rows
    out["all_chapter_numbers_reproduced"] = bool(all(r["ok"] for r in rows)) and not QUICK
    out["checks"] = checks
    out["timing_seconds"] = time.time() - t0
    out["log"] = stamps

    w = max(len(r["item"]) for r in rows)
    print("\nchapter value against script value")
    for r in rows:
        print(f"{'ok ' if r['ok'] else 'BAD'} {r['item']:<{w}}  chapter {r['chapter']:>14}  script {r['script']}")
    print("\ninternal checks")
    for k, v in checks.items(): print(f"{'ok ' if v['ok'] else 'BAD'} {k}: {v['value']:.3g} (limit {v['limit']})")
    print(f"\n{sum(r['ok'] for r in rows)} of {len(rows)} chapter numbers reproduced, {sum(v['ok'] for v in checks.values())} of {len(checks)} internal checks passed, {out['timing_seconds']:.0f} s")
    with open("stage4c.json", "w") as fh: fh.write(dumps(out) + "\n")
    log("wrote stage4c.json")


if __name__ == "__main__":
    main()
