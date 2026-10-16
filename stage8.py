import numpy as np, math, json, cypari2
pari = cypari2.Pari(); pari.allocatemem(2 * 10**9)
out = {}
# ---------- Duke: Heegner points equidistribute for (3/pi) dx dy / y^2 ----------
def heegner(D):
    N = -D; pts = []
    amax = int(math.isqrt(N // 3)) + 1
    for a in range(1, amax + 1):
        for b in range(-a + 1, a + 1):
            if (b * b + N) % (4 * a): continue
            c = (b * b + N) // (4 * a)
            if c < a: continue
            if c == a and b < 0: continue
            if math.gcd(math.gcd(a, abs(b)), c) != 1: continue
            pts.append((-b / (2 * a), math.sqrt(N) / (2 * a)))
    return pts
Ds = [d for d in range(-10**6 - 3000, -10**6, 1) if int(pari(f"isfundamental({d})"))][:60]
allp = []; hs = []
for D in Ds:
    P = heegner(D); hs.append((D, len(P), int(pari(f"qfbclassno({D})")))); allp += P
allp = np.array(allp)
y = allp[:, 1]; x = allp[:, 0]
tail = [(t, float(np.mean(y > t)), 3 / (math.pi * t)) for t in (1.0, 1.5, 2, 3, 5, 10)]
# below y = 1 the region is {|x|<=1/2, x^2+y^2>=1}; test x distribution conditional on y > 1 (uniform)
xs = x[y > 1]
from scipy.stats import kstest
out["duke"] = {"n_discriminants": len(Ds), "points": int(len(allp)), "class_number_check": all(a == b for _, a, b in hs),
               "tail": tail, "ks_x_given_y_gt_1_p": float(kstest(xs, "uniform", args=(-0.5, 1)).pvalue),
               "D_range": [min(Ds), max(Ds)]}
np.save("heegner.npy", allp)
print(out["duke"], flush=True)

# ---------- location scale family with the Delta Polya kernel shape: right vs left Haar coverage ----------
us, ph = np.load("phi.npy"); du = us[1] - us[0]
cdf = np.cumsum(ph) * du; cdf /= cdf[-1]
logf = lambda z: np.log(np.maximum(np.interp(z, us, ph, left=0, right=0), 1e-300))
rng = np.random.default_rng(2026)
def sample(n): return np.interp(rng.uniform(size=n), cdf, us)
n = 5; reps = 2000; level = 0.90
mu0, sig0 = 1.3, 0.7
mus = np.linspace(-1.5, 1.5, 151); ls = np.linspace(-2.5, 1.5, 161); sg = np.exp(ls)
cover = {"right": [0, 0], "left": [0, 0]}
for r in range(reps):
    yv = mu0 + sig0 * sample(n)
    m = np.median(yv); s = np.std(yv) + 1e-3
    MU = m + s * mus[:, None]; SG = s * sg[None, :]          # equivariant grid around the data
    ll = sum(logf((yi - MU) / SG) for yi in yv) - n * np.log(SG)
    for name, pw in (("right", 1), ("left", 2)):
        # prior d mu d sigma / sigma^pw; grid in log sigma adds a Jacobian sigma
        lp = ll - pw * np.log(SG) + np.log(SG)
        w = np.exp(lp - lp.max()); w /= w.sum()
        # marginal of sigma and of mu
        wsig = w.sum(0); wmu = w.sum(1)
        cs = np.cumsum(wsig); cm = np.cumsum(wmu)
        lo_s, hi_s = SG[0, np.searchsorted(cs, (1 - level) / 2)], SG[0, np.searchsorted(cs, (1 + level) / 2)]
        lo_m, hi_m = MU[np.searchsorted(cm, (1 - level) / 2), 0], MU[np.searchsorted(cm, (1 + level) / 2), 0]
        cover[name][0] += lo_s <= sig0 <= hi_s; cover[name][1] += lo_m <= mu0 <= hi_m
out["coverage"] = {"n": n, "reps": reps, "level": level,
                   "right_haar": {"sigma": cover["right"][0] / reps, "mu": cover["right"][1] / reps},
                   "left_haar": {"sigma": cover["left"][0] / reps, "mu": cover["left"][1] / reps},
                   "mc_se": math.sqrt(level * (1 - level) / reps)}
print(out["coverage"], flush=True)
json.dump(out, open("stage8.json", "w"), indent=1)
