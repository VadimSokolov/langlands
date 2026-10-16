import cypari2, numpy as np, json, math, time
def arr(expr):
    t = str(pari(f"Vecsmall({expr})"))
    return np.array(t[t.index("[")+1:t.rindex("]")].split(","), dtype=np.int64)
from scipy.special import kolmogorov
pari = cypari2.Pari(); pari.allocatemem(4 * 10**9)
out = {}
X = 10**7
t0 = time.time()

def traces(coeffs, X):
    pari(f"E=ellinit({coeffs}); P=primes([3,{X}]); V=vector(#P,i,ellap(E,P[i]));")
    P = arr("P"); V = arr("V")
    return P, V

P1, V1 = traces("[0,0,0,1,1]", X); good = P1 != 31; P1, V1 = P1[good], V1[good]
P2, V2 = traces("[0,0,0,-1,0]", X)
out["n_E1"] = int(len(P1)); out["time_traces"] = time.time() - t0
x1 = V1 / np.sqrt(P1); x2 = V2 / np.sqrt(P2)

def F_st(x):
    t = np.arccos(np.clip(x / 2, -1, 1)); return 1 - (t - np.sin(2 * t) / 2) / np.pi
def Un(n, x):
    t = np.arccos(np.clip(x / 2, -1, 1)); s = np.sin(t)
    return np.sin((n + 1) * t) / np.where(s > 1e-12, s, 1e-12)
cat = [math.comb(2*m, m) // (m + 1) for m in range(7)]
out["moments_E1"] = [(2*m, float(np.mean(x1 ** (2*m))), cat[m]) for m in range(1, 7)]
out["moments_E2"] = [(2*m, float(np.mean(x2 ** (2*m))), 0.5 * math.comb(2*m, m)) for m in range(1, 7)]
out["weyl_E1"] = [(n, float(np.mean(Un(n, x1)))) for n in range(1, 13)]
out["weyl_E2"] = [(n, float(np.mean(Un(n, x2)))) for n in range(1, 13)]
out["zero_frac_E2"] = float(np.mean(V2 == 0))

# discrepancy path for E1
ks = []
for k in [10**2, 10**3, 10**4, 10**5, len(x1)]:
    s = np.sort(x1[:k]); e = np.arange(1, k + 1) / k
    d = float(max(np.max(np.abs(e - F_st(s))), np.max(np.abs(e - 1 / k - F_st(s)))))
    ks.append((k, d, d * math.sqrt(k)))
out["ks_E1"] = ks

# Lang Trotter with pi_{1/2}(x) = int_2^x dt / (2 sqrt t log t)
from scipy.integrate import quad
lt = []
for Xc in [10**4, 10**5, 10**6, 10**7]:
    c = int(np.sum((V1 == 0) & (P1 <= Xc)))
    pih = quad(lambda t: 1 / (2 * math.sqrt(t) * math.log(t)), 2, Xc, limit=500)[0]
    lt.append((Xc, c, pih, c / pih))
out["lang_trotter"] = lt

# Ramanujan tau Sato Tate
pari(f"T=mfcoefs(mfDelta(),{10**6}); PP=primes([2,{10**6}]); tv=vector(#PP,i,T[PP[i]+1]);")
PP = arr("PP"); tv = np.array([float(z) for z in pari("tv")])
xt = tv / PP ** 5.5
out["n_tau"] = int(len(xt)); out["tau_max_abs"] = float(np.max(np.abs(xt)))
out["moments_tau"] = [(2*m, float(np.mean(xt ** (2*m))), cat[m]) for m in range(1, 7)]
s = np.sort(xt); e = np.arange(1, len(s) + 1) / len(s)
out["ks_tau"] = float(max(np.max(np.abs(e - F_st(s))), np.max(np.abs(e - 1 / len(s) - F_st(s))))) * math.sqrt(len(s))  # both ECDF steps (fixed 2026-10-09; stage1.json holds the old upper step value 0.6424, the two sided value is 0.6460)
np.save("../work/xt.npy", xt)

# rank bias: sum_{p<=x} a_p log p / p vs (1/2 - r) log x
curves = {"11a1": ("[0,-1,1,-10,-20]", 0), "37a1": ("[0,0,1,-1,0]", 1), "389a1": ("[0,1,1,-2,0]", 2), "5077a1": ("[0,0,1,-7,6]", 3)}
bias = {}
for name, (c, r) in curves.items():
    P, V = traces(c, X)
    w = V * np.log(P) / P
    cs = np.cumsum(w)
    pts = []
    for Xc in [10**3, 10**4, 10**5, 10**6, 10**7]:
        i = np.searchsorted(P, Xc, side="right") - 1
        pts.append((Xc, float(cs[i]), float((0.5 - r) * math.log(Xc))))
    rank = int(pari(f"ellanalyticrank(ellinit({c}))[1]"))
    bias[name] = {"rank": rank, "pts": pts}
    xs = np.unique(np.logspace(2, 7, 300).astype(int)); idx = np.searchsorted(P, xs, side="right") - 1
    bias[name]["curve"] = [(int(a), float(cs[b])) for a, b in zip(xs, idx) if b >= 0]
out["bias"] = bias

# discrepancy across 200 curves without CM, primes up to 10^6
rng = np.random.default_rng(7); vals = []; tried = 0
Xd = 10**6
while len(vals) < 200:
    a, b = int(rng.integers(-50, 51)), int(rng.integers(-50, 51))
    if 4 * a**3 + 27 * b**2 == 0: continue
    # CM check via j-invariant in the list of 13 rational CM j invariants
    cmj = [0, 1728, -3375, 8000, -32768, 54000, 287496, -884736, -12288000, 16581375, -884736000, -147197952000, -262537412640768000]
    if pari(f"denominator(ellinit([{a},{b}]).j)") == 1 and int(pari(f"ellinit([{a},{b}]).j")) in cmj: continue
    D = int(pari(f"ellinit([{a},{b}]).disc"))
    pari(f"E=ellinit([{a},{b}]); P=primes([5,{Xd}]); V=vector(#P,i,ellap(E,P[i]));")
    P = arr("P"); V = arr("V")
    g = (D % P) != 0
    x = np.sort(V[g] / np.sqrt(P[g])); n = len(x); e = np.arange(1, n + 1) / n
    d = max(np.max(np.abs(e - F_st(x))), np.max(np.abs(e - 1 / n - F_st(x))))
    vals.append(float(d * math.sqrt(n)))
vals = np.array(vals)
from scipy.stats import kstest
out["cross_curve"] = {"n": len(vals), "mean": float(vals.mean()), "sd": float(vals.std()), "kolm_mean": 0.8687, "kolm_sd": 0.2603,
                      "ks_vs_kolmogorov_p": float(kstest(vals, lambda z: 1 - kolmogorov(z)).pvalue),
                      "quantiles": [float(q) for q in np.quantile(vals, [.1, .5, .9])]}
np.save("../work/cross.npy", vals)
out["time_total"] = time.time() - t0
json.dump(out, open("../work/stage1.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "bias"}, indent=1))
for k, v in out["bias"].items(): print(k, v["rank"], v["pts"])
