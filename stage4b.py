import cypari2, numpy as np, json, math
pari = cypari2.Pari(); pari.allocatemem(2 * 10**9)
out = {}
X = 100000
curves = {"C1: y^2 = x^5 - x + 1": "x^5-x+1", "C2: y^2 = x^6 + 3x^4 + x^2 + 1": "x^6+3*x^4+x^2+1",
          "C3: y^2 = x^6 + x^3 + 1": "x^6+x^3+1"}
data = {}
def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]
coef = {"x^5-x+1": [1, 0, 0, 0, -1, 1], "x^6+3*x^4+x^2+1": [1, 0, 3, 0, 1, 0, 1], "x^6+x^3+1": [1, 0, 0, 1, 0, 0, 1]}
for name, f in curves.items():
    c = coef[f]; disc = int(pari(f"poldisc({f})")); res = []
    for p in primes_upto(X)[1:]:
        p = int(p)
        if disc % p == 0: continue
        x = np.arange(p, dtype=np.int64); v = np.zeros(p, dtype=np.int64)
        for a in c: v = (v * x + a) % p
        sq = np.zeros(p, np.int8); sq[(x * x) % p] = 1
        chi = np.where(v == 0, 0, np.where(sq[v] == 1, 1, -1))
        inf = 1 if len(c) == 6 else (2 if sq[c[0] % p] else 0)
        npts = p + int(chi.sum()) + inf
        res.append((p, p + 1 - npts))
    res = np.array(res); data[name] = res[:, 1] / np.sqrt(res[:, 0])
# trace densities on [-4,4] for three Sato Tate groups
g = np.linspace(-2, 2, 2001); dx = g[1] - g[0]
sc = np.sqrt(np.clip(4 - g ** 2, 0, None))
X1, X2 = np.meshgrid(g, g, indexing="ij")
W = {"USp(4)": (X1 - X2) ** 2 * np.outer(sc, sc), "SU(2)xSU(2)": np.outer(sc, sc)}
tg = np.linspace(-4, 4, 4001)
dens = {}
for k, w in W.items():
    w = w / (w.sum() * dx * dx)
    h, edges = np.histogram((X1 + X2).ravel(), bins=tg, weights=w.ravel() * dx * dx)
    dens[k] = h / np.diff(edges)
mid = (tg[1:] + tg[:-1]) / 2
dens["SU(2) diagonal"] = np.where(np.abs(mid) < 4, np.sqrt(np.clip(4 - (mid / 2) ** 2, 0, None)) / (2 * np.pi) / 2, 0)
for k in dens: dens[k] = dens[k] / (dens[k].sum() * (mid[1] - mid[0]))
def logf(k, t):
    return np.log(np.maximum(np.interp(t, mid, dens[k]), 1e-300))
groups = list(dens)
# theoretical second and fourth moments
out["theory_moments"] = {k: [float((mid ** m * dens[k]).sum() * (mid[1] - mid[0])) for m in (2, 4, 6)] for k in groups}
M = 20
res = {}
for name, x in data.items():
    ll = np.array([np.clip(logf(k, x), -M - 50, None) for k in groups])
    # pairwise clipped log Bayes factors relative to the best group, and posterior with uniform prior
    tot = {k: float(np.sum(np.clip(ll[i] - ll[0], -M, M))) for i, k in enumerate(groups)}
    cum = np.cumsum(ll, axis=1)
    post_n = {}
    for nn in (10, 50, 200, len(x)):
        c = cum[:, nn - 1]; p = np.exp(c - c.max()); p /= p.sum()
        post_n[nn] = {k: float(v) for k, v in zip(groups, p)}
    res[name] = {"n": int(len(x)), "moments": [float(np.mean(x ** m)) for m in (2, 4, 6)], "posterior": post_n,
                 "rate_vs_USp4": {k: tot[k] / len(x) for k in groups}}
# KL divergences between groups (the theoretical exponents)
kl = {}
for a in groups:
    for b in groups:
        if a == b: continue
        pa, pb = dens[a], dens[b]; m = pa > 1e-12
        kl[f"{a}||{b}"] = float(np.sum(pa[m] * np.clip(np.log(pa[m] / np.maximum(pb[m], 1e-300)), -M, M)) * (mid[1] - mid[0]))
out["kl"] = kl; out["curves"] = res
np.save("/home/claude/langlands/g2dens.npy", np.vstack([mid] + [dens[k] for k in groups]))
for name, x in data.items(): np.save(f"/home/claude/langlands/g2_{name[:2]}.npy", x)
json.dump(out, open("/home/claude/langlands/stage4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
