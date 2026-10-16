import cypari2, numpy as np, json, math
pari = cypari2.Pari(); pari.allocatemem(2 * 10**9)
out = {}
X = 30000
curves = {"C1: y^2 = x^5 - x + 1": "x^5-x+1", "C2: y^2 = x^6 + 3x^4 + x^2 + 1": "x^6+3*x^4+x^2+1",
          "C3: y^2 = x^6 + x^3 + 1": "x^6+x^3+1"}
data = {}
for name, f in curves.items():
    pari(f"f={f}; D=poldisc(f); P=select(p->D%p, primes([3,{X}])); V=vector(#P,i,-polcoef(hyperellcharpoly(Mod(1,P[i])*f),3));")
    P = np.array([int(z) for z in pari("P")]); V = np.array([int(z) for z in pari("V")])
    data[name] = V / np.sqrt(P)

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
np.save("../work/g2dens.npy", np.vstack([mid] + [dens[k] for k in groups]))
for name, x in data.items(): np.save(f"../work/g2_{name[:2]}.npy", x)
json.dump(out, open("../work/stage4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
