# Kloosterman sums: Katz vertical Sato Tate and Kowalski Sawin paths
import numpy as np, json, math
from scipy.stats import ks_2samp, kstest
out = {}; rng = np.random.default_rng(5)
def kloosterman_all(p):
    x = np.arange(1, p); inv = np.array([pow(int(v), p - 2, p) for v in x])
    K = np.zeros(p - 1)
    for i in range(0, p - 1, 200):
        a = np.arange(1 + i, min(p, 1 + i + 200))[:, None]
        K[i:i + len(a)] = np.cos(2 * np.pi * ((x[None, :] + a * inv[None, :]) % p) / p).sum(1)
    return K / math.sqrt(p), inv
def F_st(x):
    t = np.arccos(np.clip(x / 2, -1, 1)); return 1 - (t - np.sin(2 * t) / 2) / np.pi
res = {}
for p in (1009, 10007, 100003):
    if p > 20000:
        # sample of a only
        x = np.arange(1, p); inv = np.array([pow(int(v), p - 2, p) for v in x])
        A = rng.choice(np.arange(1, p), 3000, replace=False)
        K = np.array([np.cos(2 * np.pi * ((x + a * inv) % p) / p).sum() for a in A]) / math.sqrt(p)
    else:
        K, inv = kloosterman_all(p)
    res[p] = {"n": int(len(K)), "max_abs": float(np.abs(K).max()), "weil_bound": 2.0,
              "moments": [float(np.mean(K ** (2 * m))) for m in range(1, 5)], "catalan": [1, 2, 5, 14],
              "ks_vs_sato_tate_p": float(kstest(K, F_st).pvalue)}
    if p == 10007: np.save("kloost_10007.npy", K)
    print(p, res[p], flush=True)
out["katz"] = res
# Kowalski Sawin paths for p = 10007
p = 10007; x = np.arange(1, p); inv = np.array([pow(int(v), p - 2, p) for v in x])
A = rng.choice(np.arange(1, p), 2000, replace=False)
tq = [0.25, 0.5, 0.75]; vals = {t: [] for t in tq}; keep = []
for k, a in enumerate(A):
    path = np.concatenate([[0], np.cumsum(np.exp(2j * np.pi * ((x + a * inv) % p) / p))]) / math.sqrt(p)
    for t in tq: vals[t].append(path[int(round(t * (p - 1)))])
    if k < 40: keep.append(path[::5])
np.save("kloost_paths.npy", np.array(keep))
# random Fourier series K(t) = t ST_0 + sum_{h != 0} (e(ht) - 1)/(2 pi i h) ST_h, ST_h iid Sato Tate on [-2,2]
def st_sample(size):
    out_ = np.empty(size); k = 0
    while k < size:
        th = rng.uniform(0, np.pi, 2 * size); u = rng.uniform(size=2 * size); acc = th[u < np.sin(th) ** 2]
        take = min(len(acc), size - k); out_[k:k + take] = 2 * np.cos(acc[:take]); k += take
    return out_
H = 3000; hs = np.concatenate([np.arange(-H, 0), np.arange(1, H + 1)]); R = 4000
ST = st_sample(R * (2 * H + 1)).reshape(R, 2 * H + 1)
rfs = {}
for t in tq:
    coef = (np.exp(2j * np.pi * hs * t) - 1) / (2j * np.pi * hs)
    rfs[t] = t * ST[:, H] + ST[:, np.r_[0:H, H + 1:2 * H + 1]] @ coef
ks = {}
for t in tq:
    a = np.array(vals[t]); b = rfs[t]
    ks[str(t)] = {"re_p": float(ks_2samp(a.real, b.real).pvalue), "im_p": float(ks_2samp(a.imag, b.imag).pvalue),
                  "var_re": [float(a.real.var()), float(b.real.var())], "var_im": [float(a.imag.var()), float(b.imag.var())]}
out["kowalski_sawin"] = {"p": p, "paths": 2000, "rfs_samples": R, "H": H, "tests": ks}
# a few random Fourier series paths for the figure
tt = np.linspace(0, 1, 801); coefs = (np.exp(2j * np.pi * np.outer(tt, hs[np.abs(hs) <= 400])) - 1) / (2j * np.pi * hs[np.abs(hs) <= 400])
sel = np.r_[H - 400:H, H + 1:H + 401]
rf_paths = np.outer(ST[:12, H], tt).T.T + (ST[:12][:, sel] @ coefs.T)
np.save("rfs_paths.npy", rf_paths)
print(out["kowalski_sawin"], flush=True)
json.dump(out, open("stage9d.json", "w"), indent=1)
print("done")
