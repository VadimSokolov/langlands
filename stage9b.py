# Random models: Sato Tate random Euler product as a log correlated field; Harper's random multiplicative functions
import numpy as np, json, math, time
from scipy import sparse
out = {}; rng = np.random.default_rng(99)
def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]
def st_angles(k):
    # Sato Tate: density (2/pi) sin^2 on [0, pi], by rejection
    outa = []
    while len(outa) < k:
        th = rng.uniform(0, math.pi, 2 * k); u = rng.uniform(size=2 * k)
        outa.extend(th[u < np.sin(th) ** 2].tolist())
    return np.array(outa[:k])
P = primes_upto(10**6); logp = np.log(P)
# 1. field X(t) = sum_p 2 cos(theta_p) cos(t log p)/sqrt p on t in [0,1], fine grid
tg = np.linspace(0, 1, 2001)
RP = {10**3: 300, 10**4: 200, 10**5: 80, 10**6: 30}; maxima = {}; Xs = []
for X in (10**3, 10**4, 10**5, 10**6):
    m = P <= X; lp = logp[m]; w = 1 / np.sqrt(P[m])
    vals = []
    for r in range(RP[X]):
        th = st_angles(m.sum())
        # chunked evaluation
        f = np.zeros_like(tg)
        for i in range(0, len(lp), 4000):
            f += np.cos(np.outer(tg, lp[i:i+4000])) @ (2 * np.cos(th[i:i+4000]) * w[i:i+4000])
        vals.append(f.max())
        if X == 10**6 and r < 3: Xs.append(f)
    ll = math.log(math.log(X))
    maxima[X] = {"mean_max": float(np.mean(vals)), "sd_max": float(np.std(vals)), "loglog": ll, "fhk": ll - 0.75 * math.log(ll)}
    print(X, maxima[X], flush=True)
out["random_field_maxima"] = maxima
np.save("rf_paths.npy", np.vstack([tg] + Xs))
# branching random walk picture: partial sums over primes up to e^{e^k}
th = st_angles((P <= 10**6).sum()); tg2 = np.linspace(0, 1, 2001)
levels = [math.exp(math.exp(k / 2)) for k in range(1, 6)] + [10**6]
parts = []
for Lv in levels:
    m = P <= Lv
    parts.append(np.cos(np.outer(tg2, logp[m])) @ (2 * np.cos(th[:m.sum()]) / np.sqrt(P[m])))
np.save("brw_levels.npy", np.vstack([tg2] + parts)); out["brw_levels"] = levels
# 2. Harper: Steinhaus random multiplicative function, E|S(x)|/sqrt(x)
N = 10**6
spf = np.zeros(N + 1, dtype=np.int64)
for p in primes_upto(int(N ** .5)):
    idx = np.arange(p, N + 1, p); sel = spf[idx] == 0; spf[idx[sel]] = p
nz = spf == 0; spf[nz] = np.arange(N + 1)[nz]
pi_idx = {int(p): i for i, p in enumerate(primes_upto(N))}
rows, cols, vals_ = [], [], []
for n in range(2, N + 1):
    m = n
    while m > 1:
        p = int(spf[m]); rows.append(n); cols.append(pi_idx[p]); vals_.append(1.0); m //= p
A = sparse.csr_matrix((vals_, (rows, cols)), shape=(N + 1, len(pi_idx)))
xs = [10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6]
R = 400; acc = {x: [] for x in xs}; acc_r = {x: [] for x in xs}
for r in range(R):
    phase = A @ rng.uniform(0, 2 * math.pi, len(pi_idx))
    f = np.exp(1j * phase); f[0] = 0
    S = np.cumsum(f)
    for x in xs: acc[x].append(abs(S[x]))
    # Rademacher (random signs on primes, supported on squarefree) for comparison
out["harper"] = {str(x): {"E|S|/sqrt x": float(np.mean(acc[x]) / math.sqrt(x)), "loglog^-1/4": float(math.log(math.log(x)) ** -0.25),
                          "ratio": float(np.mean(acc[x]) / math.sqrt(x) / math.log(math.log(x)) ** -0.25)} for x in xs}
print(out["harper"], flush=True)
json.dump(out, open("stage9b.json", "w"), indent=1)
print("done")
