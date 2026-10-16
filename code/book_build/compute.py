import numpy as np, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams.update({"font.family": "serif", "font.size": 10})
out = {}
N = 40000

def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(N)

def ap_curve(a, b, bad):
    res = []
    for p in P:
        p = int(p)
        if p in bad: continue
        x = np.arange(p, dtype=np.int64)
        f = (x * x % p * x + a * x + b) % p
        sq = np.zeros(p, np.int8); sq[(x * x) % p] = 1
        leg = np.where(f == 0, 0, np.where(sq[f] == 1, 1, -1))
        res.append((p, -int(leg.sum())))
    return np.array(res)

E1 = ap_curve(1, 1, {2, 31})      # y^2 = x^3 + x + 1, no CM
E2 = ap_curve(-1, 0, {2})         # y^2 = x^3 - x, CM by Z[i]
assert np.all(np.abs(E1[:, 1]) <= 2 * np.sqrt(E1[:, 0]))   # Hasse bound check
x1 = E1[:, 1] / np.sqrt(E1[:, 0]); x2 = E2[:, 1] / np.sqrt(E2[:, 0])
out["nprimes_E1"] = len(x1); out["nprimes_E2"] = len(x2)

# moments vs Catalan numbers (semicircle on [-2,2]) and CM law
cat = [math.comb(2*m, m) // (m + 1) for m in range(6)]
cm = [1] + [0.5 * math.comb(2*m, m) for m in range(1, 6)]
out["moments"] = [(2*m, float(np.mean(x1 ** (2*m))), cat[m], float(np.mean(x2 ** (2*m))), cm[m]) for m in range(1, 6)]
out["frac_zero_E2"] = float(np.mean(E2[:, 1] == 0))
out["frac_zero_E1"] = float(np.mean(E1[:, 1] == 0))
# Weyl criterion: averages of U_n(x/2) = sin((n+1)t)/sin t
def Un(n, x):
    t = np.arccos(np.clip(x / 2, -1, 1)); s = np.sin(t)
    return np.where(s > 1e-12, np.sin((n + 1) * t) / np.where(s > 1e-12, s, 1), (n + 1) * np.sign(np.cos(t)) ** n)
out["weyl_E1"] = [(n, float(np.mean(Un(n, x1)))) for n in range(1, 9)]
out["weyl_E2"] = [(n, float(np.mean(Un(n, x2)))) for n in range(1, 9)]

# Kolmogorov distance of E1 to Sato Tate as primes grow
def F_st(x):
    t = np.arccos(np.clip(x / 2, -1, 1))
    return 1 - (t - np.sin(2 * t) / 2) / np.pi
ks = []
for k in [100, 300, 1000, 3000, len(x1)]:
    s = np.sort(x1[:k]); e = np.arange(1, k + 1) / k
    ks.append((k, float(np.max(np.abs(e - F_st(s))))))
out["ks_E1"] = ks

xs = np.linspace(-2, 2, 801)
st = np.sqrt(np.clip(4 - xs ** 2, 0, None)) / (2 * np.pi)
arc = 1 / (np.pi * np.sqrt(np.clip(4 - xs ** 2, 1e-9, None)))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].hist(x1, bins=50, density=True, color="#9fb7d4", edgecolor="white")
ax[0].plot(xs, st, color="#1f3b63"); ax[0].set_title(r"$y^2=x^3+x+1$, no CM")
ax[0].set_xlabel(r"$a_p/\sqrt{p}$")
nz = x2[E2[:, 1] != 0]
ax[1].hist(nz, bins=50, density=True, weights=np.full(len(nz), len(nz) / len(x2)), color="#d4b59f", edgecolor="white")
ax[1].plot(xs, 0.5 * arc, color="#63381f"); ax[1].set_ylim(0, 1.2)
ax[1].set_title(r"$y^2=x^3-x$, CM (atom at 0 removed)"); ax[1].set_xlabel(r"$a_p/\sqrt{p}$")
plt.tight_layout(); plt.savefig("../work/fig_satotate.pdf"); plt.close()

# Kesten McKay measure, moments c_{2m} = q^{-m}
def km(q, x):
    return (q + 1) / ((math.sqrt(q) + 1 / math.sqrt(q)) ** 2 - x ** 2) * np.sqrt(np.clip(4 - x ** 2, 0, None)) / (2 * np.pi)
from scipy.integrate import quad
chk = []
for q in [2, 3, 5, 11]:
    row = []
    for n in [2, 4, 6]:
        v = quad(lambda x: Un(n, np.array(x)) * km(q, x), -2, 2, limit=200)[0]
        row.append((n, v, q ** (-n / 2)))
    chk.append((q, row))
out["km_check"] = chk

# return probabilities of SRW on (q+1) regular tree vs Kesten McKay moments
def ret_prob(q, T):
    d = np.zeros(T + 2); d[0] = 1; r = []
    for t in range(1, T + 1):
        nd = np.zeros_like(d)
        nd[1] += d[0]
        nd[2:] += d[1:-1] * q / (q + 1)
        nd[:-1] += d[1:] / (q + 1)
        d = nd; r.append(d[0])
    return r
tree = []
for q in [2, 5]:
    r = ret_prob(q, 12)
    for m in [1, 3, 6]:
        mom = quad(lambda x: x ** (2*m) * km(q, x), -2, 2, limit=200)[0]
        walks = r[2*m - 1] * (q + 1) ** (2*m) / q ** m
        tree.append((q, 2*m, walks, mom))
out["tree_check"] = tree

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for q, c in zip([2, 3, 5, 11], ["#a6611a", "#dfc27d", "#80cdc1", "#018571"]):
    ax[0].plot(xs, km(q, xs), color=c, label=f"$p={q}$")
ax[0].plot(xs, st, "k--", lw=1, label="Sato Tate"); ax[0].legend(fontsize=8); ax[0].set_title("Plancherel measures $\\mu_p$")
# random 4 regular graph spectrum (q=3)
G = nx.random_regular_graph(4, 3000, seed=1)
ev = np.linalg.eigvalsh(nx.to_numpy_array(G)) / math.sqrt(3)
ax[1].hist(ev[np.abs(ev) <= 2.05], bins=60, density=True, color="#bcd4bf", edgecolor="white")
ax[1].plot(xs, km(3, xs), color="#1d5c2a"); ax[1].set_title("random 4 regular graph, $n=3000$")
plt.tight_layout(); plt.savefig("../work/fig_tree.pdf"); plt.close()
lam = np.sort(np.abs(ev))[-2] * math.sqrt(3)
out["graph_lambda2"] = float(lam); out["ramanujan_bound"] = 2 * math.sqrt(3)

# Bayesian identification of the Sato Tate group
def loglik_st(x): return np.log(np.sqrt(np.clip(4 - x ** 2, 1e-12, None)) / (2 * np.pi))
def loglik_cm(x, zero):
    return np.where(zero, np.log(0.5), np.log(0.5 / (np.pi * np.sqrt(np.clip(4 - x ** 2, 1e-12, None)))))
def bf_path(E):
    x = E[:, 1] / np.sqrt(E[:, 0]); z = E[:, 1] == 0
    eps = 1e-3   # small atom at zero under SU(2) model so a single zero is not decisive
    l_st = np.where(z, np.log(eps), np.log(1 - eps) + loglik_st(x))
    l_cm = loglik_cm(x, z)
    return np.cumsum(np.clip(l_cm - l_st, -20, 20))
b1 = bf_path(E1); b2 = bf_path(E2)
out["bf_E1"] = [(k, float(b1[k - 1])) for k in [10, 50, 100, 1000, len(b1)]]
out["bf_E2"] = [(k, float(b2[k - 1])) for k in [10, 50, 100, 1000, len(b2)]]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.plot(np.arange(1, 201), b1[:200], color="#1f3b63", label="no CM curve")
ax.plot(np.arange(1, 201), b2[:200], color="#63381f", label="CM curve")
ax.axhline(0, color="gray", lw=.6); ax.set_xlabel("number of good primes"); ax.set_ylabel(r"$\log B_{\rm CM:SU(2)}$")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("../work/fig_bayes.pdf"); plt.close()

# Lang Trotter: count of a_p = 0 for the curve without CM
lt=[]
for X in [5000,10000,20000,40000]:
    m=E1[:,0]<=X
    c=int(np.sum(E1[m,1]==0)); lt.append((X,c,c/(math.sqrt(X)/math.log(X))))
out["lang_trotter"]=lt
out["zeros_E1"]=[int(p) for p in E1[E1[:,1]==0,0]]
# discrepancy scaling: sqrt(n) D_n
out["ks_scaled"]=[(k,d,d*math.sqrt(k)) for k,d in ks]
# limits of (1/n) log B
out["bf_rate_E1"]=float(b1[-1]/len(b1)); out["bf_rate_E2"]=float(b2[-1]/len(b2))
eps=1e-3
out["theory_E1"]=-1-math.log(1-eps); out["theory_E2"]=0.5*math.log(0.5/eps)-0.5*math.log(1-eps)
json.dump(out, open("../work/results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
