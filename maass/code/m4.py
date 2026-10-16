"""Serre's vertical Sato Tate law at p = 2: normalised Hecke eigenvalues a_2(f)/2^{(k-1)/2} of all level one
eigenforms of weight k <= K, against the p-adic Plancherel measure
mu_p(x) = (p+1)/pi * sqrt(1 - x^2/4) / ((p^{1/2} + p^{-1/2})^2 - x^2) dx on [-2, 2]."""
import cypari2, numpy as np, json, sys
pari = cypari2.Pari(); pari.allocatemem(4*10**9)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 400
p = 2
xs = []
for k in range(12, K + 1, 2):
    mf = pari.mfinit([1, k], 1)
    M = pari.mfheckemat(mf, p)
    n = int(pari.matsize(M)[0])
    if n == 0: continue
    A = np.array([[float(M[i, j]) for j in range(n)] for i in range(n)])
    ev = np.linalg.eigvals(A).real
    xs.extend((ev/2.0**((k - 1)/2)).tolist())
xs = np.array(xs); print("eigenvalues:", len(xs), "max |x|", np.abs(xs).max())
def mu(x):
    return (p + 1)/np.pi*np.sqrt(np.clip(1 - x*x/4, 0, None))/((np.sqrt(p) + 1/np.sqrt(p))**2 - x*x)
def st(x): return np.sqrt(np.clip(1 - x*x/4, 0, None))/np.pi
# moments
for m in [2, 4, 6]:
    from scipy.integrate import quad
    th = quad(lambda x: x**m*mu(x), -2, 2)[0]; stm = quad(lambda x: x**m*st(x), -2, 2)[0]
    print("moment", m, "empirical", np.mean(xs**m), "Plancherel", th, "Sato Tate", stm)
# KS distance to mu_p
from scipy.integrate import quad
grid = np.linspace(-2, 2, 401); cdf = np.array([quad(mu, -2, g)[0] for g in grid])
emp = np.searchsorted(np.sort(xs), grid)/len(xs)
print("KS distance to Plancherel:", np.max(np.abs(emp - cdf)))
cdfst = np.array([quad(st, -2, g)[0] for g in grid]); print("KS distance to Sato Tate:", np.max(np.abs(emp - cdfst)))
np.save("vertical_p2.npy", xs)
json.dump(dict(K=K, n=len(xs), ks_pl=float(np.max(np.abs(emp - cdf))), ks_st=float(np.max(np.abs(emp - cdfst))),
               m2=float(np.mean(xs**2)), m4=float(np.mean(xs**4))), open("m4.json", "w"), indent=1)
