"""Sato Tate over a CM field (Allen, Calegari, Caraiani, Gee, Helm, Le Hung, Newton, Scholze, Taylor and Thorne):
Frobenius angles of E: y^2 = x^3 + i x + 1 over Q(i) at the primes of norm <= X, against the Sato Tate law."""
import cypari2, numpy as np, json, sys, math
pari = cypari2.Pari(); pari.allocatemem(2*10**9)
X = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
K = pari.nfinit("y^2+1"); E = pari.ellinit("[0,0,0,y,1]", K)
print("conductor norm:", pari.idealnorm(K, pari.ellglobalred(E)[0]), " j =", E[12])
s = np.ones(X+1, bool); s[:2] = False
for i in range(2, int(X**0.5)+1):
    if s[i]: s[i*i::i] = False
P = np.nonzero(s)[0]
norms = []; ap = []; deg = []
bad = int(pari.idealnorm(K, pari.ellglobalred(E)[0]))
for p in P:
    p = int(p)
    for pr in pari.idealprimedec(K, p):
        f = int(pr[3]); N = p**f
        if N > X or bad % p == 0: continue
        norms.append(N); ap.append(int(pari.ellap(E, pr))); deg.append(f)
norms = np.array(norms, float); ap = np.array(ap, float); deg = np.array(deg)
x = ap/np.sqrt(norms); print("primes used", len(x), "degree one", np.sum(deg == 1), "degree two", np.sum(deg == 2), "max |x|", np.abs(x).max())
from scipy.integrate import quad
st = lambda u: np.sqrt(np.clip(1 - u*u/4, 0, None))/np.pi
mom = {m: (float(np.mean(x**m)), quad(lambda u: u**m*st(u), -2, 2)[0]) for m in [2, 4, 6]}
print("moments (empirical, Sato Tate):", mom)
grid = np.linspace(-2, 2, 401); cdf = np.array([quad(st, -2, g)[0] for g in grid]); emp = np.searchsorted(np.sort(x), grid)/len(x)
ks = float(np.max(np.abs(emp - cdf))); print("KS to Sato Tate:", ks, " 1/sqrt(n) =", 1/math.sqrt(len(x)))
# split and inert separately
for f in [1, 2]:
    xx = x[deg == f]; emp = np.searchsorted(np.sort(xx), grid)/len(xx); print("degree", f, "n", len(xx), "KS", float(np.max(np.abs(emp - cdf))), "E x^2", float(np.mean(xx**2)))
np.savez("ec_qi.npz", norms=norms, ap=ap, deg=deg)
json.dump(dict(X=X, n=int(len(x)), n1=int(np.sum(deg == 1)), n2=int(np.sum(deg == 2)), mom=mom, ks=ks, cond=bad), open("c2.json", "w"), indent=1)
