"""Prime geodesic theorem on the modular surface.
Primitive closed geodesics <-> (d, narrow class of primitive forms of discriminant d), d > 0 not a square,
norm N(P) = eps_d^2 with eps_d the smallest unit > 1 of norm +1 in the order of discriminant d,
multiplicity h^+(d) = h(d) if the order has a unit of norm -1, else 2h(d).
Counts pi_Gamma(x) = #{P : N(P) <= x} against li(x), and psi_Gamma(x) = sum_{N(P)^k <= x} log N(P) against x.
Writes geodesics.npz with (N(P), multiplicity) for N(P) <= X."""
import cypari2, numpy as np, math, sys, json
from mpmath import li
pari = cypari2.Pari(); pari.allocatemem(2*10**9)
X = float(sys.argv[1]) if len(sys.argv) > 1 else 1e5
# eps_d >= (sqrt(d) - ?)... for a unit (t + u sqrt d)/2 with t^2 - d u^2 = +-4, eps >= (sqrt(d-4)+sqrt d)/2 >= sqrt(d)/2 roughly; N = eps^2 >= d/4 - 1
Dmax = int(4*X + 8)
norms = []; mults = []; discs = []
sq = set(i*i for i in range(int(math.isqrt(Dmax)) + 2))
for d in range(5, Dmax + 1):
    if d % 4 not in (0, 1) or d in sq: continue
    u = pari.quadunit(d)
    nu = int(pari.norm(u))
    # real embedding: u = a + b*w, w = sqrt(d)/2-ish; use pari to get float
    a = int(pari.component(u, 2)); b = int(pari.component(u, 3))
    if max(abs(a), abs(b)) > 10**12: continue   # unit far above the cutoff
    w = ((d % 2) + math.sqrt(d))/2
    ur = a + b*w
    if nu == -1: ur = ur*ur
    N = ur*ur
    if N <= X:
        h = int(pari.qfbclassno(d)); hp = h if nu == -1 else 2*h
        norms.append(N); mults.append(hp); discs.append(d)
norms = np.array(norms); mults = np.array(mults); discs = np.array(discs)
order = np.argsort(norms); norms = norms[order]; mults = mults[order]; discs = discs[order]
print("primitive geodesics with N <= %g:" % X, int(mults.sum()), " distinct norms", len(norms))
print("smallest:", [(float(n), int(m), int(dd)) for n, m, dd in zip(norms[:6], mults[:6], discs[:6])])
rows = []
for x in [10**k for k in range(2, int(math.log10(X)) + 1)]:
    pi = int(mults[norms <= x].sum()); L = float(li(x))
    # psi: sum over k>=1 of sum_{N^k <= x} h^+(d) log N
    psi = 0.0
    for n, m in zip(norms, mults):
        k = 1
        while n**k <= x:
            psi += m*math.log(n); k += 1
    rows.append((x, pi, L, pi/L, psi, psi/x)); print("x %g  pi_Gamma %d  li %.1f  ratio %.4f   psi %.1f psi/x %.4f" % (x, pi, L, pi/L, psi, psi/x))
np.savez("geodesics.npz", norms=norms, mults=mults, discs=discs)
json.dump(dict(X=X, total=int(mults.sum()), rows=rows), open("m1.json", "w"), indent=1)
