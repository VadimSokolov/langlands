"""Rankin Selberg subordinators of the CM curve 32a2 over Q and over Q(i), and of 11a1 over Q(i).
Laplace transform E exp(-lambda (sigma-1) X_sigma) = D^S(sigma + lambda(sigma-1))/D^S(sigma),
D^S(s) = zeta^{(2)}(s) L(s+1, Sym^2 E) for 32a2 over Q (partial product over p != 2; PARI's Sym^2 has trivial factor at 2),
and D_K = D_Q^2 for a CM curve over its CM field.
For 11a1 over K = Q(i): D_K(s) = D_Q(s) L(s, rho x rho^v x chi_4), with partial products over p not in {2, 11}."""
import cypari2, json, numpy as np
pari = cypari2.Pari(); pari.allocatemem(2*10**9); pari.default("realprecision", 30)
E = pari.ellinit([0,0,0,-1,0]); S2 = pari.lfunsympow(E, 2)
E11 = pari.ellinit([0,-1,1,-10,-20]); S211 = pari.lfunsympow(E11, 2)
chi4 = pari.lfuncreate(pari("Mod(3,4)"))
def zeta2(s): return float(pari.zeta(s))*(1-2.0**(-s))
def D32(s):  return zeta2(s)*float(pari.lfun(S2, s+1))
# 11a1: local factors at 2 and 11 of the partial products. Sym^2 of 11a1 from PARI: check Euler factor at 11 (multiplicative, a_11 = 1)
def eulerfac(L, p, s):
    # returns the local factor L_p(s) of PARI L-function via lfuneuler if available
    try:
        f = pari.lfuneuler(L, p)  # polynomial in x = p^-s ; local factor 1/f
        return 1.0/float(pari.subst(f, "x", pari(p)**(-s)))
    except Exception as e:
        return None
print("Sym2(11a1) factor at 11, s=3:", eulerfac(S211, 11, 3), "zeta factor", 1/(1-11.**-3))
print("Sym2(32a2) factor at 2, s=3:", eulerfac(S2, 2, 3))
out = {}
for sigma in [1.1, 1.01, 1.001]:
    row = []
    for lam in [0.5, 1, 2, 4, 8]:
        sp = sigma + lam*(sigma-1)
        LQ = D32(sp)/D32(sigma)
        row.append((lam, LQ, 1/(1+lam), LQ**2, 1/(1+lam)**2))
    out[str(sigma)] = row
    print("sigma", sigma)
    for r in row: print("  lam %4.1f  Q: %.5f vs %.5f   K: %.5f vs %.5f" % r)
json.dump(out, open("g3.json","w"), indent=1)
