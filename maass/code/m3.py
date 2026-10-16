"""Eichler Selberg trace formula at level one against PARI's Hecke matrices.
tr T_n | S_k = -1/2 sum_{t^2 <= 4n} P_k(t,n) H(4n - t^2) - 1/2 sum_{d d' = n} min(d,d')^{k-1},
P_k(t,n) = (rho^{k-1} - rhobar^{k-1})/(rho - rhobar), rho + rhobar = t, rho rhobar = n; H the Hurwitz class number, H(0) = -1/12."""
import cypari2, json, math
from fractions import Fraction
pari = cypari2.Pari(); pari.allocatemem(2*10**9)
def H(N):
    if N == 0: return Fraction(-1, 12)
    return Fraction(str(pari("qfbhclassno(%d)" % N)))
def P(k, t, n):
    # Gegenbauer: coefficients of x^{k-2} in 1/(1 - t x + n x^2)
    a, b = 1, t   # P_2 = 1 (k=2), P_3 = t ; recursion P_{k} = t P_{k-1} - n P_{k-2}
    if k == 2: return 1
    if k == 3: return t
    for _ in range(k - 3):
        a, b = b, t*b - n*a
    return b
def trace(k, n):
    s = Fraction(0)
    tmax = math.isqrt(4*n)
    for t in range(-tmax, tmax + 1):
        s += P(k, t, n)*H(4*n - t*t)
    s = -s/2
    d = Fraction(0)
    for dd in range(1, n + 1):
        if n % dd == 0: d += min(dd, n//dd)**(k - 1)
    return s - d/2
bad = 0; checked = 0
for k in range(12, 41, 2):
    mf = pari.mfinit([1, k], 1)  # cuspidal space
    for n in range(1, 31):
        M = pari.mfheckemat(mf, n)
        tr = int(pari.trace(M)) if int(pari.matsize(M)[0]) > 0 else 0
        es = trace(k, n)
        checked += 1
        if es != tr:
            bad += 1; print("MISMATCH k", k, "n", n, "pari", tr, "formula", es)
print("checked", checked, "mismatches", bad)
print("tr T_2 on S_12:", trace(12, 2), " tau(2) = -24;  tr T_2 on S_24:", trace(24, 2))
json.dump(dict(checked=checked, bad=bad), open("m3.json", "w"))
