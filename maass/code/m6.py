"""Moments of the vertical angle law from the Eichler Selberg formula: (1/d_k) tr T_{p^m} / p^{m(k-1)/2} for p = 2, m = 2, 4,
as functions of k, converging to p^{-m/2}; and the geodesic Euler product R(s) on real s."""
import json, math, numpy as np
from fractions import Fraction
import cypari2
pari = cypari2.Pari()
def H(N):
    if N == 0: return Fraction(-1, 12)
    return Fraction(str(pari("qfbhclassno(%d)" % N)))
def P(k, t, n):
    a, b = 1, t
    if k == 2: return 1
    if k == 3: return t
    for _ in range(k - 3): a, b = b, t*b - n*a
    return b
def trace(k, n):
    s = Fraction(0); tmax = math.isqrt(4*n)
    for t in range(-tmax, tmax + 1): s += P(k, t, n)*H(4*n - t*t)
    s = -s/2; d = Fraction(0)
    for dd in range(1, n + 1):
        if n % dd == 0: d += min(dd, n//dd)**(k - 1)
    return s - d/2
def dim(k): return k//12 - 1 if k % 12 == 2 else k//12
rows = []
for k in range(12, 401, 2):
    dk = dim(k)
    if dk == 0: continue
    m2 = float(trace(k, 4))/(dk*2.0**(2*(k-1)/2)); m4 = float(trace(k, 16))/(dk*2.0**(4*(k-1)/2))
    rows.append((k, m2, m4))
json.dump(rows, open("m6.json", "w"))
print(rows[:3], rows[-2:])
