"""Second check for the chapter on zeta values: the direct sum for L(3,chi_12) (the nsum of the coauthor's script fails on
this nonsmooth summand), w_0 of the 2 adic matrix, and the arithmetic of the recurrence as printed, in exact fractions."""
import json
from fractions import Fraction
from math import lcm
import numpy as np
from scipy.linalg import eigh_tridiagonal

out = {}
K = 200000
k = np.arange(1, K + 1, dtype=float)
ki = np.arange(1, K + 1) % 12
chi = np.where((ki == 1) | (ki == 11), 1.0, np.where((ki == 5) | (ki == 7), -1.0, 0.0))
out["L3chi12_partial_2e5"] = float(np.sum(chi / k**3))

res = {}
for N in (100, 200, 400):
    n = np.arange(N, dtype=float)
    b = 32 * (2 * n + 1) * (8 * n**4 + 16 * n**3 + 20 * n**2 + 12 * n + 3)
    a = 256 * n[1:] ** 5
    lam, V = eigh_tridiagonal(b, a)
    res[str(N)] = dict(lam0=float(lam[0]), w0=float(V[0, 0] ** 2), sum=float(np.sum(V[0, :] ** 2 / lam)))
out["two_adic_matrix"] = res


def seq(quartic, nmax=40):
    r = [Fraction(1), Fraction(32 * quartic(0))]
    for n in range(1, nmax):
        r.append((32 * (2 * n + 1) * quartic(n) * r[n] - 2**16 * n**5 * r[n - 1]) / (n + 1) ** 5)
    return r


printed = seq(lambda n: 8 * n**4 + 16 * n**3 + 20 * n**2 + 12 * n + 3)
out["rho_first"] = [str(x) for x in printed[:8]]
out["rho_all_integer_to"] = max((i for i in range(len(printed)) if all(x.denominator == 1 for x in printed[: i + 1])), default=-1)
L = 1
ok = []
for i in range(1, len(printed)):
    L = lcm(L, i)
    ok.append(bool((printed[i] * L**5).denominator == 1))
out["rho_times_lcm5_integer"] = all(ok)
variant = seq(lambda n: 8 * n**4 + 16 * n**3 + 18 * n**2 + 10 * n + 3)
out["variant_18_10_denominator_at_6"] = str(variant[6].denominator)
json.dump(out, open("check_zeta2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
