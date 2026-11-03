"""Independent checks of statements of the chapter on zeta values that the coauthor's scripts do not test.
A: the zeta(2) fraction as a real symmetric Jacobi matrix with alternating diagonal.
B: eigenvalues for large k against second order perturbation theory and Bohr Sommerfeld.
C: log w_k for large k against the product formula and the WKB limit.
D: the Apery GGC near the origin, the total variation bound, the mean and variance.
E: closed forms of L(3,chi_{-3}).
Writes check_zeta.json."""
import json
import numpy as np
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
import mpmath as mp

mp.mp.dps = 40
out = {}


def P(n):
    return 34 * n**3 + 51 * n**2 + 27 * n + 5


# A
res = {}
for N in (50, 500, 3000):
    n = np.arange(N, dtype=float)
    b = (-1.0) ** n * (11 * n**2 + 11 * n + 3)
    a = n[1:] ** 2
    lam, V = eigh_tridiagonal(b, a)
    w = V[0, :] ** 2
    res[str(N)] = dict(five_sum=float(5 * np.sum(w / lam)), mass_pos=float(w[lam > 0].sum()), mass_neg=float(w[lam < 0].sum()),
                       smallest_pos=float(lam[lam > 0].min()), largest_neg=float(lam[lam < 0].max()))
out["A_zeta2_alternating"] = dict(by_N=res, zeta2=float(mp.zeta(2)), dominance_min=float(min(11 * k * k + 11 * k + 3 - k * k - (k + 1) ** 2 for k in range(1000))))

# B
N = 6000
n = np.arange(N, dtype=float)
b = P(n)
a = n[1:] ** 3
kmax = 3000
lam = eigvalsh_tridiagonal(b, a, select="i", select_range=(0, kmax))


def Delta(k):
    return P(k + 1.0) - P(float(k))


M = mp.quad(lambda p: (34 - 2 * mp.cos(p)) ** (-mp.mpf(1) / 3), [0, 2 * mp.pi]) / (2 * mp.pi)
C = 1 / M**3
out["B_bohr_sommerfeld"] = dict(C=float(C), c_eps=float(34 - C), two_over_51=2 / 51, difference=float(34 - C - mp.mpf(2) / 51))
rows = []
for k in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000):
    eps = P(float(k)) - lam[k]
    second = (k + 1.0) ** 6 / Delta(k) - k**6 / Delta(k - 1)
    rows.append(dict(k=k, eps=float(eps), second=float(second), eps_over_k3=float(eps / k**3), rel_gap=float((eps - second) / second), gap_over_k3=float((eps - second) / k**3), coupling_over_gap=float(k**3 / Delta(k))))
out["B_eigenvalues"] = rows


# C
def log10w(k):
    lk, Vk = eigh_tridiagonal(b, a, select="i", select_range=(k, k))
    lk = lk[0]
    v = np.abs(Vk[:, 0])
    m = int(np.argmax(v >= 0.1 * v.max()))
    if m == 0:
        return float(2 * np.log10(v[0]))
    s = 0.0
    r = a[0] / (lk - b[0])
    s += np.log10(abs(r))
    for j in range(1, m):
        r = a[j] / (lk - b[j] - a[j - 1] * r)
        s += np.log10(abs(r))
    return float(2 * (s + np.log10(v[m])))


def prod10(k):
    return float(sum(6 * np.log10(float(j)) - 2 * np.log10(P(float(k)) - P(float(j - 1))) for j in range(1, k + 1)))


rows = []
for k in (1, 2, 3, 5, 9, 20, 30, 50, 100, 200, 400, 800, 1600):
    w0 = log10w(k)
    w1 = log10w(k + 10)
    rows.append(dict(k=k, log10w=w0, log10_prod=prod10(k), log10w_over_k=w0 / k, local_slope=(w1 - w0) / 10, prod_slope=(prod10(k + 10) - prod10(k)) / 10))
out["C_weights"] = rows
tstar = (C / 36) ** (mp.mpf(1) / 3)
Iw = mp.quad(lambda t: mp.re(mp.acosh(max((C - 34 * t**3) / (2 * t**3), mp.mpf(1)))), [0, tstar / 100, tstar / 2, tstar])
I0 = mp.quad(lambda t: mp.log(1 - t**3), [0, 1])
out["C_limits"] = dict(wkb_ln=float(-2 * Iw), wkb_log10=float(-2 * Iw / mp.log(10)), product_ln=float(-(2 * mp.log(34) + 6 + 2 * I0)), product_log10=float(-(2 * mp.log(34) + 6 + 2 * I0) / mp.log(10)), tstar=float(tstar))

# D
N = 40
J = mp.matrix(N, N)
for i in range(N):
    J[i, i] = P(i)
    if i > 0:
        J[i, i - 1] = J[i - 1, i] = mp.mpf(i) ** 3
E, Q = mp.eigsy(J)
L = [E[k] for k in range(N)]
W = [Q[0, k] ** 2 for k in range(N)]
mean = sum(W[k] / L[k] for k in range(N))
rem = sum(W[k] / L[k] for k in range(1, N))
logm = sum(W[k] * mp.log(L[k]) for k in range(N))
tv = mp.gammainc(W[0], 0, L[0] * rem, regularized=True)
out["D_ggc"] = dict(total_mass=float(sum(W)), mean=str(mp.nstr(mean, 15)), zeta3_over_6=str(mp.nstr(mp.zeta(3) / 6, 15)), w0_over_l0=str(mp.nstr(W[0] / L[0], 15)), gap=float(rem), var=str(mp.nstr(sum(W[k] / L[k] ** 2 for k in range(N)), 12)), var0=str(mp.nstr(W[0] / L[0] ** 2, 12)), f0_plus=float(mp.exp(logm)), int_log_x=float(logm), log1p_moment=float(sum(W[k] * mp.log(1 + L[k]) for k in range(N))), tv_bound=float(tv), w0=float(W[0]), l0=float(L[0]))

# E
chi3 = lambda s: mp.nsum(lambda k: (mp.zeta(s, (3 * k + 1)) * 0 + 1) * ((3 * k + 1) ** (-s) - (3 * k + 2) ** (-s)), [0, mp.inf])
out["E_L3chi3"] = dict(hurwitz=float((mp.zeta(3, mp.mpf(1) / 3) - mp.zeta(3, mp.mpf(2) / 3)) / 27), closed=float(4 * mp.pi**3 / (81 * mp.sqrt(3))), diff=float((mp.zeta(3, mp.mpf(1) / 3) - mp.zeta(3, mp.mpf(2) / 3)) / 27 - 4 * mp.pi**3 / (81 * mp.sqrt(3))))
json.dump(out, open("check_zeta.json", "w"), indent=1)
print(json.dumps(out, indent=1))
