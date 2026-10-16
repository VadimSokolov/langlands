"""Dirichlet classes mod 4 as a two colour random integer.
(a) Mertens constants per class from L(1,chi_4) = pi/4 against sums over primes below 10^7.
(b) atoms at zero: P(N_sigma has no prime factor = 3 mod 4) = prod_{p=3(4)} (1 - p^-sigma) ~ e^{-h_3(1)} (sigma-1)^{1/2}.
(c) Wirsing/Landau: proportion of n <= x with no prime factor = 3 mod 4 against e^{-h_3(1)} (log x)^{-1/2}/Gamma(1/2),
    and the Landau Ramanujan constant for sums of two squares for comparison."""
import numpy as np, json, math
from mpmath import mp, mpf, log, zeta, euler, gamma, pi, sqrt
mp.dps = 30
d = np.load("gl1_data.npz"); P = d["P"]; X = int(P[-1])
P1 = P[P % 4 == 1]; P3 = P[P % 4 == 3]
# (a) empirical Mertens per class
ll = math.log(math.log(X))
M1 = np.sum(1.0/P1) - 0.5*ll; M3 = np.sum(1.0/P3) - 0.5*ll
# theory: sum_{p odd} 1/p = loglog x + M - 1/2 + o(1), M = Mertens constant
M = mpf("0.2614972128476427837554268386086958590516")
# S_chi(1) = sum_p chi(p)/p = log L(1,chi) - sum_{k>=2} sum_p chi(p)^k/(k p^k)
def Lchi4(s): return mpf(sum(((-1)**((int(p)-1)//2))*mpf(int(p))**(-s) for p in P[:0]))  # unused
# compute sum_{k>=2} sum_p chi(p^k)/(k p^k) with primes to 10^7 (tail negligible)
tail = mpf(0)
for k in range(2, 60):
    sk = mpf(0)
    pk = P.astype(float)**(-k)
    chi = np.where(P % 4 == 1, 1.0, np.where(P % 4 == 3, -1.0, 0.0))
    sk = np.sum((chi**k)*pk)/k
    tail += mpf(float(sk))
    if abs(sk) < 1e-25: break
S = log(pi/4) - tail
M1th = (M - mpf(1)/2 + S)/2; M3th = (M - mpf(1)/2 - S)/2
print("Mertens per class: empirical", M1, M3, " theory", float(M1th), float(M3th))
# (b) atoms at zero
# h_3(1): prod_{p=3(4), p<=x}(1-1/p)^{-1} ~ e^{h_3(1)} (log x)^{1/2}: by Mertens for the class, exp(sum 1/p + sum_k>=2 ...) 
# exact: prod_{p=3(4)} (1 - p^{-s})^{-1} = e^{h_3(1)} (s-1)^{-1/2}(1+o(1)); compute at sigma via the identity
# prod_{p=3(4)}(1-p^-s)^{-2} = zeta^{(2)}(s)/L(s,chi) * prod_{p=1(4)}(1-p^{-2s})   [since (1-p^-s)(1+p^-s) = 1-p^{-2s} for p=1 mod 4 ... careful]
# zeta^{(2)}(s)/L(s,chi) = prod_{p odd} (1 - chi(p)p^-s)/(1 - p^-s) = prod_{p=3} (1+p^-s)/(1-p^-s) = prod_{p=3} (1-p^{-2s})/(1-p^-s)^2
# so prod_{p=3}(1-p^-s)^{-2} = zeta^{(2)}(s)/(L(s,chi) prod_{p=3}(1-p^{-2s}))
def Lchi(s):
    # L(s, chi_4) via mpmath: Dirichlet beta
    from mpmath import dirichlet
    return dirichlet(s, [0, 1, 0, -1])
def prod3sq(s):  # prod_{p=3(4)} (1 - p^{-2s}) with primes to 10^7 (tail < 1e-13)
    return float(np.exp(np.sum(np.log1p(-P3.astype(float)**(-2*float(s))))))
def atom3(sigma):  # prod_{p=3}(1-p^-sigma) exact via the identity
    v = zeta(sigma)*(1-mpf(2)**(-sigma))/(Lchi(sigma)*prod3sq(sigma))
    return 1/sqrt(v)
def atom1(sigma):  # prod_{p=1(4)}(1-p^-sigma): zeta^{(2)} L = prod_{p=1}(1-p^-s)^{-2} prod_{p=3}(1-p^{-2s})^{-1}
    v = zeta(sigma)*(1-mpf(2)**(-sigma))*Lchi(sigma)*prod3sq(sigma)
    return 1/sqrt(v)
res = {}
for sigma in [1.1, 1.01, 1.001, 1.0001]:
    a3 = atom3(mpf(sigma)); a1 = atom1(mpf(sigma))
    res[str(sigma)] = (float(a3), float(a1), float(a3/sqrt(sigma-1)), float(a1/sqrt(sigma-1)))
    print("sigma", sigma, "P(no 3 mod 4)", float(a3), "P(no 1 mod 4)", float(a1), " /sqrt(sigma-1):", float(a3/sqrt(sigma-1)), float(a1/sqrt(sigma-1)))
# e^{-h_3(1)} = lim a3/sqrt(sigma-1): h_3(1) via the identity at sigma -> 1: a3^2 (sigma-1) -> L(1,chi) prod3sq(1) / 1 (zeta ~ 1/(s-1), 1-2^-1 = 1/2) => e^{-2h_3} = (pi/4) prod3sq(1) * (1/2)... 
eh3 = sqrt(2*Lchi(1)*prod3sq(1))      # e^{-h_3(1)}: prod_{p=3}(1-p^-s)^2 ~ 2(s-1) L(1,chi) prod3sq
eh1 = 1/sqrt(Lchi(1)*prod3sq(1)/2)   # e^{-h_1(1)}  (check: a1^2 (sigma-1) -> (1/2) L(1) prod3sq)
print("e^{-h_3(1)} =", eh3, " e^{-h_1(1)} =", eh1, " e^{-h_1} e^{-h_3} =", eh1*eh3, "(should be 2)")
# (c) Wirsing: proportion of n <= x with no prime factor 3 mod 4 (2 allowed) ~ e^{-h_3(1)} ... wait: integers composed of 2 and primes 1 mod 4;
# generating Dirichlet series F(s) = (1-2^-s)^{-1} prod_{p=1}(1-p^-s)^{-1} ~ e^{h_1(1)+log 2} (s-1)^{-1/2}; count ~ 2 e^{h_1(1)} x (log x)^{-1/2}/Gamma(1/2)
K_ours = 2*(1/eh1)/gamma(mpf(1)/2)
# Landau Ramanujan constant: K = 2^{-1/2} prod_{p=3}(1-p^-2)^{-1/2}
K_LR = 1/sqrt(2*prod3sq(1))
print("constant for 'no prime factor 3 mod 4':", K_ours, "  Landau Ramanujan:", K_LR, " ratio", K_ours/K_LR)
# sieve count to 10^8
N = 10**8
sv = np.ones(N+1, bool); sv[:2] = False
for i in range(2, int(N**0.5)+1):
    if sv[i]: sv[i*i::i] = False
PN = np.nonzero(sv)[0]; P3N = PN[PN % 4 == 3]
good = np.ones(N+1, bool); good[0] = False
for p in P3N:
    good[p::p] = False
cnt = np.cumsum(good)
rows = []
for x in [10**5, 10**6, 10**7, 10**8]:
    c = int(cnt[x]); pred = float(K_ours)*x/math.sqrt(math.log(x))
    rows.append((x, c, pred, c/pred)); print("x", x, "count", c, "pred", round(pred), "ratio", c/pred)
# sums of two squares count for comparison
json.dump(dict(M1=M1, M3=M3, M1th=float(M1th), M3th=float(M3th), atoms=res, eh3=float(eh3), eh1=float(eh1), K=float(K_ours), KLR=float(K_LR), rows=rows), open("g4.json","w"), indent=1)
