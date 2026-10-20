"""Thorin test for a genus two curve (Boxer, Calegari, Gee and Pilloni give the meromorphic continuation and functional equation):
C: y^2 + (x^3+1) y = x^2 + x, conductor 249. Xi(w) = Lambda(1+w), zeros 1 +- i gamma; Xi(sqrt l)/Xi(0) against prod (1 + l/gamma^2) with tail;
Sato Tate moments of a_p/sqrt p against USp(4)."""
import cypari2, numpy as np, json, math, sys
from scipy.integrate import quad
pari = cypari2.Pari(); pari.allocatemem(4*10**9); pari.default("realprecision", 19)
L = pari.lfungenus2(pari("[x^2+x, x^3+1]"))
print("conductor", pari.lfunconductor(L) if False else L[5] if False else "")
T = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
z = np.array([float(g) for g in pari.lfunzeros(L, T)]); print("zeros below", T, len(z), z[:5])
cond = 249.0
# counting function for degree 4, conductor N, gamma factor Gamma_C(s)^2: N(T) ~ (T/pi) log(T^2 N^{1/2} / (2 pi e)^2 ) ... use general formula (dT/2pi) log(T/(2 pi e)) + (T/2pi) log N
Ncnt = lambda t: 4*t/(2*math.pi)*math.log(t/(2*math.pi*math.e)) + t/(2*math.pi)*math.log(cond)
print("N(T) formula", Ncnt(T), "actual", len(z))
Lam = lambda s: float(pari.lfunlambda(L, s))
X0 = Lam(1.0); print("Lambda(1)", X0, " Lambda(1.5)", Lam(1.5))
dens = lambda t: (4/(2*math.pi))*math.log(t/(2*math.pi)) + math.log(cond)/(2*math.pi)
rows = []
for lam in [0.25, 1, 4, 9, 16, 36]:
    lhs = Lam(1 + math.sqrt(lam))/X0
    lp = np.sum(np.log1p(lam/z**2)); tl = quad(lambda t: math.log1p(lam/t**2)*dens(t), T, np.inf, limit=300)[0]
    rhs = math.exp(lp + tl); rows.append((lam, lhs, rhs, lhs/rhs - 1)); print("lambda", lam, "Xi ratio", lhs, "product", rhs, "rel", lhs/rhs - 1)
# Sato Tate moments for USp(4): E[a^2] = 1, E[a^4] = 3, E[a^6] = 14 (normalised trace a_p/sqrt p)
xs = []
# a_p = p + 1 - #C(F_p) with C: Y^2 = (x^3+1)^2 + 4(x^2+x), two points at infinity (leading coefficient 1):
# #C(F_p) = p + 2 + sum_x legendre(f(x)); so a_p = -1 - sum_x legendre(f(x)), checked against hyperellcharpoly at p = 101 (a_101 = -12)
for p in pari.primes(20000):
    p = int(p)
    if 249 % p == 0 or p == 2: continue
    x = np.arange(p, dtype=np.int64)
    f = ((x**3 + 1) % p)**2 % p; f = (f + 4*((x*x + x) % p)) % p
    sq = np.zeros(p, np.int8); sq[(x*x) % p] = 1
    chi = np.where(f == 0, 0, 2*sq[f] - 1)
    ap = -1 - int(chi.sum()); xs.append(ap/math.sqrt(p))
    if p == 101: assert ap == -12, ap
xs = np.array(xs); print("primes", len(xs), "moments", np.mean(xs**2), np.mean(xs**4), np.mean(xs**6), "USp(4): 1, 3, 14")
json.dump(dict(T=T, nzeros=int(len(z)), first=z[:6].tolist(), rows=rows, Ncnt=Ncnt(T), mom=[float(np.mean(xs**k)) for k in (2, 4, 6)], nprimes=int(len(xs))), open("c3.json", "w"), indent=1)
np.save("zeros249.npy", z); np.save("g2_x.npy", xs)
