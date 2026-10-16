"""Thorin form of L(s, chi_4): Xi(sqrt(lambda))/Xi(0) = prod_k (1 + lambda/gamma_k^2), Xi(w) = Lambda(1/2 + w, chi_4),
Lambda(s) = (4/pi)^{s/2} Gamma((s+1)/2) L(s, chi_4). Zeros from PARI lfunzeros to height T, tail by the smoothed counting function
N(t) ~ (t/2pi) log(4t/(2 pi e)) + 1/4 ... (for chi_4 the zero counting function is (t/pi) log(4t/(2 pi e))/2 ... computed by lfunzeros count check)."""
import cypari2, numpy as np, json, math
pari = cypari2.Pari(); pari.allocatemem(2*10**9); pari.default("realprecision", 25)
L = pari.lfuncreate(pari("Mod(3,4)"))
T = 3000.0
z = np.array([float(g) for g in pari.lfunzeros(L, T)])
print("zeros below", T, ":", len(z), "first", z[:5])
# empirical counting density check: N(T) vs (T/(2 pi)) log(4 T/(2 pi e)) ... fit
def Ncount(t): return (t/(2*math.pi))*math.log(4*t/(2*math.pi*math.e))
print("N(T) formula", Ncount(T), "actual", len(z))
def Lam(s):
    s = pari(s)
    return float((4/pari.pi())**(s/2)*pari.gamma((s+1)/2)*pari.lfun(L, s))
X0 = Lam(0.5)
def tail(lam):
    # integral_T^inf log(1 + lam/t^2) dN(t), dN = (1/2pi) log(4t/(2pi)) dt
    from scipy.integrate import quad
    f = lambda t: math.log1p(lam/t**2)*math.log(4*t/(2*math.pi))/(2*math.pi)
    v, _ = quad(f, T, np.inf, limit=200)
    return v
rows = []
for lam in [1, 4, 16, 36, 64, 100, 400]:
    lhs = Lam(0.5 + math.sqrt(lam))/X0
    lp = np.sum(np.log1p(lam/z**2)); tl = tail(lam)
    rhs = math.exp(lp + tl)
    rows.append((lam, lhs, rhs, lhs/rhs - 1)); print("lambda", lam, "Xi ratio", lhs, "product", rhs, "rel err", lhs/rhs-1)
json.dump(dict(T=T, nzeros=int(len(z)), first=z[:6].tolist(), rows=rows), open("g5.json","w"), indent=1)
np.save("chi4_zeros.npy", z)
