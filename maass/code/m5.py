"""Thorin density of the geodesic Euler product R(s) = Z(s+1)/Z(s) = prod_P (1 - N(P)^{-s})^{-1} at basepoint alpha > 1,
by the sine squared lemma: U(z) = (1/(2 pi sqrt z)) sum_{P,k} l(P) N(P)^{-k alpha} 2 sin^2(k l(P) sqrt z / 2), l(P) = log N(P).
Also the identity R(alpha)/R(alpha + sqrt z) = exp(-b sqrt z) E exp(-z H) checked by direct evaluation of both sides from the geodesics."""
import numpy as np, json
d = np.load("geodesics.npz"); N = d["norms"]; m = d["mults"]; l = np.log(N)
alpha = 2.0
def U(z):
    r = np.sqrt(z); tot = 0.0
    for k in range(1, 40):
        tot += np.sum(m*l*N**(-k*alpha)*2*np.sin(k*l*r/2)**2)
    return tot/(2*np.pi*r)
zs = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
Us = np.array([U(z) for z in zs])
print("U(z) at alpha=2:", dict(zip(zs.tolist(), np.round(Us, 6).tolist())))
# check the representation: log R(alpha+s) = sum_{P,k} N(P)^{-k(alpha+s)}/k
def logR(s): return sum(np.sum(m*N**(-k*s))/k for k in range(1, 60))
b = -sum(np.sum(m*l*N**(-k*alpha)) for k in range(1, 60))   # (log R)'(alpha) < 0
from scipy.integrate import quad
for z in [0.5, 2, 10]:
    lhs = logR(alpha) - logR(alpha + np.sqrt(z))           # log [R(alpha)/R(alpha+sqrt z)]
    # rhs: -b sqrt z - int log(1 + z/t) U(t) dt
    integ = quad(lambda t: np.log1p(z/t)*U(t), 0, np.inf, limit=400)[0]
    rhs = -b*np.sqrt(z) - integ
    print("z", z, "lhs", lhs, "rhs", rhs, "diff", lhs - rhs)
json.dump(dict(alpha=alpha, b=b, zs=zs.tolist(), Us=Us.tolist()), open("m5.json", "w"), indent=1)
np.save("U_alpha2.npy", np.array([ (z, U(z)) for z in np.linspace(0.01, 60, 400)]))
