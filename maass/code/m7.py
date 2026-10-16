"""The Laplace eigenvalues seen in the geodesic count: (psi_Gamma(x) - x)/sqrt(x) on a log grid and its periodogram in u = log x,
whose peaks sit at the r_j (explicit formula psi_Gamma(x) = x + sum_j x^{s_j}/s_j + ..., s_j = 1/2 + i r_j)."""
import numpy as np, json, math
d = np.load("geodesics.npz"); N = d["norms"]; m = d["mults"]; L = np.log(N)
# all prime powers with norm <= 1e6
norms = []; wts = []
for n, mm, l in zip(N, m, L):
    k = 1
    while n**k <= 1e6: norms.append(n**k); wts.append(mm*l); k += 1
norms = np.array(norms); wts = np.array(wts); o = np.argsort(norms); norms = norms[o]; wts = wts[o]; cw = np.cumsum(wts)
u = np.linspace(math.log(50), math.log(1e6), 20000); x = np.exp(u)
psi = cw[np.searchsorted(norms, x, side="right") - 1]
f = (psi - x)/np.sqrt(x)
# periodogram
f0 = f - f.mean(); w = np.hanning(len(f0)); du = u[1]-u[0]
rs = np.linspace(2, 30, 1400)
per = np.array([abs(np.sum(f0*w*np.exp(-1j*r*u)))*du for r in rs])
rj = [9.533695, 12.173008, 13.779751, 14.358510, 16.138073, 16.644259, 17.738563, 18.180918, 19.423481, 19.484715]
print("peaks near:", [float(rs[i]) for i in np.argsort(per)[-6:]])
np.savez("m7.npz", u=u, f=f, rs=rs, per=per, rj=rj)
