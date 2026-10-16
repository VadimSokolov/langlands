import cypari2, numpy as np, json, math
import mpmath as mp
pari = cypari2.Pari(); pari.allocatemem(2 * 10**9); pari.set_real_precision(38)
mp.mp.dps = 30
out = {}
pari("mf=mfinit([1,12]); L=lfunmf(mf,mfDelta());")
T = 400
z = [float(t) for t in pari(f"lfunzeros(L,{T})")]
z = [t for t in z if t > 1e-8]
out["n_zeros"] = len(z); out["first_zeros"] = z[:6]
lam = lambda s: complex(pari(f"lfunlambda(L,{s.real}+I*{s.imag})")) if isinstance(s, complex) else float(pari(f"lfunlambda(L,{s})"))
L6 = lam(6.0)
out["L6"] = float(pari("lfun(L,6)"))

# Polya type kernel phi(u) = Delta(i e^u) e^{6u}; Delta(iy) = eta(iy)^24 (true eta)
def Delta(y):
    return mp.mpf(float(pari(f"real(eta(I*{y},1)^24)")))
us = np.linspace(-4, 4, 801)
phi = np.array([float(Delta(math.exp(u)) * mp.e ** (6 * u)) for u in us])
du = us[1] - us[0]
Z = phi.sum() * du
varW = float((us ** 2 * phi).sum() * du / Z)
meanW = float((us * phi).sum() * du / Z)
tail = (math.log(T / (2 * math.pi)) + 1) / (math.pi * T)
s2 = sum(1 / t ** 2 for t in z)
out["varW_kernel"] = varW; out["meanW"] = meanW
out["two_sum_inv_gamma2"] = 2 * s2; out["two_sum_with_tail"] = 2 * (s2 + tail)
# check characteristic function: Lambda(6+it)/Lambda(6) = E e^{itW}
cf = []
for t in [2.0, 5.0, z[0], 12.0]:
    emp = complex((phi * np.exp(1j * t * us)).sum() * du / Z)
    cf.append((t, emp.real, lam(complex(6, t)).real / L6))
out["charfun_check"] = cf
# kernel symmetry
out["kernel_asym"] = float(np.max(np.abs(phi - phi[::-1])) / phi.max())
np.save("../work/phi.npy", np.vstack([us, phi / Z]))

# GGC product check: Lambda(6)/Lambda(6+sqrt s) vs prod (1+s/g^2)^{-1} exp(-s*tail)
gg = []
for s in [1, 10, 50, 100, 200, 400]:
    lhs = L6 / lam(6 + math.sqrt(s))
    rhs = math.exp(-sum(math.log1p(s / t ** 2) for t in z) - s * tail)
    gg.append((s, lhs, rhs))
out["ggc_check"] = gg

# extreme value representation: L(s,Delta) = E[Q(s, 2 pi M*)], F(y) = Delta(iy) e^{2 pi y}
def EQ(s):
    f = lambda y: Delta(y) * mp.e ** (2 * mp.pi * y) * (2 * mp.pi) ** s * y ** (s - 1) * mp.e ** (-2 * mp.pi * y) / mp.gamma(s)
    return mp.quad(f, [0.01, 0.1, 0.3, 1, 3, 12])
ev = []
for s in [mp.mpf(3), mp.mpf(6), mp.mpf(10), mp.mpc(6, 5), mp.mpc(6, z[0])]:
    v = complex(EQ(s))
    ex = complex(pari(f"lfun(L,{float(mp.re(s))}+I*{float(mp.im(s))})"))
    ev.append((str(complex(s)), [v.real, v.imag], [ex.real, ex.imag]))
out["extreme_value_check"] = ev
# modular identity F(y) = y^{-12} e^{2 pi (y - 1/y)} F(1/y)
F = lambda y: Delta(y) * mp.e ** (2 * mp.pi * y)
out["modular_check"] = [(y, float(F(y)), float(y ** -12 * mp.e ** (2 * mp.pi * (y - 1 / y)) * F(1 / y))) for y in [0.5, 0.8, 1.3]]
# Monte Carlo: M = max_n E_n/(2 pi n), M* = max of 24 copies, truncated at n <= 400
rng = np.random.default_rng(3); n = np.arange(1, 401)
R = 20000
Ms = np.concatenate([np.max(rng.exponential(size=(1000, 24, 400)) / (2 * np.pi * n), axis=(1, 2)) for _ in range(R // 1000)])
from scipy.special import gammaincc
mc = gammaincc(6, 2 * np.pi * Ms)
out["mc_L6"] = (float(mc.mean()), float(mc.std() / math.sqrt(R)))
out["Mstar_quantiles"] = [float(q) for q in np.quantile(Ms, [.1, .5, .9])]
json.dump(out, open("../work/stage3.json", "w"), indent=1)
print(json.dumps(out, indent=1))
