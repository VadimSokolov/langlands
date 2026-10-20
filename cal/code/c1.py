"""Parity from positivity in every weight.
Delta_w(x) = W^{hol} - W^{even} for the gamma factors Gamma_C(s+w/2) and Gamma_R(s+w/2)^2:
direct archimedean integral against the closed form (1/(4 sqrt(pi x))) int_0^inf e^{-wu/2} sech(u/2) e^{-u^2/4x} du = (1/4) E e^{-x C_w^2},
C_w = sech variable + Cauchy of scale w/2. Then the heat trace of 11a1 (weight 2, w = 1) from its zeros, with both gamma factors."""
import numpy as np, json, cypari2
from mpmath import mp, mpf, quad, digamma, exp, sqrt, pi, cosh, inf, mpc, re
mp.dps = 20
def delta_direct(x, w):
    f = lambda t: exp(-x*t*t)*re(digamma((mpf(3)/2 + mpf(w)/2 + 1j*t)/2) - digamma((mpf(1)/2 + mpf(w)/2 + 1j*t)/2))
    return quad(f, [-inf, inf])/(4*pi)
def delta_closed(x, w):
    g = lambda u: exp(-mpf(w)*u/2)/cosh(u/2)*exp(-u*u/(4*x))
    return quad(g, [0, inf])/(4*sqrt(pi*x))
rows = []
for w in [0, 1, 3, 11]:
    for x in [0.1, 1, 10]:
        a = float(delta_direct(mpf(x), w)); b = float(delta_closed(mpf(x), w)); rows.append((w, x, a, b))
        print("w", w, "x", x, "direct", a, "closed", b, "diff", a - b)
# density of C_w: f_w(t) = (1/2pi) FT of e^{-w|u|/2} sech(u/2) = convolution of sech(pi t) with Cauchy(w/2)
def f_w(t, w):
    if w == 0: return float(1/cosh(pi*t))
    a = mpf(w)/2
    return float(quad(lambda s: (1/cosh(pi*(t - s)))*(a/pi)/(a*a + s*s), [-inf, inf]))
print("f_w(0):", {w: f_w(0, w) for w in [0, 1, 3, 11]})
# 11a1 heat trace from zeros
pari = cypari2.Pari(); pari.allocatemem(10**9)
E = pari.ellinit([0, -1, 1, -10, -20]); L = pari.lfuncreate(E)
T = 300.0; z = np.array([float(g) for g in pari.lfunzeros(L, T)])
print("zeros of 11a1 below", T, ":", len(z), z[:4])
def Whol(x):  # 1/2 sum over zeros 1/2 +- i gamma: both signs
    s = np.sum(np.exp(-x*z**2))  # sum over gamma > 0 counts the pair once each; W = 1/2 sum over all zeros = sum over gamma>0
    # tail: N(t) ~ (t/pi) log(sqrt(11) t/(2 pi e)) for degree 2 conductor 11
    from scipy.integrate import quad as q
    tail = q(lambda t: np.exp(-x*t*t)*np.log(np.sqrt(11)*t/(2*np.pi))/np.pi, T, np.inf)[0]
    return s + tail
xs = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
out = []
for x in xs:
    wh = Whol(x); d = float(delta_closed(mpf(x), 1)); out.append((x, wh, wh - d, d))
    print("x", x, "W_hol", wh, "W_even", wh - d, "Delta_1", d)
json.dump(dict(rows=rows, f0={w: f_w(0, w) for w in [0, 1, 3, 11]}, nzeros=int(len(z)), heat=out), open("c1.json", "w"), indent=1)
np.save("zeros11a1.npy", z)
