"""Tail of the Thorin variable of zeta at alpha = 2: the data of Section sec:laptail and of the right panel of Figure
fig:lapzeta of Chapter 23 (book/chapters/ch_laplace*.tex).

Written for the book on 2026-10-13; not a script of the coauthor. It is taken from the independent check
misc/data-checks/ch_laplace/d2/zeta_indep.py (Parts A to D) of the data review. No script of the coauthor writes the tail
data (the entries H2, tailA and logzeta3 of zeta_switch.json, which the coauthor's figs.py read), and the mass 1.0006 and
mean 0.4287 stored there are trapezoid sums with a quadrature error; this script writes the same density and replaces the
two sums by exact values. Run it on Hopper through misc/tools/hopper_run.sh, never locally (about a minute).
Reads nothing; writes zeta_tail.json, which figs.py reads.

H is the Thorin variable of zeta at alpha = 2 (Lemma lem:10dsin2, Theorem thm:laptail), with Laplace transform
    phi(s) = E exp(-sH) = exp(b sqrt s) zeta(2)/zeta(2 + sqrt s),    b = zeta'/zeta(2) < 0.
With f_k = (log zeta)^(k)(2): E H = f_2/2, and the density p of H satisfies p(t) = A t^(-5/2) + B t^(-7/2) + ..., with
A = -f_3/(8 sqrt pi) and B = (f_2 f_3/12 - f_5/120)/Gamma(-5/2), the transfers (Doetsch) of the terms in s^(3/2) and s^(5/2)
of phi at 0; so P(H > t) ~ (2/3) A t^(-3/2) and p(t) t^(5/2)/A = 1 + (B/A)/t + ...
Entries of zeta_tail.json:
  t, dens          p on the coauthor's grid (60 points evenly spaced on [0.0005, 0.05], 160 geometrically spaced on
                   [0.051, 400]) by Talbot inversion of phi (mpmath invertlaplace at 25 digits, as in zeta_switch.py);
                   below t = 0.002 the density is under 1e-22 and the two values there are rounding noise
  b, f2, f3, f5, EH, A, B
  mass             P(0.0005 <= H <= 400), from Talbot inversions of phi(s)/s (the distribution function) at both ends
  surv400          P(H > 400), from (1 - phi(s))/s; surv400_tail is (2/3) A 400^(-3/2) + (2/5) B 400^(-5/2)
  mean400          E[H; H <= 400], from -phi'(s)/s; EH - mean400 is the mean carried beyond 400, against
                   tailmean400 = A/10 + B/12000 (the integral of t (A t^(-5/2) + B t^(-7/2)) over t > 400)
  ratios           p(t) t^(5/2)/A at the grid points nearest 50, 100, 200 and 400, with the prediction 1 + (B/A)/t
  trapezoid_mass, trapezoid_mean   the trapezoid sums over the grid (the coauthor's 1.0006 and 0.4287), for the record
"""
import json
import time

import mpmath as mp
import numpy as np

T0 = time.time()
ALPHA = 2


def logzeta(x):
    return mp.log(mp.zeta(x))


def make_phi(dps):
    """phi at the working precision dps, with b computed at that precision (as zeta_switch.py does at 25 digits)."""
    mp.mp.dps = dps
    al = mp.mpf(ALPHA)
    b = mp.zeta(al, derivative=1) / mp.zeta(al)
    return (lambda s: mp.exp(b * mp.sqrt(s)) * mp.zeta(al) / mp.zeta(al + mp.sqrt(s))), b, al


# constants, 30 digits
mp.mp.dps = 30
two = mp.mpf(ALPHA)
b = mp.zeta(two, derivative=1) / mp.zeta(two)
fk = {k: mp.diff(logzeta, two, k) for k in (2, 3, 5)}
A = -fk[3] / (8 * mp.sqrt(mp.pi))
B = (fk[2] * fk[3] / 12 - fk[5] / 120) / mp.gamma(mp.mpf(-5) / 2)
EH = fk[2] / 2
out = {"alpha": ALPHA, "b": float(b), "f2": float(fk[2]), "f3": float(fk[3]), "f5": float(fk[5]), "EH": float(EH),
       "A": float(A), "B": float(B), "B_over_A": float(B / A)}
print("b %.12f  E H %.12f  A %.12f  B/A %.6f  (%.1f s)" % (out["b"], out["EH"], out["A"], out["B_over_A"], time.time() - T0),
      flush=True)

# the density on the coauthor's grid, Talbot at 25 digits
phi25, _, _ = make_phi(25)
t = np.concatenate([np.linspace(0.0005, 0.05, 60), np.geomspace(0.051, 400, 160)])
dens = np.array([float(mp.invertlaplace(phi25, float(x), method="talbot")) for x in t])
out["t"] = t.tolist()
out["dens"] = dens.tolist()
i = int(np.argmax(dens))
print("density on %d points of [%.4g, %.4g]: maximum %.6f at t = %.4f  (%.1f s)" % (len(t), t[0], t[-1], dens[i], t[i],
                                                                                 time.time() - T0), flush=True)

# mass, survival and truncated mean, Talbot at 30 digits
phi30, b30, al30 = make_phi(30)


def dphi30(s):
    w = mp.sqrt(s)
    return phi30(s) * (b30 - mp.zeta(al30 + w, derivative=1) / mp.zeta(al30 + w)) / (2 * w)


def inv(G, x):
    mp.mp.dps = 30
    return float(mp.invertlaplace(G, x, method="talbot"))


cdf_lo = inv(lambda s: phi30(s) / s, 0.0005)
cdf_hi = inv(lambda s: phi30(s) / s, 400.0)
out["cdf_0.0005"] = cdf_lo
out["cdf_400"] = cdf_hi
out["mass"] = cdf_hi - cdf_lo
out["surv400"] = inv(lambda s: (1 - phi30(s)) / s, 400.0)
out["surv400_tail"] = float(2 * A / 3 * mp.mpf(400) ** mp.mpf(-1.5) + 2 * B / 5 * mp.mpf(400) ** mp.mpf(-2.5))
out["mean400"] = inv(lambda s: -dphi30(s) / s, 400.0)
out["EH_minus_mean400"] = out["EH"] - out["mean400"]
out["tailmean400"] = float(A / 10 + B / 12000)
out["trapezoid_mass"] = float(np.trapezoid(dens, t))
out["trapezoid_mean"] = float(np.trapezoid(t * dens, t))
print("mass on [0.0005, 400] %.10f (trapezoid %.10f); P(H > 400) %.6e (tail formula %.6e)"
      % (out["mass"], out["trapezoid_mass"], out["surv400"], out["surv400_tail"]), flush=True)
print("mean on [0, 400] %.10f (trapezoid %.10f); E H minus it %.7f (tail formula %.7f)"
      % (out["mean400"], out["trapezoid_mean"], out["EH_minus_mean400"], out["tailmean400"]), flush=True)

rows = []
for target in (50, 100, 200, 400):
    j = int(np.argmin(np.abs(t - target)))
    rows.append({"target": target, "t": float(t[j]), "ratio": float(dens[j] * t[j] ** 2.5 / out["A"]),
                 "predicted": 1 + out["B_over_A"] / float(t[j])})
    print("p(t) t^(5/2)/A at t = %.2f: %.6f (1 + (B/A)/t = %.6f)" % (rows[-1]["t"], rows[-1]["ratio"], rows[-1]["predicted"]),
          flush=True)
out["ratios"] = rows
out["meta"] = {"talbot_dps": 25, "constants_dps": 30, "mpmath": mp.__version__, "numpy": np.__version__,
               "seconds": round(time.time() - T0, 1)}
json.dump(out, open("zeta_tail.json", "w"))
print("wrote zeta_tail.json in %.1f s" % (time.time() - T0), flush=True)
