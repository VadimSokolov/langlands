"""
Signed Thorin measures of the clocks G(s) = Lambda(c + sqrt s) / Lambda(c)
for zeta, the Dirichlet character chi_12, and the Ramanujan form Delta.

Stieltjes inversion.  If 1/G is a GGC with Thorin measure U and no drift then
    G'(s)/G(s) = int U(dtau) / (tau + s),
so the eps smoothed signed density is
    u_eps(tau) = -(1/pi) Im [ G'/G ](-tau + i eps),
with G'/G(s) = (Lambda'/Lambda)(c + w) / (2 w),  w = sqrt(s), Im w > 0.
No hypothesis is used: u_eps is defined from Lambda'/Lambda alone.
Under GRH, U = sum_gamma delta_{gamma^2}, and u_eps is a sum of unit mass
Lorentzians of width eps centred at gamma^2.  An off line zero c + d + i g
gives the conjugate pair of atoms at (g -+ i d)^2 and a NEGATIVE lobe of
u_eps once eps < 2 g |d|.
"""
import mpmath as mp
import sys, json

mp.mp.dps = 20

# ---------------------------------------------------------------- zeta
def LamZ(s):
    return mp.mpf('0.5') * s * (s - 1) * mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s)

def dlogLamZ(s):
    return 1 / s + 1 / (s - 1) - mp.log(mp.pi) / 2 + mp.digamma(s / 2) / 2 + mp.zeta(s, derivative=1) / mp.zeta(s)

# ---------------------------------------------------------------- chi_12
CHI12 = [0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 1]
Q12 = 12

def LamChi(s):
    return (mp.mpf(Q12) / mp.pi) ** (s / 2) * mp.gamma(s / 2) * mp.dirichlet(s, CHI12)

def dlogLamChi(s):
    L = mp.dirichlet(s, CHI12)
    dL = mp.dirichlet(s, CHI12, derivative=1)
    return mp.log(mp.mpf(Q12) / mp.pi) / 2 + mp.digamma(s / 2) / 2 + dL / L

# ---------------------------------------------------------------- Delta
def Delta_iy(y):
    q = mp.exp(-2 * mp.pi * y)
    prod = mp.mpf(1)
    n = 1
    while True:
        qn = q ** n
        if qn < mp.mpf(10) ** (-mp.mp.dps - 5):
            break
        prod *= (1 - qn) ** 24
        n += 1
    return q * prod

def LamD(s):
    f = lambda y: Delta_iy(y) * (y ** s + y ** (12 - s)) / y
    return mp.quad(f, [1, 2, 4, 8, mp.inf])

def dLamD(s):
    f = lambda y: Delta_iy(y) * (y ** s - y ** (12 - s)) * mp.log(y) / y
    return mp.quad(f, [1, 2, 4, 8, mp.inf])

def dlogLamD(s):
    return dLamD(s) / LamD(s)

CASES = {
    'zeta':  dict(c=mp.mpf('0.5'), Lam=LamZ,   dlog=dlogLamZ,   label='Riemann zeta'),
    'chi12': dict(c=mp.mpf('0.5'), Lam=LamChi, dlog=dlogLamChi, label='L(s, chi_12)'),
    'delta': dict(c=mp.mpf(6),     Lam=LamD,   dlog=dlogLamD,   label='L(s, Delta)'),
}

def u_eps(case, tau, eps):
    c, dlog = case['c'], case['dlog']
    s = mp.mpc(-tau, eps)
    w = mp.sqrt(s)                 # principal branch, Im w > 0
    GG = dlog(c + w) / (2 * w)
    return -mp.im(GG) / mp.pi

def xi_line(case, t):
    """Real Xi(t) = Lambda(c + i t), real for self dual cases."""
    return mp.re(case['Lam'](case['c'] + 1j * t))

def on_line_zeros(case, T, n_grid):
    """Zeros of Xi on (0,T] by sign changes and bisection."""
    ts = [mp.mpf(T) * k / n_grid for k in range(1, n_grid + 1)]
    vals = [xi_line(case, t) for t in ts]
    zeros = []
    for i in range(len(ts) - 1):
        if vals[i] == 0:
            zeros.append(ts[i])
        elif vals[i] * vals[i + 1] < 0:
            z = mp.findroot(lambda t: xi_line(case, t), (ts[i], ts[i + 1]), solver='bisect', tol=1e-14)
            zeros.append(z)
    return zeros

def scan(case, T, eps, n_tau):
    """Scan u_eps on (0, T^2]; return grid, min, and integrated mass."""
    taus = [mp.mpf(T) ** 2 * k / n_tau for k in range(1, n_tau + 1)]
    us = [u_eps(case, tau, eps) for tau in taus]
    dtau = mp.mpf(T) ** 2 / n_tau
    mass = sum(us) * dtau
    umin = min(us)
    tmin = taus[us.index(umin)]
    return taus, us, umin, tmin, mass

if __name__ == '__main__':
    name = sys.argv[1]
    T = float(sys.argv[2])
    eps = float(sys.argv[3])
    n_tau = int(sys.argv[4])
    n_grid = int(sys.argv[5])
    case = CASES[name]

    zeros = on_line_zeros(case, T, n_grid)
    taus, us, umin, tmin, mass = scan(case, T, eps, n_tau)

    # locate peaks of u_eps and compare with gamma^2
    peaks = []
    for i in range(1, len(us) - 1):
        if us[i] > us[i - 1] and us[i] > us[i + 1] and us[i] > 1 / (mp.pi * eps) * 0.2:
            peaks.append((taus[i], us[i]))

    out = dict(
        case=case['label'], T=T, eps=eps,
        n_on_line_zeros=len(zeros),
        gammas=[float(z) for z in zeros],
        gamma_sq=[float(z ** 2) for z in zeros],
        peaks_tau=[float(p[0]) for p in peaks],
        peaks_sqrt=[float(mp.sqrt(p[0])) for p in peaks],
        min_u=float(umin), argmin_tau=float(tmin),
        lorentzian_peak=float(1 / (mp.pi * eps)),
        integrated_mass=float(mass),
    )
    print(json.dumps(out, indent=1))
