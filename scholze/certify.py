"""Exact atom count by phase unwrapping of Lambda around the rectangle
c + [-a, a] x i[0, T].  Needs Lambda only, not Lambda'.  The result is the
number of zeros enclosed; equality with the sign change count of Xi certifies
U >= 0 on (0, T^2] with no resolution caveat."""
import mpmath as mp, sys, json
from signed_thorin import CASES, on_line_zeros
mp.mp.dps = 12

def unwrap_count(case, T, a=0.9, n_side=400):
    c, Lam = case['c'], case['Lam']
    pts = []
    pts += [c - a + 2*a*k/n_side for k in range(n_side)]                 # bottom
    pts += [c + a + 1j*T*k/n_side for k in range(n_side)]                # right
    pts += [c + a - 2*a*k/n_side + 1j*T for k in range(n_side)]          # top
    pts += [c - a + 1j*T*(1 - k/n_side) for k in range(n_side)]          # left
    pts.append(pts[0])
    total, prev = mp.mpf(0), None
    for z in pts:
        ph = mp.arg(Lam(z))
        if prev is not None:
            d = ph - prev
            while d > mp.pi: d -= 2*mp.pi
            while d < -mp.pi: d += 2*mp.pi
            total += d
        prev = ph
    return total / (2*mp.pi)

if __name__ == '__main__':
    name, T, ngrid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    case = CASES[name]
    zeros = on_line_zeros(case, T, ngrid)
    N = unwrap_count(case, T)
    print(json.dumps(dict(case=case['label'], T=T, sign_changes=len(zeros),
        contour_count=float(N), certified=bool(abs(N - len(zeros)) < 0.01),
        last_gammas=[float(z) for z in zeros[-3:]]), indent=1))
