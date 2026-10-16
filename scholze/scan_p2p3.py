import mpmath as mp, sys, json
from signed_thorin import CASES, on_line_zeros, u_eps
mp.mp.dps = 12
name, T, eps, n_tau, ngrid = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
case = CASES[name]
zeros = on_line_zeros(case, T, ngrid)
taus = [mp.mpf(T)**2*k/n_tau for k in range(1, n_tau+1)]
dtau = mp.mpf(T)**2/n_tau
us = [u_eps(case, t, eps) for t in taus]
vs = [sum(eps/(mp.pi*((t-g**2)**2+eps**2)) for g in zeros) for t in taus]
tv = sum(abs(u-v) for u,v in zip(us,vs))*dtau
mass = sum(us)*dtau
print(json.dumps(dict(case=case['label'], T=T, eps=eps, n_zeros=len(zeros),
   min_u=float(min(us)), argmin_tau=float(taus[us.index(min(us))]),
   excluded_delta_at_top=float(eps/(2*T)), mass=float(mass), tv=float(tv),
   peak_height_expected=float(1/(mp.pi*eps)), max_u=float(max(us))), indent=1))
