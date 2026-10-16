"""Brownian motion on the modular surface X = SL(2,Z) \\ H, generator Delta = y^2 (d_x^2 + d_y^2).
In the half plane: dx = sqrt(2) y dB_1, d log y = sqrt(2) dB_2 - dt. Each step the point is reduced to the
standard fundamental domain. Checks ergodicity against the normalised hyperbolic area (3/pi) dx dy / y^2:
P(y > Y) = 3/(pi Y) for Y >= 1, and the law of y on y >= 1 has density 3/(pi y^2)."""
import numpy as np, json, sys
rng = np.random.default_rng(1)
T = float(sys.argv[1]) if len(sys.argv) > 1 else 2e5
dt = 1e-3; nsteps = int(T/dt)
npaths = 64
x = rng.uniform(-0.5, 0.5, npaths); ly = np.log(rng.uniform(1.0, 1.5, npaths))
def reduce(x, y):
    # reduce (x + i y) to |x| <= 1/2, x^2 + y^2 >= 1
    for _ in range(60):
        x = x - np.round(x)
        r2 = x*x + y*y
        m = r2 < 1
        if not m.any(): break
        x = np.where(m, -x/r2, x); y = np.where(m, y/r2, y)
    return x, y
ys = []; sample_every = 100
cnt_gt = np.zeros(4); Ys = [1.0, 2.0, 4.0, 8.0]; total = 0
traj = []
for step in range(nsteps):
    y = np.exp(ly)
    dB = rng.standard_normal((2, npaths))*np.sqrt(dt)
    x = x + np.sqrt(2)*y*dB[0]
    ly = ly + np.sqrt(2)*dB[1] - dt
    x, y = reduce(x, np.exp(ly)); ly = np.log(y)
    if step % sample_every == 0:
        ys.append(y.copy()); total += npaths
        for i, Y in enumerate(Ys): cnt_gt[i] += np.sum(y > Y)
        if step % (sample_every*50) == 0: traj.append((x[0], y[0]))
ys = np.concatenate(ys)
res = {}
for i, Y in enumerate(Ys):
    emp = cnt_gt[i]/total; th = 3/(np.pi*Y)
    res[str(Y)] = (emp, th); print("P(y > %g): empirical %.4f  theory %.4f" % (Y, emp, th))
# density of y on [1, 8] in log bins
bins = np.geomspace(1, 32, 21); h, _ = np.histogram(ys, bins=bins); dens = h/(total*np.diff(bins))
mid = np.sqrt(bins[1:]*bins[:-1]); th = 3/(np.pi*mid**2)
print("density ratio empirical/theory by bin:", np.round(dens/th, 3))
# time average of a test function: the Eisenstein series is unbounded; use f = y^{-1} 1_{y>=1} with integral (3/pi) * int_1^inf y^{-3} dy = 3/(2pi)
f = np.mean(np.where(ys >= 1, 1/ys, 0.0)); print("mean of y^{-1} on y>=1:", f, "theory", 3/(2*np.pi))
json.dump(dict(T=T, dt=dt, npaths=npaths, tails=res, dens=(mid.tolist(), (dens/th).tolist()), f=(float(f), 3/(2*np.pi))), open("m2.json", "w"), indent=1)
np.savez("bm_traj.npz", traj=np.array(traj), ys=ys[:200000])
