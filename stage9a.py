# Log correlated structure for Delta: Selberg CLT, FHK maxima, pair correlation; random models
import cypari2, numpy as np, json, math, time
pari = cypari2.Pari(); pari.allocatemem(6 * 10**9); pari.set_real_precision(19)
out = {}; t0 = time.time()
pari("L=lfunmf(mfinit([1,12]),mfDelta()); Li=lfuninit(L,[6,0.01,2600]);")
T0, T1, step = 500.0, 2500.0, 0.04
ts = np.arange(T0, T1, step)
vals = np.array([complex(pari(f"lfun(Li,6+I*{t})")) for t in ts])
logabs = np.log(np.abs(vals))
np.save("delta_logabs.npy", np.vstack([ts, logabs]))
out["time_values"] = time.time() - t0
# Selberg CLT: compare log|L(1/2+it)| with N(0, (1/2) log log t)
v = 0.5 * np.log(np.log(ts))
z = logabs / np.sqrt(v)
from scipy.stats import kstest, norm, skew, kurtosis
out["selberg"] = {"n": int(len(z)), "mean_logabs": float(logabs.mean()), "var_logabs": float(logabs.var()),
                  "pred_var_mid": float(0.5 * math.log(math.log(1500))), "mean_z": float(z.mean()), "sd_z": float(z.std()),
                  "skew": float(skew(logabs)), "excess_kurtosis": float(kurtosis(logabs)),
                  "ks_standardised_p": float(kstest((logabs - logabs.mean()) / logabs.std(), "norm").pvalue)}
print("selberg", out["selberg"], flush=True)
# FHK: maxima over unit windows
W = int(round(1 / step))
nwin = len(ts) // W
mx = np.array([logabs[i * W:(i + 1) * W].max() for i in range(nwin)]); tc = np.array([ts[i * W + W // 2] for i in range(nwin)])
pred = np.log(np.log(tc)) - 0.75 * np.log(np.log(np.log(tc)))
out["fhk"] = {"n_windows": int(nwin), "mean_max": float(mx.mean()), "sd_max": float(mx.std()), "mean_pred": float(pred.mean()),
              "mean_max_minus_pred": float((mx - pred).mean()), "sd_max_minus_pred": float((mx - pred).std())}
np.save("delta_fhk.npy", np.vstack([tc, mx, pred]))
print("fhk", out["fhk"], flush=True)
# pair correlation and spacings of zeros up to height 3000
t1 = time.time()
zs = np.array([float(g) for g in pari("lfunzeros(L,[0,3000])") if float(g) > 1e-6])
out["n_zeros"] = int(len(zs)); out["time_zeros"] = time.time() - t1
# unfold: N(t) ~ (t/pi) log(t/(2 pi e)) + c ; use smooth N(t) = (t/pi)(log(t/(2pi)) - 1)
Nsm = lambda t: (t / math.pi) * (np.log(t / (2 * math.pi)) - 1)
u = Nsm(zs[zs > 50])
sp = np.diff(u); sp = sp / sp.mean()
np.save("delta_spacings.npy", sp)
# pair correlation of unfolded zeros in [0, 3]
d = []
for i in range(len(u)):
    j = i + 1
    while j < len(u) and u[j] - u[i] < 3: d.append(u[j] - u[i]); j += 1
d = np.array(d); np.save("delta_paircorr.npy", d)
dens = len(u) / (u[-1] - u[0])
h, e = np.histogram(d, bins=np.linspace(0, 3, 31)); c = (e[:-1] + e[1:]) / 2
R2 = h / (len(u) * dens * (e[1] - e[0]))
mont = 1 - (np.sin(np.pi * c) / (np.pi * c)) ** 2
out["pair_correlation"] = {"bins": c.tolist(), "empirical": R2.tolist(), "montgomery": mont.tolist(), "max_abs_dev": float(np.max(np.abs(R2 - mont)))}
# nearest neighbour spacing vs GUE Wigner surmise (32/pi^2) s^2 e^{-4 s^2/pi}
from scipy.integrate import quad
F = lambda s: quad(lambda x: 32 / math.pi**2 * x * x * math.exp(-4 * x * x / math.pi), 0, s)[0]
srt = np.sort(sp); emp = np.arange(1, len(srt) + 1) / len(srt)
grid = np.linspace(0.05, 3, 60); ks = max(abs(np.searchsorted(srt, g) / len(srt) - F(g)) for g in grid)
out["spacings"] = {"n": int(len(sp)), "var": float(sp.var()), "gue_var": 0.178, "ks_vs_wigner_surmise_grid": float(ks)}
print("zeros", out["n_zeros"], out["pair_correlation"]["max_abs_dev"], out["spacings"], flush=True)
json.dump(out, open("stage9a.json", "w"), indent=1)
print("done", time.time() - t0)
