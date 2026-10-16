# Arithmetic statistics: Cohen Lenstra for class groups; Poonen Rains for 2 Selmer; ranks
import cypari2, numpy as np, json, math, time
pari = cypari2.Pari(); pari.allocatemem(4 * 10**9)
out = {}; t0 = time.time()
def eta(p, start=1): return math.prod(1 - p ** (-k) for k in range(start, 200))
def cl_imag(p, r): return p ** (-r * r) * eta(p) / math.prod((1 - p ** (-k)) ** 2 for k in range(1, r + 1))
def cl_real(p, r): return p ** (-r * (r + 1)) * eta(p) / (math.prod(1 - p ** (-k) for k in range(1, r + 1)) * math.prod(1 - p ** (-k) for k in range(1, r + 2)))
def prank(cyc, p): return sum(1 for c in cyc if c % p == 0)
res = {}
for kind, lo, hi in (("imaginary", -2 * 10**6, -10**6), ("real", 10**5, 4 * 10**5)):
    rk = {3: [], 5: [], 7: []}; Ds = []
    pari(f"V=select(d->isfundamental(d),[{lo}..{hi}]);")
    n = int(pari("#V"))
    pari("C=vector(#V,i,quadclassunit(V[i]).cyc);")
    for i in range(1, n + 1):
        cyc = [int(c) for c in pari(f"C[{i}]")]
        for p in rk: rk[p].append(prank(cyc, p))
    Ds = [int(d) for d in pari("V")]
    f = cl_imag if kind == "imaginary" else cl_real
    res[kind] = {"range": [lo, hi], "n": n}
    for p in rk:
        a = np.array(rk[p])
        res[kind][str(p)] = {"observed": [float(np.mean(a == r)) for r in range(4)], "cohen_lenstra": [f(p, r) for r in range(4)],
                             "p_divides_h_obs": float(np.mean(a > 0)), "p_divides_h_cl": 1 - f(p, 0)}
    a3 = np.array(rk[3]); res[kind]["mean_size_Cl3"] = float(np.mean(3.0 ** a3)); res[kind]["dh_prediction"] = 2.0 if kind == "imaginary" else 4 / 3
    # convergence of the mean of |Cl[3]| in windows of |D|
    Da = np.abs(np.array(Ds)); order = np.argsort(Da); a3s = a3[order]
    res[kind]["windows"] = [(int(Da[order][k[0]]), float(np.mean(3.0 ** a3s[k]))) for k in np.array_split(np.arange(n), 10)]
    np.save(f"cl3_{kind}.npy", np.vstack([np.array(Ds), a3, np.array(rk[5]), np.array(rk[7])]))
    print(kind, n, {p: res[kind][str(p)]["observed"][:3] for p in (3, 5, 7)}, res[kind]["mean_size_Cl3"], time.time() - t0, flush=True)
out["cohen_lenstra"] = res
# Poonen Rains: 2 Selmer ranks in the family y^2 = x^3 + a x + b with no rational 2 torsion
c = math.prod(1 / (1 + 2.0 ** (-j)) for j in range(0, 200))
pr = lambda d: c * math.prod(2 / (2 ** j - 1) for j in range(1, d + 1))
sel = []; ranks = []; roots = []; unresolved = 0; heights = []
for a in range(-40, 41):
    for b in range(-40, 41):
        if 4 * a**3 + 27 * b**2 == 0: continue
        if len(pari(f"nfroots(,x^3+{a}*x+{b})")) > 0: continue
        E = pari(f"ellinit([{a},{b}])")
        r = pari.ellrank(E)
        if int(r[0]) != int(r[1]): unresolved += 1; continue
        sel.append(int(r[1]) + int(r[2])); ranks.append(int(r[0])); roots.append(int(pari.ellrootno(E)))
        heights.append(max(4 * abs(a) ** 3, 27 * b * b))
sel = np.array(sel); ranks = np.array(ranks); roots = np.array(roots)
out["poonen_rains"] = {"curves": int(len(sel)), "unresolved": unresolved,
                       "sel2_observed": [float(np.mean(sel == d)) for d in range(6)], "poonen_rains": [pr(d) for d in range(6)],
                       "mean_size_sel2": float(np.mean(2.0 ** sel)), "bhargava_shankar_mean": 3.0,
                       "rank_dist": [float(np.mean(ranks == d)) for d in range(5)], "mean_rank": float(ranks.mean()),
                       "root_number_plus": float(np.mean(roots == 1)), "parity_consistent": float(np.mean((ranks % 2 == 0) == (roots == 1)))}
np.save("selmer.npy", np.vstack([sel, ranks, roots, np.array(heights)]))
print(out["poonen_rains"], flush=True)
json.dump(out, open("stage9c.json", "w"), indent=1)
print("done", time.time() - t0)
