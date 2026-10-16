import cypari2, numpy as np, json, math, time
pari = cypari2.Pari(); pari.allocatemem(2 * 10**9)
pari('read("ss_graph.gp")')
def graph(p, l):
    pari(f"A=ssgraph({p},{l})"); n = int(pari("#A"))
    return np.array([[int(pari(f"A[{i},{j}]")) for j in range(1, n + 1)] for i in range(1, n + 1)], dtype=float)
def U(n, x):
    t = np.arccos(np.clip(x / 2, -1, 1)); s = np.sin(t); return np.sin((n + 1) * t) / np.where(s > 1e-12, s, 1e-12)
out = {}; t0 = time.time()
Ns = [int(z) for z in pari("select(p->p%12==1, primes([1000,8000]))")]
pool = {2: [], 3: []}; rows = []
for N in Ns:
    row = {"N": N}
    for l in (2, 3):
        M = graph(N, l); h = len(M); ev = np.sort(np.linalg.eigvalsh(M))
        cusp = ev[:-1]; pool[l] += list(cusp / math.sqrt(l))
        row[f"h"] = h
        row[f"maxabs{l}"] = float(np.max(np.abs(cusp)) / (2 * math.sqrt(l)))
        row[f"kemeny{l}"] = float(np.sum(1 / (1 - cusp / (l + 1))))
        row[f"K_over_h{l}"] = row[f"kemeny{l}"] / h
        row[f"ihara{l}"] = float(max(abs(abs(r) - 1 / math.sqrt(l)) for x in cusp for r in np.roots([l, -x, 1])))
    rows.append(row)
out["n_levels"] = len(Ns); out["rows"] = rows
out["vertical"] = {l: {"n": len(pool[l]), "U": [(n, float(np.mean(U(n, np.array(pool[l])))), (l ** (-n / 2) if n % 2 == 0 else 0)) for n in (1, 2, 3, 4, 6, 8)],
                       "max_ratio": float(max(r[f"maxabs{l}"] for r in rows))} for l in (2, 3)}
np.save("../work/vert2.npy", np.array(pool[2])); np.save("../work/vert3.npy", np.array(pool[3]))
# hitting times on explicit graphs
hit = []
for p in (1009, 2017, 4129):
    M = graph(p, 2); n = len(M); P = M / 3; I = np.eye(n); H = np.zeros((n, n))
    for v in range(n):
        idx = [i for i in range(n) if i != v]
        H[idx, v] = np.linalg.solve((I - P)[np.ix_(idx, idx)], np.ones(n - 1))
    ev = np.sort(np.linalg.eigvalsh(M))
    hit.append(dict(p=p, h=n, kemeny_direct=float(H.mean()), kemeny_spec=float(np.sum(1 / (1 - ev[:-1] / 3))),
                    mean_hit_over_h=float(H.mean() / n), max_hit_over_h=float(H.max() / n),
                    ret=[(t, float(np.mean(np.diag(np.linalg.matrix_power(P, t))) - 1 / n), float((2 * math.sqrt(2) / 3) ** t)) for t in (10, 20, 40)]))
    if p == 4129: np.save("../work/hit4129.npy", H); np.save("../work/adj4129.npy", M)
out["hitting"] = hit; out["time"] = time.time() - t0
json.dump(out, open("../work/stage2.json", "w"), indent=1)
print(json.dumps(out["vertical"], indent=1)); print(json.dumps(hit, indent=1))
for r in rows[::10]: print(r)
