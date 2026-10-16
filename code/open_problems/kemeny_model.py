import json, numpy as np, sys, cypari2, math
from scipy import sparse
pari=cypari2.Pari()
f=sys.argv[1]; o=json.load(open(f)); l=o["l"]
d0s=o["d0"]; w=np.array(o["w"]); c=o["c"]
# random model for primes N = 1 mod 12: chi_d0(N) = prod_{p>=5 | d0} eta_p * eta_8^{[8|d0]}, eta iid signs
rows=[];cols=[];pidx={}
for i,d in enumerate(d0s):
    m=abs(d); ps=[int(p) for p in pari.factor(m)[0]]
    for p in ps:
        if p==3: continue
        if p==2:
            if m%8!=0: continue
            key=8
        else: key=p
        if key not in pidx: pidx[key]=len(pidx)
        rows.append(i); cols.append(pidx[key])
M=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(d0s),len(pidx)))
rng=np.random.default_rng(1); R=20000
E=np.empty(R)
for b in range(0,R,500):
    neg=(rng.random((len(pidx),500))<0.5).astype(np.float32)
    par=np.asarray(M@neg)%2; E[b:b+500]=c+w@(1-2*par)
nn=np.diff(M.indptr); det=[(d,round(wi,4)) for d,wi,k in zip(d0s,w,nn) if k==0]
print("deterministic d0",det[:6], "model mean",E.mean(),"sd",E.std(),"q",np.percentile(E,[5,50,95]))
np.save(f.replace(".json","_model.npy"),E)
# exact series value for actual levels N, using kronecker
rows2=json.load(open("../stage2.json"))["rows"]
ex=[]
for r in rows2:
    N=r["N"]; chiN=np.array([int(pari.kronecker(d,N)) for d in d0s])
    ex.append((N,r["h"],r["kemeny2"]-2*r["h"] if l==2 else r["kemeny3"]-1.5*r["h"], c+w@chiN))
ex=np.array(ex); np.save(f.replace(".json","_levels.npy"),ex)
print("observed mean sd",ex[:,2].mean(),ex[:,2].std(),"series(J) at same N mean sd",ex[:,3].mean(),ex[:,3].std(),"corr",np.corrcoef(ex[:,2],ex[:,3])[0,1])
