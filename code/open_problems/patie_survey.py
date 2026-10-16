import cypari2, mpmath as mp, json, sys, time
pari=cypari2.Pari(); pari.allocatemem(4*10**9); mp.mp.dps=320
NT=26; PREC=300; KMAX=16
def test(ctor,c=None):
    pari.set_real_precision(PREC); pari(f"L={ctor};"); par=pari("lfunparams(L)"); cond=int(par[0]) if c is None else 1
    if c is None: c=float(par[1])/2
    ser=pari(f"lfunlambda(L,{c}+x+O(x^{2*NT+2}))")
    a=[mp.mpf(str(pari.polcoef(ser,2*n)).replace(" E","E")) for n in range(NT+1)]
    if a[0]<=0: return None
    psi=[a[n-1]/a[n] for n in range(1,NT+1)]
    cur=[psi[i+2]-2*psi[i+1]+psi[i] for i in range(len(psi)-2)]; first=None
    for k in range(KMAX):
        neg=[i+1 for i,v in enumerate(cur) if (-1)**k*v<0]
        if neg: first=(k,neg[0],float((-1)**k*cur[neg[0]-1])); break
        cur=[cur[i+1]-cur[i] for i in range(len(cur)-1)]
    pari.set_real_precision(38)
    z=[float(t) for t in pari("lfunzeros(L,40)")]; z=[t for t in z if t>1e-8]
    return {"conductor":cond,"gamma":str(par[2]),"S1":float(a[1]/a[0]),"psi1":float(psi[0]),"zeros":z[:6],"violation":first}
which=sys.argv[1]; out={}; t0=time.time()
if which=="dir":
    Ds=[d for d in range(-400,401) if d not in (0,1) and int(pari(f"isfundamental({d})"))]
    for d in Ds:
        out[str(d)]=test(f"lfuncreate({d})"); print(d,out[str(d)]["violation"],round(time.time()-t0),flush=True)
elif which=="mf":
    for k in (12,16,20):
        out[f"level1_wt{k}"]=test(f"lfunmf(mfinit([1,{k}]),mfeigenbasis(mfinit([1,{k}]))[1])",c=k/2); print(k,out[f"level1_wt{k}"]["violation"],flush=True)
    for k in (24,28,32):
        try:
            out[f"level1_wt{k}_emb1"]=test(f"lfunmf(mfinit([1,{k}]),mfeigenbasis(mfinit([1,{k}]))[1])[1]",c=k/2); print(k,out[f"level1_wt{k}_emb1"]["violation"],flush=True)
        except Exception as e: print(k,"skip",e)
elif which=="ec":
    seen={}; cands=[]
    for a in range(-40,41):
        for b in range(-40,41):
            if 4*a**3+27*b*b==0: continue
            E=pari(f"ellinit([{a},{b}])"); N=int(pari.ellglobalred(E)[0])
            if N>1500: continue
            if int(pari.ellrootno(E))!=1: continue
            fp=tuple(int(x) for x in pari.ellan(E,30))
            if fp in seen: continue
            seen[fp]=1; cands.append((N,a,b))
    cands.sort()
    for N,a,b in cands[:120]:
        r=test(f"lfuncreate(ellinit([{a},{b}]))")
        if r is None: continue
        out[f"N{N}_a{a}_b{b}"]=r; print(N,a,b,r["violation"],round(time.time()-t0),flush=True)
json.dump(out,open(f"patie_survey_{which}.json","w"),indent=1)
