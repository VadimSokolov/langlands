import cypari2, mpmath as mp, json
pari=cypari2.Pari(); pari.allocatemem(6*10**9); mp.mp.dps=500
def run(ctor, zeta, prec, NT, K):
    pari.set_real_precision(prec); pari(f"L={ctor};"); c=float(pari("lfunparams(L)")[1])/2
    ser=pari(f"lfunlambda(L, {c} + x + O(x^{2*NT+2}))"+(" * (x^2 - 1/4)" if zeta else ""))
    a=[mp.mpf(str(pari.polcoef(ser,2*n)).replace(" E","E")) for n in range(NT+1)]
    psi=[a[n-1]/a[n] for n in range(1,NT+1)]
    d2=[psi[i+2]-2*psi[i+1]+psi[i] for i in range(len(psi)-2)]
    cur=d2; first=None; mins=[]
    for k in range(K+1):
        vals=[(-1)**k*v for v in cur]
        mins.append(float(min(vals)))
        neg=[i+1 for i,v in enumerate(vals) if v<0]
        if neg and first is None: first=(k,neg[0],float(vals[neg[0]-1]))
        cur=[cur[i+1]-cur[i] for i in range(len(cur)-1)]
        if len(cur)<1: break
    return first, mins
out={}
for name,ctor,z in [("xi","lfuncreate(1)",True),("Delta","lfunmf(mfinit([1,12]),mfDelta())",False),("L(chi_-4)","lfuncreate(-4)",False),("11a1","lfuncreate(ellinit([0,-1,1,-10,-20]))",False)]:
    r1=run(ctor,z,350,40,30); print(name,"prec350",r1[0],flush=True)
    r2=run(ctor,z,500,40,30); print(name,"prec500",r2[0],[f"{m:.2e}" for m in r2[1][:31]],flush=True)
    out[name]={"first_violation_350":r1[0],"first_violation_500":r2[0],"mins":r2[1]}
json.dump(out,open("stage7b.json","w"),indent=1)
