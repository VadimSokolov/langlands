import cypari2, mpmath as mp, numpy as np, json, math
pari=cypari2.Pari(); pari.allocatemem(3*10**9); mp.mp.dps=200
out={}
def patie(ctor, zeta=False, prec=150, NT=18):
    pari.set_real_precision(prec); pari(f"L={ctor};"); par=pari("lfunparams(L)"); c=float(par[1])/2
    s = f"lfunlambda(L, {c} + x + O(x^{2*NT+2}))" + (" * (x^2 - 1/4)" if zeta else "")
    ser=pari(s); a=[mp.mpf(str(pari.polcoef(ser,2*n)).replace(" E","E")) for n in range(NT+1)]
    psi=[a[n-1]/a[n] for n in range(1,NT+1)]
    d2=[psi[i+2]-2*psi[i+1]+psi[i] for i in range(len(psi)-2)]
    cur=d2; first=None
    for k in range(9):
        neg=[i+1 for i,v in enumerate(cur) if (-1)**k*v<0]
        if neg and first is None: first=(k,neg[0],float((-1)**k*cur[neg[0]-1]))
        cur=[cur[i+1]-cur[i] for i in range(len(cur)-1)]
    return {"gamma_shifts": str(par[2]), "degree": len(par[2]), "conductor": int(par[0]), "psi": [float(p) for p in psi[:5]], "violation": first}
tests={
 "zeta (xi)":("lfuncreate(1)",True),"L(chi_-3)":("lfuncreate(-3)",False),"L(chi_-4)":("lfuncreate(-4)",False),"L(chi_5)":("lfuncreate(5)",False),
 "L(chi_-4)L(chi_5)":("lfunmul(lfuncreate(-4),lfuncreate(5))",False),"L(chi_5)L(chi_13)":("lfunmul(lfuncreate(5),lfuncreate(13))",False),
 "Delta":("lfunmf(mfinit([1,12]),mfDelta())",False),"11a1":("lfuncreate(ellinit([0,-1,1,-10,-20]))",False),
 "Artin x^3-x-1":("lfundiv(lfuncreate(x^3-x-1),lfuncreate(1))",False),"zeta_K, K=Q(x^3-x-1)":("lfuncreate(x^3-x-1)",True),
 "Sym2 11a1":("lfunsympow(ellinit([0,-1,1,-10,-20]),2)",False),"Sym2 37a1":("lfunsympow(ellinit([0,0,1,-1,0]),2)",False),
 "Sym3 11a1":("lfunsympow(ellinit([0,-1,1,-10,-20]),3)",False),"Sym4 11a1":("lfunsympow(ellinit([0,-1,1,-10,-20]),4)",False),
 "11a1 x 37a1":("lfunmul(lfuncreate(ellinit([0,-1,1,-10,-20])),lfuncreate(ellinit([0,0,1,-1,0])))",False),
}
res={}
for k,(c,z) in tests.items():
    try:
        res[k]=patie(c,z); print(k,res[k],flush=True)
    except Exception as e: print(k,"ERR",e,flush=True)
out["patie_survey"]=res
# zeta subordinator and first passage (ballot / Kendall) check
sig=2.0; c=0.5; x0=1.0; R=200000
rng=np.random.default_rng(11)
# Levy measure: atoms at log n with weight Lambda(n)/(log n n^sig), n prime powers
pp=[]; w=[]
for p in [int(q) for q in pari("primes([2,20000])")]:
    k=1
    while p**k<=10**8:
        pp.append(k*math.log(p)); w.append(1/(k*p**(k*sig))); k+=1
pp=np.array(pp); w=np.array(w); lam=w.sum(); prob=w/lam
# simulate Z_t = t/c - X_t until it exceeds x0; first passage time tau = c(x0 + total jump size)
M=np.zeros(R); 
for r in range(R):
    t_jump_sum=0.0; level=0.0
    # time to reach x0 from current deficit: need drift to cover x0 + jumps; jumps occur at Poisson rate lam
    target=x0
    t=0.0
    while True:
        # time needed to reach target by drift from current position y: (target - y)*c ; current y = t/c - jumps
        y=t/c - t_jump_sum
        need=(target - y)*c
        dt=rng.exponential(1/lam)
        if dt>=need: break
        t+=dt; t_jump_sum+=pp[rng.choice(len(pp),p=prob)] if False else 0
        t_jump_sum+=pp[np.searchsorted(np.cumsum(prob), rng.uniform())]
    M[r]=t_jump_sum
mvals=np.exp(M); mint=np.rint(mvals).astype(np.int64)
# formula: P(tau = c(x0 + log m)) = x0/(x0+log m) * P(X_t = log m), t = c(x0+log m), P(X_t=log m)=d_t(m) m^{-sig}/zeta(sig)^t
zs=float(pari(f"zeta({sig})"))
def d_t(m,t):
    f=pari(f"factor({m})"); v=1.0
    for i in range(len(f[0])):
        kk=int(f[1][i]); v*=float(mp.binomial(t+kk-1,kk))
    return v
rows=[]
for m in [1,2,3,4,5,6,8,12]:
    t=c*(x0+math.log(m)); pf=x0/(x0+math.log(m))*d_t(m,t)*m**(-sig)/zs**t
    rows.append((m, float(np.mean(mint==m)), pf))
out["zeta_first_passage"]={"sigma":sig,"c":c,"x":x0,"R":R,"rows":rows,"sum_formula_m_le_10^6":None}
print(rows,flush=True)
# symmetric z distribution mixing density check (delta=0.7)
d=mp.mpf("0.7")
lamk=lambda k:(d+k)**2/2
def p_u(u): return -mp.nsum(lambda k: mp.binomial(-2*d,k)*(d+k)/mp.beta(d,d)*mp.e**(-lamk(k)*u),[1,mp.inf]) if False else None
# direct partial fractions: 1/prod(1+s/lam_k) residues
def lt_formula(s): return 1/mp.nprod(lambda k: 1+s/lamk(k),[0,mp.inf])
def lt_gamma(s): return mp.gamma(d+mp.sqrt(2*s))*mp.gamma(d-mp.sqrt(2*s)+0j)/mp.gamma(d)**2 if False else (abs(mp.gamma(d+1j*mp.sqrt(2*s)))**2/mp.gamma(d)**2)
out["symz"]=[(float(s), float(lt_formula(s)), float(lt_gamma(s))) for s in (0.1,1,5)]
print(out["symz"])
json.dump(out,open("stage7.json","w"),indent=1)
