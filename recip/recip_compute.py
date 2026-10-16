import cypari2, mpmath as mp, numpy as np, math, json
pari=cypari2.Pari(); pari.allocatemem(2*10**9); pari.set_real_precision(38); mp.mp.dps=30
out={}
NMAX=60000
# ---------- S3 Artin rep of x^3-x-1: Frobenius data from factorisation mod p ----------
P=[int(p) for p in pari(f"primes([2,{NMAX}])")]
def frob_S3(p):
    if p==23: return ("ram",1)
    f=pari(f"factormod(x^3-x-1,{p})"); degs=sorted(int(pari.poldegree(g)) for g in f[0]) if False else None
    nroots=len(pari(f"polrootsmod(x^3-x-1,{p})"))
    if nroots==3: return (2,1)
    if nroots==1: return (0,-1)
    return (-1,1)
def local_data(ap_eps):
    return ap_eps
def coeffs_from_frob(fr, N):
    a=np.zeros(N+1); a[1]=1.0
    # prime power coefficients
    pp={}
    for p,(ap,eps) in fr.items():
        if ap=="ram":
            seq=[1.0]; v=1.0
            while True:
                v*=p
                if v>N: break
                seq.append(float(eps)**len(seq))
        else:
            seq=[1.0,float(ap)]; q=p*p
            while q<=N: seq.append(ap*seq[-1]-eps*seq[-2]); q*=p
        pp[p]=seq
    # multiplicative fill via smallest prime factor
    spf=list(range(N+1))
    for p in P:
        if p*p>N: break
        for m in range(p*p,N+1,p):
            if spf[m]==m: spf[m]=p
    for n in range(2,N+1):
        p=spf[n]; m=n; k=0
        while m%p==0: m//=p; k+=1
        a[n]=a[m]*pp[p][k]
    return a,pp
fr={p:frob_S3(p) for p in P}
a_true,pp_true=coeffs_from_frob(fr,NMAX)
# check against PARI's Artin L function coefficients
chk=[int(v) for v in pari("lfunan(lfundiv(lfuncreate(x^3-x-1),1),40)")]
out["check_coeffs_match"]=bool(all(abs(chk[i]-a_true[i+1])<1e-9 for i in range(40)))
fr_imp=dict(fr); fr_imp[2]=(2,1)   # pretend Frob_2 is trivial (split) instead of order 3
a_imp,pp_imp=coeffs_from_frob(fr_imp,NMAX)
# ---------- Polya kernel Phi(u) = f(i e^u / sqrt N) e^{k u/2} ----------
def kernel(a,N,k,u):
    y=math.exp(u)/math.sqrt(N); n=np.arange(1,len(a)); w=np.exp(-2*math.pi*n*y)
    return float((a[1:]*w).sum())*math.exp(k*u/2)
us=np.linspace(-3,3,241)
K_true=[kernel(a_true[:4000],23,1,u) for u in us]; K_imp=[kernel(a_imp[:4000],23,1,u) for u in us]
out["S3_kernel_asym_true"]=float(max(abs(K_true[i]-K_true[-1-i]) for i in range(len(us))))
out["S3_kernel_asym_imp"]=float(max(abs(K_imp[i]-K_imp[-1-i]) for i in range(len(us))))
# Delta kernel with tau(2) sign flipped (keeping Ramanujan bound)
pari("T=mfcoefs(mfDelta(),3000);")
tau=np.array([float(pari(f"T[{n}+1]")) for n in range(0,3001)])
# impostor Delta: Frob at 2 with trace -tau(2): multiplicatively rebuild
def delta_imp():
    fr2={}
    for p in [int(q) for q in pari("primes([2,3000])")]:
        ap=tau[p]/p**5.5
        fr2[p]=(ap,1.0)
    fr2[2]=(-fr2[2][0],1.0)
    a,_=coeffs_from_frob_generic(fr2,3000)
    return a
def coeffs_from_frob_generic(fr,N):
    a=np.zeros(N+1); a[1]=1.0; spf=list(range(N+1))
    for p in fr:
        if p*p>N: break
        for m in range(p*p,N+1,p):
            if spf[m]==m: spf[m]=p
    pp={}
    for p,(ap,eps) in fr.items():
        seq=[1.0,ap]; q=p*p
        while q<=N: seq.append(ap*seq[-1]-eps*seq[-2]); q*=p
        pp[p]=seq
    for n in range(2,N+1):
        p=spf[n]; m=n; k=0
        while m%p==0: m//=p; k+=1
        a[n]=a[m]*pp[p][k]
    return a,pp
taun=np.array([tau[n]/n**5.5 if n>0 else 0 for n in range(3001)])
aD_imp=delta_imp()
KD_true=[kernel(taun,1,12-11,u) for u in us]   # unitary normalisation: weight 1 analogue: f(iy) sum tau(n)/n^{11/2} e^{-2pi n y}? see note
# Exact Polya kernel of Delta: Phi(u)=Delta(i e^u) e^{6u}
def kerD(a_norm,u):
    y=math.exp(u); n=np.arange(1,len(a_norm)); return float((a_norm[1:]*n**5.5*np.exp(-2*math.pi*n*y)).sum())*math.exp(6*u)
KD_true=[kerD(taun,u) for u in us]; KD_imp=[kerD(aD_imp,u) for u in us]
out["Delta_kernel_asym_true"]=float(max(abs(KD_true[i]-KD_true[-1-i]) for i in range(len(us))))
out["Delta_kernel_asym_imp"]=float(max(abs(KD_imp[i]-KD_imp[-1-i]) for i in range(len(us))))
np.save("kernels.npy",np.vstack([us,K_true,K_imp,KD_true,KD_imp]))
print({k:v for k,v in out.items()},flush=True)
# ---------- heat trace from primes vs zeros, S3 ----------
L=pari("L=lfundiv(lfuncreate(x^3-x-1),1)")
zs=[float(z) for z in pari("lfunzeros(L,120)") if float(z)>1e-8]
out["S3_first_zeros"]=zs[:5]; out["S3_nzeros_120"]=len(zs)
def cm(pp,p,m):
    seq=pp[p]  # coefficients a(p^k); need alpha^m+beta^m: from Euler poly
    return None
def vonmangoldt_coeffs(fr):
    # c_m(p) = alpha^m + beta^m
    C={}
    for p,(ap,eps) in fr.items():
        if ap=="ram": C[p]=[None]+[float(eps)**m for m in range(1,300)]
        else:
            c=[2.0,float(ap)]
            for m in range(2,300): c.append(ap*c[-1]-eps*c[-2])
            C[p]=c
    return C
C_true=vonmangoldt_coeffs(fr); C_imp=vonmangoldt_coeffs(fr_imp)
def W_prime(x,C,N=23,mu=0.0):
    arch=mp.quad(lambda t: mp.e**(-x*t*t)*(mp.log(N)+2*mp.re(-mp.log(2*mp.pi)+mp.digamma(0.5+mu+1j*t))),[-60,0,60])/(2*mp.pi)
    umax=math.sqrt(4*x*70)+1; pr=0.0
    for p in P:
        Lp=math.log(p)
        if Lp>umax: break
        m=1
        while m*Lp<=umax:
            pr+=C[p][m]*Lp*p**(-m/2)*math.sqrt(math.pi/x)*math.exp(-(m*Lp)**2/(4*x)); m+=1
    return (float(arch)-pr/math.pi)/2
def W_zero(x): return sum(math.exp(-x*g*g) for g in zs)
xs=[0.02,0.05,0.1,0.2,0.3,0.5,0.7,1.0]
heat=[(x,W_zero(x),W_prime(x,C_true),W_prime(x,C_imp)) for x in xs]
out["S3_heat"]=heat
for r in heat: print(r,flush=True)
# ---------- Gamma shape: (sigma-1) X^(sigma), D = zeta * L(chi_-23) * L(rho) ----------
pari("Z=lfuncreate(1); C=lfuncreate(-23);")
def D(s,mult=1):
    v=pari(f"lfun(Z,{s})*lfun(C,{s})*lfun(L,{s})"); return float(v)**mult
gam={}
for mult in (1,4):
    for sig in (1.1,1.01,1.001):
        lam=[0.5,1,2,4]
        gam[f"m{mult}_s{sig}"]=[(l,D(sig+l*(sig-1),mult)/D(sig,mult),(1+l)**(-mult)) for l in lam]
out["gamma_limit"]=gam
for k,v in gam.items(): print(k,[(l,round(a,5),round(b,5)) for l,a,b in v],flush=True)
json.dump(out,open("recip.json","w"),indent=1)
