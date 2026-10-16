import cypari2, numpy as np, mpmath as mp, json, math
pari=cypari2.Pari(); pari.allocatemem(4*10**9); pari.set_real_precision(30)
mp.mp.dps=20
out={}
# A. Polya kernels for 11a1 (eps=+1) and 37a1 (eps=-1), weight 2: Phi(u)=f(i e^u/sqrt N) e^u
for name,ai,N in [('11a1',[0,-1,1,-10,-20],11),('37a1',[0,0,1,-1,0],37)]:
    pari(f"E=ellinit({ai}); an=ellan(E,4000);"); a=np.array([float(x) for x in pari("an")])
    n=np.arange(1,len(a)+1)
    us=np.linspace(-2.5,2.5,201)
    K=np.array([np.sum(a*np.exp(-2*np.pi*n*np.exp(u)/np.sqrt(N)))*np.exp(u) for u in us])
    eps=int(pari("ellrootno(E)"))
    asym=float(np.max(np.abs(K-eps*K[::-1])))
    out['kernel_'+name]={'u':us.tolist(),'K':K.tolist(),'eps':eps,'asym':asym,'min':float(K.min())}
    print(name,'eps',eps,'asym',asym,'min',K.min(),'max',K.max())
# B. heat trace of 11a1 from Frobenius (analytic normalisation, centre 1/2), explosion mean
X=10**7
pari(f"E=ellinit([0,-1,1,-10,-20]);")
P=np.array([int(p) for p in pari(f"primes([2,{X}])")],dtype=np.int64)
d=np.load('data.npz'); x11=d['x11']   # a_p/sqrt p for p<=1e7 (bad p=11 has a_11 entry)
assert len(x11)==len(P)
logs=[];coef=[]
th=np.arccos(np.clip(x11/2,-1,1))
for m in range(1,24):
    mask=P.astype(float)**m<=X
    if not mask.any(): break
    pp=P[mask].astype(float)
    v=2*np.cos(m*th[mask])
    v=np.where(P[mask]==11,(1/np.sqrt(11))**m*1.0,v)   # a_11=+1 split multiplicative: alpha=1/sqrt(11)
    logs.append(m*np.log(pp)); coef.append(v*np.log(pp)/pp**(m/2))
L=np.concatenate(logs); Cf=np.concatenate(coef)
g=lambda t: mp.log(11)+2*mp.re(-mp.log(2*mp.pi)+mp.digamma(mp.mpf(1)+1j*t))   # Gamma_C(s+1/2) at s=1/2+it
def arch(x): return float(mp.quad(lambda t: mp.e**(-x*t*t)*g(t),[0,1,5,20,mp.inf])/mp.pi)
def W(x):
    prime=(1/np.pi)*np.sum(Cf*np.sqrt(np.pi/x)*np.exp(-L**2/(4*x)))
    return 0.5*(arch(x)-prime)
pari("Li=lfuninit(lfuncreate(E),[1,300]); Z=lfunzeros(Li,250);")   # arithmetic normalisation centre 1
zs=np.array([float(z) for z in pari("Z")]); zs=zs[zs>1e-9]
chk=[(x,W(x),float(np.sum(np.exp(-x*zs**2)))) for x in [0.002,0.01,0.05,0.2]]
print(chk); out['heat_check']=chk
had=float(pari("real(lfunlambda(Li,1,2)/lfunlambda(Li,1))"))/2
ET=float(mp.quad(lambda y: W(float(mp.e**(-y)))*mp.e**(-y),[-mp.log(2.0),2,6,12,30,60]))
out['ET_frobenius']=ET; out['ET_hadamard']=had; out['gamma1']=float(zs[0])
print('E T frobenius',ET,'hadamard',had,'gamma1',zs[0])
# derivative signs (complete monotonicity spot check) by finite differences of W from Frobenius
xs=[0.05,0.1,0.2,0.4]
out['cm_check']=[]
for x in xs:
    h=1e-3; w0,w1,w2,w3=W(x),W(x+h),W(x+2*h),W(x+3*h)
    out['cm_check'].append([x,w0,-(w1-w0)/h,(w2-2*w1+w0)/h**2])
print(out['cm_check'])
json.dump(out,open('thorin_ec.json','w'))
