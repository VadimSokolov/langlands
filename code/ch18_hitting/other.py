import numpy as np, json, mpmath as mp, cypari2
from scipy.special import exp1
out={}
# (a) Chebotarev hitting: largest prime factor of N_sigma and its Frobenius, S3 field x^3-x-1
d=np.load('../work/data.npz'); P=d['P']; c1=d['c1']; X=1e7
keep=P!=23; P=P[keep].astype(float); c1=c1[keep]
mu=np.array([1/6,1/2,1/3]); res={}
for s in [1.5,1.2,1.1,1.05,1.02,1.01,1.005]:
    q=P**(-s); logsurv=np.log1p(-q)                     # P(e_p=0)
    xX=(s-1)*np.log(X); tailsurv=np.exp(-exp1(xX))      # P(no prime factor > X), PNT tail
    # suffix products: prob no prime factor among primes > p (up to X)
    suf=np.concatenate([np.cumsum(logsurv[::-1])[::-1][1:],[0.0]])
    w=q*np.exp(suf)*tailsurv                            # P(e_p>=1 and p is the largest prime factor), approx e_p>=1 prob ~ q
    top=[(1-tailsurv)*mu[k]+np.sum(w[c1==k]) for k in range(3)]
    res[str(s)]=top
    print(s,[round(t/sum(top),4) for t in top], 'P(N>1)~',round(sum(top),4))
out['top']=res
# E1 law: P(no prime factor in C above level y) for C = identity at s=1.01
s=1.01; ys=np.linspace(0.05,3,30); k=0
lp=[]
for y in ys:
    m=(c1==k)&((s-1)*np.log(P)>y)
    tail=-mu[k]*exp1(max(y,(s-1)*np.log(X)))
    lp.append(float(np.sum(np.log1p(-P[m]**(-s)))+tail))
out['e1']={'y':ys.tolist(),'logP':lp,'limit':(-mu[k]*exp1(ys)).tolist()}
# (b) mean explosion times: sum 1/gamma^2 via Lambda'' at centre vs zeros
pari=cypari2.Pari(); pari.allocatemem(2*10**9); pari.set_real_precision(30)
def S2(Lstr,c,T):
    pari(f"L={Lstr}; Z=lfunzeros(L,{T});")
    lam0=float(pari(f"lfunlambda(L,{c})")); lam2=float(pari(f"lfunlambda(L,{c},2)")); lam1=float(pari(f"lfunlambda(L,{c},1)"))
    an=(lam2/lam0-(lam1/lam0)**2)/2
    zs=[float(z) for z in pari("Z")]; zs=[z for z in zs if z>1e-8]
    cond=float(pari("L[5]")); deg=len(pari("L[3]"))
    tail=(np.log(cond)+deg*np.log(T/(2*np.pi))+deg)/(2*np.pi*T)
    return an, sum(1/z**2 for z in zs), tail, len(zs)
for name,Ls,c in [('zeta','lfuncreate(1)',0.5),('Delta','lfunmf(mfinit([1,12],1),mfDelta())',6),('rho23','lfundiv(lfuncreate(x^3-x-1),1)',0.5),('chi_m4','lfuncreate(-4)',0.5)]:
    an,sz,tail,nz=S2(Ls,c,400 if name!='Delta' else 300)
    an=an-4 if name=='zeta' else an
    out[name]={'analytic':an,'zeros':sz,'tail':tail,'nzeros':nz}; print(name,an,sz+tail,nz)
# (c) Brownian exit time functional equation
mp.mp.dps=30; c=mp.pi**2/4
def ET(s): return (2/mp.pi)*2**(s+1)*mp.gamma(s+1)*mp.nsum(lambda k:(-1)**k*(2*k+1)**(-2*s-1),[0,mp.inf])
def ETb(s): return (2/mp.pi)*2**(s+1)*mp.gamma(s+1)*mp.dirichlet(2*s+1,[0,1,0,-1])
chk=[]
for s in [0.3,0.7,1.3,mp.mpf(1)/4+2j]:
    lhs=ETb(s); rhs=mp.sqrt(2/mp.pi)*(mp.pi/2)**(2*s+1)*ETb(-0.5-s); chk.append(float(abs(lhs-rhs)/abs(lhs)))
# Monte Carlo: exit time of BM from (-1,1), mean should be 1 (E tau = 1) and E T = pi^2/4
out['exit_check']=chk; print('exit fe',chk, 'E[T]=',ETb(1), 'expected', c*1)
json.dump(out,open('other.json','w'),default=float)
