# Random integers below x: Poisson Dirichlet, Dickman, Erdos Kac, Frobenius colouring
import numpy as np, json, math, time
from scipy import integrate
d=np.load('data.npz'); P=d['P']; c1=d['c1']   # S3 classes of x^3-x-1: 0 identity,1 transposition,2 three cycle
haar={0:1/6,1:1/2,2:1/3}
X=10**7
t0=time.time()
# smallest prime factor sieve to factorise all n<=X
spf=np.zeros(X+1,dtype=np.int32)
for p in P:
    if p*p>X: break
    m=spf[p::p]; sel=(m==0); m[sel]=p; spf[p::p]=m
idx=np.nonzero(spf==0)[0]; spf[idx]=idx  # primes and 1
print('sieve',time.time()-t0)
rng=np.random.default_rng(7); M=400000
N=rng.integers(2,X+1,size=M)
cls_of={int(p):int(c) for p,c in zip(P,c1)}
pidx={int(p):i for i,p in enumerate(P)}
omega=np.zeros(M,dtype=np.int32); Omega=np.zeros(M,dtype=np.int32)
frac_max=np.zeros(M); cls_max=np.zeros(M,dtype=np.int8)
frac_sb=np.zeros(M); cls_sb=np.zeros(M,dtype=np.int8)
counts=np.zeros((M,3),dtype=np.int16)     # distinct prime factors per class
second=np.zeros(M)
for i,n in enumerate(N):
    n=int(n); L=math.log(n); ps=[]; es=[]
    while n>1:
        p=int(spf[n]); e=0
        while n%p==0: n//=p; e+=1
        ps.append(p); es.append(e)
    omega[i]=len(ps); Omega[i]=sum(es)
    w=np.array([e*math.log(p) for p,e in zip(ps,es)])/L
    j=int(np.argmax([math.log(p) for p in ps])); frac_max[i]=math.log(ps[j])/L; cls_max[i]=cls_of[ps[j]] if ps[j]!=23 else 3
    k=rng.choice(len(ps),p=w); frac_sb[i]=w[k]; cls_sb[i]=cls_of[ps[k]] if ps[k]!=23 else 3
    for p in ps:
        if p!=23: counts[i,cls_of[p]]+=1
    if len(ps)>1: second[i]=sorted(w)[-2]
print('factorised',time.time()-t0)
# Dickman rho on [0,3]
us=np.linspace(0,3,301); rho=np.ones_like(us)
for k in range(1,len(us)):
    u=us[k]
    if u<=1: rho[k]=1
    else:
        f=lambda t: np.interp(t,us[:k],rho[:k]); rho[k]=integrate.quad(f,u-1,u-1e-12 if u-1<=0 else u-1 if False else u-1,limit=50)[0] if False else None
# simpler: rho(u)=1 on [0,1]; rho'(u)=-rho(u-1)/u, integrate with trapezoid
h=us[1]-us[0]; rho=np.ones_like(us)
for k in range(1,len(us)):
    u=us[k]
    if u<=1: continue
    rho[k]=rho[k-1]-h*0.5*(np.interp(u-1,us,rho)/u+np.interp(us[k-1]-1,us,rho)/us[k-1])
# P(log P+ / log N <= v) = rho(1/v)
vs=np.linspace(0.05,1,96); cdf_th=np.array([np.interp(1/v,us,rho) if 1/v<=3 else 0 for v in vs]); cdf_emp=np.array([np.mean(frac_max<=v) for v in vs])
out={'M':M,'X':X,'dickman_maxdev':float(np.max(np.abs(cdf_emp-cdf_th))),'vs':vs.tolist(),'cdf_emp':cdf_emp.tolist(),'cdf_th':cdf_th.tolist(),
 'rho_1_2':float(np.interp(2,us,rho)),'rho_3':float(rho[-1])}
ll=math.log(math.log(X)); out['omega_mean']=float(omega.mean()); out['omega_var']=float(omega.var()); out['loglog']=ll
z=(omega-ll)/math.sqrt(ll); out['omega_std_z']=[float(np.mean(z<=q)) for q in [-1,0,1]]
out['Omega_minus_omega_mean']=float((Omega-omega).mean())   # -> sum_p 1/(p(p-1)) = 0.7731
# largest prime factor Frobenius class
out['cls_max']=[float(np.mean(cls_max==c)) for c in range(3)]; out['cls_max_ram']=float(np.mean(cls_max==3))
out['cls_sb']=[float(np.mean(cls_sb==c)) for c in range(3)]; out['cls_sb_ram']=float(np.mean(cls_sb==3))
tot=counts.sum(0); out['share_count']=(tot/tot.sum()).tolist(); sel=omega>0; out['share_mean']=[float(np.mean(counts[sel,c]/omega[sel])) for c in range(3)]
# size biased fraction uniform? per class histograms
hb=np.linspace(0,1,11)
out['sb_hist']={str(c):np.histogram(frac_sb[cls_sb==c],bins=hb,density=True)[0].tolist() for c in range(3)}
out['sb_hist_all']=np.histogram(frac_sb,bins=hb,density=True)[0].tolist()
out['max_hist']=np.histogram(frac_max,bins=np.linspace(0,1,41),density=True)[0].tolist()
# conditional multinomial: given omega=k, class counts
chi=[]
for k in [2,3,4,5]:
    sel=omega==k; C=counts[sel]; n=sel.sum()
    exp=[n*k*haar[c] for c in range(3)]; obs=C.sum(0)
    chi.append({'k':k,'n':int(n),'obs':obs.tolist(),'exp':exp,'chi2':float(sum((o-e)**2/e for o,e in zip(obs,exp)))})
out['multinomial']=chi
# second largest fraction mean: PD(1): E V2 = ? ; size biased pick uniform mean 1/2; E V1 = 0.6243 (Golomb Dickman)
out['EV1']=float(frac_max.mean()); out['EV2']=float(second.mean()); out['E_sb']=float(frac_sb.mean())
print({k:v for k,v in out.items() if not isinstance(v,(list,dict))})
print(out['cls_max'],out['cls_sb'],out['omega_std_z']); print(chi)
json.dump(out,open('pd1.json','w'))
np.savez_compressed('pd_samples.npz',omega=omega,frac_max=frac_max,cls_max=cls_max,frac_sb=frac_sb,cls_sb=cls_sb,counts=counts)
