import numpy as np, json
from scipy.special import exp1
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D","#3B8B5A"]
o=json.load(open('thorin_ec.json'))
d=np.load('data.npz'); P0=d['P']; X=1e7
res={}
# C. Xi_sigma(x^2): weight one (eta eta23 trace) vs weight two (11a1)
chi=np.array([2.,0.,-1.])
k1=P0!=23; Pa=P0[k1].astype(float); x1=chi[d['c1'][k1]]
k2=P0!=11; Pb=P0[k2].astype(float); x2=d['x11'][k2]
th=np.linspace(0,np.pi,4001)[1:-1]; w=np.sin(th)**2; w/=w.sum(); sc=(2*np.cos(th))**2
lam=np.linspace(0,6,31)
lim1=-np.array([ (1/6)*np.log1p(4*l)+(1/3)*np.log1p(l) for l in lam])
lim2=-np.array([np.sum(w*np.log1p(l*sc)) for l in lam])
def fin(P,Q,vals,probs,s):
    q=P**(-s); r=[]
    for l in lam:
        t=P**(-(s-1)*l*Q); xX=(s-1)*np.log(X)
        tail=-np.sum(probs*(exp1(xX)-exp1((1+l*vals)*xX)))
        r.append(np.sum(np.log1p(-q)-np.log1p(-q*t))+tail)
    return np.array(r)
f1=fin(Pa,x1**2,np.array([4.,1.]),np.array([1/6,1/3]),1.01)
f2=fin(Pb,x2**2,sc,w,1.01)
res['lim1']=lim1.tolist(); res['lim2']=lim2.tolist(); res['f1']=f1.tolist(); res['f2']=f2.tolist()
print('max dev w1',np.max(np.abs(f1-lim1)),'w2',np.max(np.abs(f2-lim2)))
# E. Rankin Selberg Thorin density, prime part, 11a1, alpha=1.5 and 1.2
th11=np.arccos(np.clip(d['x11']/2,-1,1)); P=P0.astype(float)
zs=np.linspace(0.05,60,600); U={}
for al in [1.5,1.2]:
    acc=np.zeros_like(zs)
    for m in range(1,30):
        mask=(P**(-m*al)>1e-14)
        if not mask.any(): break
        b=np.where(P0[mask]==11,(1/11.0)**m,(2*np.cos(m*th11[mask]))**2)
        wgt=b*np.log(P[mask])*P[mask]**(-m*al); xx=m*np.log(P[mask])
        acc+=np.array([np.sum(wgt*2*np.sin(xx*np.sqrt(z)/2)**2) for z in zs])
    U[al]=(acc/(2*np.pi*np.sqrt(zs))).tolist(); print(al,min(U[al]))
res['U']={str(k):v for k,v in U.items()}; res['zs']=zs.tolist()
json.dump(res,open('thorin_fig.json','w'))
# Figures
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
for k,c,lab in [('kernel_11a1',C[0],r'11a1, $\varepsilon=+1$: a density'),('kernel_37a1',C[1],r'37a1, $\varepsilon=-1$: odd')]:
    u=np.array(o[k]['u']); K=np.array(o[k]['K'])
    ax[0].plot(u,K,color=c,lw=1.5,label=lab); ax[0].plot(-u,o[k]['eps']*K,'--',color=c,lw=0.9)
ax[0].axhline(0,color='k',lw=0.4); ax[0].set_xlabel('$u$'); ax[0].set_ylabel(r'$\Phi_E(u)$'); ax[0].legend(frameon=False,fontsize=6.5)
ax[0].set_title("modularity as symmetry in law",fontsize=8)
ax[1].plot(lam,-lim1,color=C[0],lw=1.5,label='weight one: atomic Thorin'); ax[1].plot(lam,-f1,'o',ms=2.5,color=C[0])
ax[1].plot(lam,-lim2,color=C[1],lw=1.5,label='weight two: continuous Thorin'); ax[1].plot(lam,-f2,'o',ms=2.5,color=C[1])
ax[1].set_xlabel(r'$\lambda$'); ax[1].set_ylabel(r'$-\log\,\mathbb{E}e^{-\lambda\,\Xi(x^2)}$'); ax[1].legend(frameon=False,fontsize=6.5)
ax[1].set_title(r"Frobenius functional $x_p^2$, limit and $\sigma=1.01$",fontsize=8)
plt.tight_layout(); plt.savefig('figs/flt_kernels.pdf'); plt.close()
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
xs=np.array([c[0] for c in o['heat_check']]); 
xx=np.logspace(-2.7,-0.6,40)
ax[0].semilogy([c[0] for c in o['heat_check']],[c[2] for c in o['heat_check']],'o',color=C[1],ms=5,label='sum over zeros')
ax[0].semilogy([c[0] for c in o['heat_check']],[c[1] for c in o['heat_check']],'x',color=C[0],ms=7,label='from point counts')
ax[0].set_xlabel('$x$'); ax[0].set_ylabel(r'$\mathcal{W}_E(x)$'); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title("heat trace of 11a1",fontsize=8)
for c,al in zip(C,[1.5,1.2]):
    ax[1].plot(zs,U[al],color=c,lw=1.2,label=rf"$\alpha={al}$")
ax[1].set_xlabel('$z$'); ax[1].set_ylabel('prime part of Thorin density'); ax[1].legend(frameon=False,fontsize=7)
ax[1].set_title("Rankin Selberg GGC of 11a1 (unconditional)",fontsize=8)
plt.tight_layout(); plt.savefig('figs/flt_thorin.pdf'); plt.close()
