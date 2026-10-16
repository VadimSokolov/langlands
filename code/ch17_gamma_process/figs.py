import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D","#3B8B5A","#A23B72"]
o=json.load(open('laws.json')); lam=np.array(o['lams'])
fig,ax=plt.subplots(1,2,figsize=(6.6,2.7))
for c,(n,lab) in zip(C,[('identity','split completely, $\\mu=1/6$'),('transposition','transpositions, $\\mu=1/2$'),('threecycle','three cycles, $\\mu=1/3$')]):
    ax[0].plot(lam,-np.array(o['lim'][n]),color=c,lw=1.6,label=lab)
    for s,ls in [('1.5',':'),('1.1','--'),('1.01','-.')]:
        ax[0].plot(lam,-np.array(o['fin'][n][s]),ls,color=c,lw=0.9,alpha=0.8)
ax[0].set_xlabel(r"$\lambda$"); ax[0].set_ylabel(r"$-\log\,\mathbb{E}\,e^{-\lambda\,\Xi_\sigma(C)}$")
ax[0].legend(frameon=False,fontsize=6.5,loc="upper left"); ax[0].set_title("Chebotarev as a Gamma process ($S_3$, $N=23$)",fontsize=8)
for c,(n,lab) in zip(C[3:],[('rho_rho31',r'$\rho$ and $\rho_{31}$'),('qr_forgot',r'$(\frac{7}{p})$ and $(\frac{p}{7})$'),('ell_pair',r'$11a1$ and $37a1$')]):
    ax[1].plot(lam,-np.array(o['lim'][n]),color=c,lw=1.6,label=lab)
    ax[1].plot(lam,-np.array(o['fin'][n]['1.01']),'-.',color=c,lw=0.9)
ax[1].plot(lam,2*np.log1p(lam),'k:',lw=1,label=r"Gamma$(2)$: $2\log(1+\lambda)$")
ax[1].set_xlabel(r"$\lambda$"); ax[1].legend(frameon=False,fontsize=6.5,loc="upper left")
ax[1].set_title("three GGC laws with mean $2$",fontsize=8)
plt.tight_layout(); plt.savefig('figs/gamma_process.pdf'); plt.close()
z=o['zero']; s=np.array(z['s'])
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
for c,(n,lab) in zip(C,[('rho_rho31',r'$\rho,\rho_{31}$'),('qr_forgot',r'$(\frac{7}{p}),(\frac{p}{7})$'),('ell_pair',r'$11a1,37a1$')]):
    ax[0].plot(np.log(s-1),z[n]['logP'],'o',ms=2.5,color=c,label=f"{lab}: slope {z[n]['slope']:.3f}, mass {z[n]['mass'] if n!='ell_pair' else 1:.3f}")
ax[0].set_xlabel(r"$\log(\sigma-1)$"); ax[0].set_ylabel(r"$\log\,\mathbb{P}(\Xi_\sigma(Q)=0)$"); ax[0].legend(frameon=False,fontsize=6.2)
ax[0].set_title("zero probability $\\asymp(\\sigma-1)^{\\mathrm{Thorin\\ mass}}$",fontsize=8)
h=json.load(open('heat.json'))
for c,k in zip(C,['5','7','12']):
    e=np.array(h[k]['eps']); ax[1].loglog(1/e,np.maximum(h[k]['tv'],1e-17),'o',ms=2.5,color=c,label=f"$c={k}$")
    ax[1].loglog(1/e,h[k]['bound'],'-',color=c,lw=0.9)
ax[1].set_ylim(1e-17,2); ax[1].set_xlabel(r"$1/\varepsilon$"); ax[1].set_ylabel("total variation to uniform")
ax[1].legend(frameon=False,fontsize=6.5); ax[1].set_title("discrete Gaussian mod $c$",fontsize=8)
plt.tight_layout(); plt.savefig('figs/zero_heat.pdf'); plt.close()
