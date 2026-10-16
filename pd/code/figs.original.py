import json, numpy as np, math, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=['#1F3B63','#9C3D1F','#4E8F5A','#B08A2E']
A=json.load(open('pd1.json')); B=json.load(open('pd2.json')); S=np.load('pd_samples.npz')
fig,ax=plt.subplots(1,3,figsize=(6.8,2.5))
# (a) largest prime factor
vs=np.array(A['vs']); ax[0].plot(vs,A['cdf_emp'],color=C[0],lw=1.4,label=r'$n\leq10^7$ uniform')
ax[0].plot(vs,A['cdf_th'],'--',color=C[1],lw=1.1,label=r'Dickman $\rho(1/v)$')
ax[0].set_xlabel(r'$v$'); ax[0].set_ylabel(r'$P(\log P^+/\log n\leq v)$'); ax[0].legend(frameon=False,fontsize=6.5); ax[0].set_title('largest prime factor',fontsize=8)
# (b) omega
om=S['omega']; ks=np.arange(0,9); emp=[np.mean(om==k) for k in ks]
mu=A['loglog']+B['B_total']
ax[1].bar(ks,emp,color='0.8',width=0.8,label=r'$\omega(n)$, $n\leq10^7$')
ax[1].plot(ks,poisson.pmf(ks,mu),'o',ms=3.5,color=C[0],label=rf'Poisson, mean $\log\log x+B={mu:.2f}$')
xx=np.linspace(-0.5,8.5,200); ax[1].plot(xx,norm.pdf(xx,A['loglog'],math.sqrt(A['loglog'])),color=C[1],lw=1,label=r'normal, Erd\H{o}s Kac'.replace(r'\H{o}','ő'))
ax[1].set_xlabel('$k$'); ax[1].legend(frameon=False,fontsize=5.6,loc='upper right'); ax[1].set_title('number of prime factors',fontsize=8)
# (c) class shares
labs=['identity','transposition','3 cycle']; x=np.arange(3); w=0.2
ax[2].bar(x-1.5*w,[1/6,1/2,1/3],w,color='0.75',label='Haar')
ax[2].bar(x-0.5*w,A['cls_max'],w,color=C[0],label='largest factor')
ax[2].bar(x+0.5*w,A['cls_sb'],w,color=C[2],label='size biased factor')
cnt=np.array(A['share_count'])
ax[2].bar(x+1.5*w,cnt,w,color=C[1],label='distinct factors')
ax[2].plot(x+1.5*w,B['pred_count'],'k_',ms=9,mew=1.2); ax[2].plot(x+0.5*w,B['pred_sb'],'k_',ms=9,mew=1.2)
ax[2].set_xticks(x); ax[2].set_xticklabels(['id.','transp.','3 cycle'],fontsize=7); ax[2].set_ylabel('share'); ax[2].legend(frameon=False,fontsize=5.6)
ax[2].set_title(r'Frobenius classes of $x^3-x-1$',fontsize=8)
plt.tight_layout(); plt.savefig('figs/pd_integers.pdf'); plt.close()
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
lam=np.array(B['lam'])
ax[0].plot(lam,B['th'],color=C[0],lw=1.4,label=r'GGC limit $\exp(-\int\log(1+\lambda Q)\,d\mu)$')
ax[0].plot(lam[::2],np.array(B['mc'])[::2],'o',ms=3.5,mfc='none',color=C[1],label=r'$G\cdot\int Q\,dP$, $P$ Dirichlet process')
ax[0].plot(lam,B['exact101'],':',color=C[2],lw=1.4,label=r'zeta integer, $\sigma=1.01$')
ax[0].set_xlabel(r'$\lambda$'); ax[0].set_ylabel(r'$\mathbb{E}e^{-\lambda H}$'); ax[0].legend(frameon=False,fontsize=6); ax[0].set_title(r'Frobenius functional $Q=\mathrm{tr}^2$ as a Dirichlet mean',fontsize=8)
hb=np.linspace(0,1,11); mids=(hb[1:]+hb[:-1])/2
for c,col in zip(range(3),[C[0],C[2],C[1]]): ax[1].plot(mids,B.get('x',None) or A['sb_hist'][str(c)],'-o',ms=3,color=col,lw=1,label=labs[c])
ax[1].axhline(1,color='k',lw=0.6,ls='--',label=r'PD(1) limit: uniform'); ax[1].axvline(math.log(59)/math.log(1e7),color='0.6',lw=0.7); ax[1].set_ylim(0,1.6); ax[1].set_xlabel(r'size biased fraction $e_p\log p/\log n$'); ax[1].set_ylabel('density'); ax[1].legend(frameon=False,fontsize=6.5)
ax[1].set_title(r'size biased fraction by class, $x=10^7$',fontsize=8)
plt.tight_layout(); plt.savefig('figs/pd_dirichlet.pdf'); plt.close()
