import json, numpy as np, math, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=['#1F3B63','#9C3D1F','#4E8F5A','#B08A2E','#7FB2F0']
A=json.load(open('switch23_a.json')); B=json.load(open('switch23_b.json')); Z=json.load(open('zeta_switch.json'))
# Figure 1: four faces of one L function, rho_23 = eta(z)eta(23z)
fig,ax=plt.subplots(2,2,figsize=(6.6,5.0))
# (a) Galois face: Levy measure atoms Lambda(n) n^-2 at log n
tab=B['bohr']; ns=[t[0] for t in tab]; lam=[t[2] for t in tab]; rec=[t[1] for t in tab]
x=[math.log(n) for n in ns]
ax[0,0].vlines(x,0,[l/n**2 for l,n in zip(lam,ns)],color=C[0],lw=2,label='from Frobenius classes')
ax[0,0].plot(x,[r/n**2 for r,n in zip(rec,ns)],'o',ms=3.5,mfc='none',color=C[1],label='Bohr means of the Thorin density')
ax[0,0].axhline(0,color='k',lw=0.4); ax[0,0].set_xlabel(r'$x=\log n$'); ax[0,0].set_ylabel(r'$\Lambda(n)n^{-2}$')
ax[0,0].legend(frameon=False,fontsize=6.5); ax[0,0].set_title('Galois face: Lévy measure at $\\alpha=2$',fontsize=8)
# (b) Thorin face at alpha=1.5 (galois) and fake
rs=np.array(A['rs']); gG=np.array(A['gG']); gF=np.array(A['gF'])
ax[0,1].plot(rs,gG,color=C[0],lw=1.1,label='Frobenius $=$ $q$ expansion')
ax[0,1].plot(rs,gF,color=C[1],lw=0.8,alpha=0.8,label=r'one wrong coefficient, $a_{59}$')
ax[0,1].set_xlabel(r'$r=\sqrt{z}$'); ax[0,1].set_ylabel(r'$2\pi r\,U_\alpha(r^2)$'); ax[0,1].set_xlim(0,40)
ax[0,1].legend(frameon=False,fontsize=6.5); ax[0,1].set_title(r'transport face: $\alpha=1.5$ (signed)',fontsize=8)
# (c) automorphic face: Polya kernel from Artin L on the critical line
us=np.array(B['us']); ax[1,0].plot(us,B['Phi_aut'],color=C[0],lw=1.4,label=r'$2f(ie^u/\sqrt{23})e^{u/2}$, $f=\eta(z)\eta(23z)$')
ax[1,0].plot(us[::4],np.array(B['Phi_inv'])[::4],'o',ms=3,mfc='none',color=C[1],label=r'inverse transform of $\Lambda(\frac{1}{2}+it,\rho)$')
ax[1,0].set_xlabel('$u$'); ax[1,0].set_ylabel(r'$\Phi(u)$'); ax[1,0].legend(frameon=False,fontsize=5.8,loc='upper left')
ax[1,0].set_title('automorphic face: from the Artin $L$ function',fontsize=8)
# (d) zero face: real Z function on the critical line
import mpmath as mp
ts=np.array(B['ts']); Lm=np.array(B['Lam'])
Zt=np.array([l/float(23**0.25*2*(2*mp.pi)**-0.5*abs(mp.gamma(0.5+1j*t))) for l,t in zip(Lm,ts)])
ax[1,1].plot(ts,Zt,color=C[2],lw=1.0); ax[1,1].axhline(0,color='k',lw=0.4)
ax[1,1].set_xlim(0,30); ax[1,1].set_xlabel('$t$'); ax[1,1].set_ylabel(r'$Z(t)=\Lambda(\frac{1}{2}+it)/|\gamma(\frac{1}{2}+it)|$'); ax[1,1].set_title('zero face: the critical line',fontsize=8)
plt.tight_layout(); plt.savefig('figs/lap_faces.pdf'); plt.close()
# Figure 2: Cauchy smoothing for zeta
fig,ax=plt.subplots(1,2,figsize=(6.6,2.7))
for a,c in [('0.5',C[0]),('0.2',C[3]),('0.05',C[1])]:
    d=Z['cauchy'][a]; r=np.array(d['r'])
    ax[0].plot(r,d['V'],color=c,lw=0.9,label=f'$a={a}$')
G=json.load(open('zeros1000.json'))
for g in G[:10]: ax[0].axvline(g,color='0.75',lw=0.5,zorder=0)
ax[0].set_xlim(0,50); ax[0].set_yscale('log'); ax[0].set_ylim(5e-4,30); ax[0].set_xlabel(r'$r=\sqrt{z}$'); ax[0].set_ylabel(r'$2r\,V_a(r^2)$'); ax[0].legend(frameon=False,fontsize=7)
ax[0].set_title(r'Thorin densities of $\xi$ at $\frac{1}{2}+a$',fontsize=8)
H=Z['H2']; t=np.array(H['t']); dd=np.array(H['dens'])
ax[1].loglog(t,dd,color=C[0],lw=1.3,label='Talbot inversion')
ax[1].loglog(t[t>5],Z['tailA']*t[t>5]**-2.5,'--',color=C[1],lw=1,label=r'$-\frac{(\log\zeta)^{\prime\prime\prime}(2)}{8\sqrt{\pi}}\,t^{-5/2}$')
pw=Z['H']['pw']; tp=[0.05,0.1,0.2,0.4]
for k,mk in [('4','v'),('32','^')]: ax[1].plot(tp,pw[k],mk,ms=4,color=C[2],mfc='none',label=f'Post Widder, $k={k}$')
ax[1].set_xlabel('$t$'); ax[1].set_ylabel('density of $H$'); ax[1].legend(frameon=False,fontsize=6.3); ax[1].set_ylim(1e-9,20)
ax[1].set_title(r'the Thorin law of $\zeta$ at $\alpha=2$',fontsize=8)
plt.tight_layout(); plt.savefig('figs/lap_zeta.pdf'); plt.close()
