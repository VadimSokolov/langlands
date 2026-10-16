import json, numpy as np, math, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=['#1F3B63','#9C3D1F','#4E8F5A','#B08A2E']
G=json.load(open('gg_mple.json')); H=json.load(open('checks.json'))['hecke']
fig,ax=plt.subplots(1,3,figsize=(6.8,2.5))
th=np.linspace(0,math.pi,400); edges=np.linspace(0,math.pi,91); mid=(edges[1:]+edges[:-1])/2
for k,(name,title) in enumerate([('11a1','11a1: Sato Tate'),('32a1','32a: complex multiplication')]):
    a=ax[k]; a.bar(mid,G[name]['hist'],width=edges[1]-edges[0],color='0.85',edgecolor='none',label='Frobenius angles')
    a.plot(th,2/math.pi*np.sin(th)**2,'--',color='k',lw=0.9,label=r'$\frac{2}{\pi}\sin^2\theta$')
    for al,c in [('0.01',C[1]),('1.0',C[0])]:
        b=np.array(G[name][al]['b']); g=sum(b[j]*np.sin((j+1)*th) for j in range(len(b)))*math.sqrt(2/math.pi)
        a.plot(th,g**2,color=c,lw=1.1,label=rf'Good Gaskins, $\alpha={float(al):g}$')
    a.set_xlabel(r'$\theta_p$'); a.set_title(title,fontsize=8); a.set_xlim(0,math.pi)
    if k==1: a.set_ylim(0,1.6)
ax[0].legend(frameon=False,fontsize=5.8,loc='upper left')
t=np.array(H['ts']); ax[2].plot(t,H['pdf'],color=C[0],lw=1.3,label='partial fractions')
ax[2].plot(t[::3],np.array(H['pdf_dd'])[::3],'o',ms=3,mfc='none',color=C[1],label='B spline divided difference')
ax[2].plot(H['mids'][::4],np.array(H['hist'])[::4],'.',ms=2.5,color=C[2],label=r'Gamma $\times$ B spline sample')
ax[2].set_xlabel('$t$'); ax[2].set_title(r'Hecke GGC $H_{67}(2)$',fontsize=8); ax[2].legend(frameon=False,fontsize=5.6)
plt.tight_layout(); plt.savefig('figs/lap_spline.pdf'); plt.close()
