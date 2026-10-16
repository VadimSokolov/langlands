import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D"]
o=json.load(open('second.json'))
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
for c,(k,lab) in zip(C,[('C0','split completely'),('C1','transpositions'),('C2','three cycles')]):
    e=np.array(o[k]['eps']); ax[0].plot(e,o[k]['dev_over_eps'],'o-',ms=3,color=c,lw=1,label=lab)
    ax[0].plot([0],[o['kappa'][int(k[1])]],'*',ms=9,color=c)
ax[0].set_xlabel(r"$\sigma-1$"); ax[0].set_ylabel(r"$[\log\mathbb{E}e^{-L_C}+\mu(C)\log2]/(\sigma-1)$")
ax[0].legend(frameon=False,fontsize=6.5,loc="center right",bbox_to_anchor=(1.0,0.5)); ax[0].set_xlim(-0.002,0.042); ax[0].set_title(r"drift $\kappa_C$ (stars: predicted)",fontsize=8)
e=np.array(o['C0']['eps']); ax[1].plot(e,o['qr']['dev_over_eps'],'o-',ms=3,color=C[3],lw=1,label=r"$Q=|(\frac{7}{p})-(\frac{p}{7})|^2$")
ax[1].plot([0],[o['qr']['kappa']],'*',ms=9,color=C[3]); ax[1].set_xlim(-0.002,0.042)
ax[1].set_xlabel(r"$\sigma-1$"); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title(r"drift $\kappa_Q=\sum_C Q_C\kappa_C$",fontsize=8)
plt.tight_layout(); plt.savefig('figs/second_order.pdf'); plt.close()
