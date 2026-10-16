import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D","#3B8B5A","#A23B72"]
o=json.load(open("recip_law.json")); g=np.array(o["grid"])
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
lab={'rho_vs_f':(r"$\rho$ and $\eta(z)\eta(23z)$",C[0],'-'),'rho_vs_impostor':(r"$\rho$ and $\rho$ altered at $2$",C[1],'-'),
 'rho_vs_chi23':(r"$\rho$ and $\chi_{-23}$",C[2],'-'),'rho_vs_rho31':(r"$\rho$ and $\rho_{31}$ (same marginal law)",C[3],'-'),
 'rho_vs_Delta':(r"$\rho$ and $\Delta$",C[4],'-'),'Delta_vs_11a1':(r"$\Delta$ and $11a1$",C[5],'--')}
for k,(l,c,ls) in lab.items():
    y=np.array(o["run"][k]); y=np.maximum(y,1e-7)
    ax[0].semilogx(g,y,ls,color=c,lw=1.4,label=l)
ax[0].axhline(2,color="k",lw=0.5,ls=":"); ax[0].set_ylim(-0.1,2.6)
ax[0].set_xlabel("$X$"); ax[0].set_ylabel(r"$\frac{1}{\pi(X)}\sum_{p\leq X}|a(p)-b(p)|^2$")
ax[0].legend(frameon=False,fontsize=5.6,loc="lower right",bbox_to_anchor=(1.0,0.07)); ax[0].set_title("Frobenius distance: $0$ or $2$",fontsize=8)
r=np.arange(1,23); cm=o["condmean23"]
ax[1].bar(r-0.3,cm["chi"],0.2,color=C[2],label=r"$\chi_{-23}(p)$ (abelian)")
ax[1].bar(r-0.1,cm["rho"],0.2,color=C[0],label=r"$\operatorname{tr}\rho(\operatorname{Frob}_p)$")
ax[1].bar(r+0.1,cm["rho31"],0.2,color=C[3],label=r"$\operatorname{tr}\rho_{31}(\operatorname{Frob}_p)$")
ax[1].bar(r+0.3,cm["Delta"],0.2,color=C[4],label=r"$\tau(p)p^{-11/2}$")
ax[1].axhline(0,color="k",lw=0.5); ax[1].set_xlabel(r"residue class $r$ of $p$ mod $23$")
ax[1].set_ylabel("conditional mean"); ax[1].legend(frameon=False,fontsize=5.8,loc="lower left"); ax[1].set_ylim(-1.6,1.3)
ax[1].set_title("congruences see only the abelian part",fontsize=8)
plt.tight_layout(); plt.savefig("figs/recip_law.pdf"); plt.close()
