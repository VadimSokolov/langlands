import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D","#3B8B5A","#A23B72"]
o=json.load(open("recip_gauss.json"))
fig,ax=plt.subplots(1,2,figsize=(6.6,3.0))
for c,(m,t) in zip(C,o["traj"].items()):
    ax[0].plot(t["re"],t["im"],color=c,lw=1.2,label=f"$m={m}$ ($m\\equiv{int(m)%4}$ mod $4$)")
    ax[0].plot(*t["lim"],"o",color=c,ms=5)
ax[0].plot([1],[0],"k^",ms=4); ax[0].annotate(r"$\varepsilon=1$",(1,0),xytext=(0.78,0.12),fontsize=7)
ax[0].axhline(0,color="k",lw=0.4); ax[0].axvline(0,color="k",lw=0.4)
ax[0].set_xlabel("real part"); ax[0].set_ylabel("imaginary part"); ax[0].set_aspect("equal")
ax[0].legend(frameon=False,fontsize=6.3,loc="upper center",bbox_to_anchor=(0.5,-0.38),ncol=2); ax[0].set_title(r"$\sqrt{\varepsilon}\,\theta(2/m+i\varepsilon)\to g(m)/m$",fontsize=8)
f=o["fejer"]; x=np.linspace(0,3,600)
for N,c in zip([2,4,8],C[:3]):
    r=x; F=np.where(np.abs(np.sin(np.pi*r))<1e-9,N,np.sin(np.pi*N*r)**2/(N*np.sin(np.pi*r)**2+1e-300))
    ax[1].plot(r,F,color=c,lw=1.1,label=f"$N={N}$")
ax[1].set_xlabel(r"$r=\sqrt{z}$"); ax[1].set_ylabel(r"$F_N(r)$"); ax[1].legend(frameon=False,fontsize=7)
ax[1].set_title("Fejér kernel: positive combs at $r=k$",fontsize=8)
plt.tight_layout(); plt.savefig("figs/recip_gauss.pdf"); plt.close()
