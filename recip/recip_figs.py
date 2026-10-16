import numpy as np, json, math, mpmath as mp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D","#3B8B5A","#A23B72"]
k=np.load("kernels.npy"); us,Kt,Ki,Dt,Di=k
fig,ax=plt.subplots(1,2,figsize=(6.6,2.5))
for a,(t,i,lab) in zip(ax,((Kt,Ki,r"$S_3$ representation of $x^3-x-1$, $N=23$"),(Dt,Di,r"$\Delta$, level one"))):
    a.plot(us,t,color=C[0],lw=1.6,label=r"$\Phi(u)$, true Frobenius")
    a.plot(us,t[::-1],"--",color=C[0],lw=1,alpha=0.8,label=r"$\Phi(-u)$")
    a.plot(us,i,color=C[1],lw=1.4,label=r"$\Phi(u)$, Frobenius at $2$ altered")
    a.plot(us,i[::-1],"--",color=C[1],lw=1,alpha=0.8,label=r"$\Phi(-u)$")
    a.set_xlabel("$u$"); a.set_title(lab,fontsize=8)
ax[0].legend(frameon=False,fontsize=6.3,loc="upper left"); ax[1].set_xlim(-1.5,1.5)
plt.tight_layout(); plt.savefig("figs/recip_kernel.pdf"); plt.close()
o=json.load(open("recip.json")); h=np.array(o["S3_heat"])
fig,ax=plt.subplots(1,2,figsize=(6.6,2.5))
ax[0].semilogy(h[:,0],h[:,1],"o-",color=C[0],ms=4,label="sum over zeros")
ax[0].semilogy(h[:,0],h[:,2],"x",color=C[1],ms=7,label="from Frobenius traces")
ax[0].set_xlabel("$x$"); ax[0].set_ylabel(r"$\mathcal{W}_\rho(x)$"); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title("the swap: primes reproduce zeros",fontsize=8)
ax[1].plot(h[:,0],h[:,2],"o-",color=C[0],ms=4,label="true Frobenius at 2")
ax[1].plot(h[:,0],h[:,3],"s-",color=C[1],ms=4,label="Frobenius at 2 altered")
ax[1].axhline(0,color="k",lw=0.6); ax[1].set_xlabel("$x$"); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title("positivity fails after one change",fontsize=8)
plt.tight_layout(); plt.savefig("figs/recip_heat.pdf"); plt.close()
g=o["gamma_limit"]; lam=np.linspace(0,4,100)
fig,ax=plt.subplots(1,2,figsize=(6.4,2.4))
for a,m in zip(ax,(1,4)):
    a.plot(lam,(1+lam)**(-m),color="k",lw=1.4,label=fr"Gamma$({m})$: $(1+\lambda)^{{-{m}}}$")
    for j,s in enumerate((1.1,1.01,1.001)):
        v=np.array(g[f"m{m}_s{s}"]); a.plot(v[:,0],v[:,1],"o",color=C[j],ms=4,label=fr"$\sigma={s}$")
    a.set_xlabel(r"$\lambda$"); a.set_yscale("log") if m==4 else None; a.set_title(r"$\rho$ irreducible, $\langle\rho,\rho\rangle=1$" if m==1 else r"$\rho\oplus\rho$, $\langle\rho,\rho\rangle=4$",fontsize=8)
    a.legend(frameon=False,fontsize=6.5)
ax[0].set_ylabel(r"$\mathbb{E}\,e^{-\lambda(\sigma-1)X_\sigma}$")
plt.tight_layout(); plt.savefig("figs/recip_gamma.pdf"); plt.close()
# schematic of the swap
fig,ax=plt.subplots(figsize=(7.2,4.0)); ax.set_xlim(0,10); ax.set_ylim(0,5.6); ax.axis("off")
def box(x,y,w,hh,txt,col):
    ax.add_patch(FancyBboxPatch((x,y),w,hh,boxstyle="round,pad=0.08",fc=col,ec="none",alpha=0.9))
    ax.text(x+w/2,y+hh/2,txt,ha="center",va="center",fontsize=8,color="white")
box(0.2,3.2,3.3,1.5,"Galois side\nFrobenius traces $a_\\rho(p)$\nEuler product on $\\mathrm{Re}\\,s>1$\nGaussian member $H_\\rho$ (GGC)",C[0])
box(6.5,3.2,3.3,1.5,"Automorphic side\nform $f$ on $iy$, Hecke eigenvalues\ncritical line $\\mathrm{Re}\\,s=\\frac{1}{2}$\nkernel member $W_\\rho$ (Polya)",C[1])
box(3.55,0.25,2.9,1.45,"zeros $\\frac{1}{2}\\pm i\\gamma_k$\nThorin measure $\\sum\\delta_{\\gamma_k^2}$\nheat trace $\\sum e^{-x\\gamma_k^2}$",C[3])
def arr(p1,p2,txt,col,rad=0.0,off=(0,0)):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="<|-|>",mutation_scale=11,color=col,lw=1.4,connectionstyle=f"arc3,rad={rad}"))
    ax.text((p1[0]+p2[0])/2+off[0],(p1[1]+p2[1])/2+off[1],txt,ha="center",va="center",fontsize=7,color=col)
arr((3.6,4.0),(6.4,4.0),"van Dantzig involution, Wald couple\n$\\mathbb{E}e^{sW}\\,\\mathbb{E}e^{-s^2H}=1$",C[2],0.0,(0,0.9))
arr((1.9,3.1),(3.5,1.2),"explicit formula\n$2\\sin^2$ transport",C[4],0.25,(-0.55,0.1))
arr((8.1,3.1),(6.5,1.2),"Hadamard product\nGRH = positivity",C[5],-0.25,(0.6,0.1))
ax.text(5,2.6,"reciprocity: the two sides\nare one function",ha="center",fontsize=7.5,style="italic",color="#333333")
plt.savefig("figs/recip_swap.pdf"); plt.close(); print("ok")
