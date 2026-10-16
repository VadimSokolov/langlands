import numpy as np, json, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
D="figs/"
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D","#3B8B5A","#A23B72"]
# 1 Kemeny law
r=json.load(open("formfactor.json"))
fig,ax=plt.subplots(1,3,figsize=(6.6,2.3))
for k,N in enumerate(("16033","32029","48049")):
    F=np.array(r[N]["F"]); j=np.arange(len(F)); part=np.cumsum(1.5*2.0**(-j/2)*F)
    ax[0].plot(j,part,"o-",ms=2,lw=0.9,color=C[k],label=f"$N={N}$")
ax[0].set_xlabel("truncation $J$"); ax[0].set_ylabel(r"partial sum of $K_N-2h$"); ax[0].legend(frameon=False,fontsize=6.5); ax[0].set_title("Hecke series",fontsize=8)
for a,l,col in ((ax[1],2,C[0]),(ax[2],3,C[1])):
    J=26 if l==2 else 16
    E=np.load(f"kemeny_l{l}_J{J}_model.npy"); lev=np.load(f"kemeny_l{l}_J{J}_levels.npy")
    g=np.linspace(np.percentile(E,0.2),np.percentile(E,99.8),300)
    a.hist(lev[:,2],bins=22,density=True,color=col,alpha=0.55,label="207 levels")
    a.plot(g,gaussian_kde(E)(g),color="#2E1F27",lw=1.3,label="random sign law")
    a.set_xlabel(rf"$K_N-\frac{{{l}}}{{{l-1}}}h$, $\ell={l}$"); a.legend(frameon=False,fontsize=6.5)
ax[1].set_title("observed vs limit model",fontsize=8); ax[2].set_title("observed vs limit model",fontsize=8)
plt.tight_layout(); plt.savefig(D+"kemeny_law.pdf"); plt.close()
# 2 vertex hitting
vh={}
for f in ("vertex_hit_1009.json","vertex_hit_8017.json"): vh.update(json.load(open(f)))
Ns=sorted(int(k) for k in vh); x=np.log2(Ns); y=np.array([vh[str(n)]["median_minus_2M"] for n in Ns])
q10=np.array([vh[str(n)]["q10"] for n in Ns]); q90=np.array([vh[str(n)]["q90"] for n in Ns])
fig,ax=plt.subplots(1,2,figsize=(6.4,2.4))
Et=np.load("Etau32029.npy"); dev=Et-2*(32029-1)/12
ax[0].hist(np.clip(dev,-60,80),bins=56,range=(-60,80),color=C[3],alpha=0.8)
nb=int((dev>80).sum()); ax[0].text(0.97,0.92,f"{nb} curves above 80\nmaximum {dev.max():.0f}",transform=ax[0].transAxes,ha="right",va="top",fontsize=6.5)
ax[0].axvline(np.median(dev),color=C[1],lw=1); ax[0].set_xlabel(r"$\mathbb{E}_\pi\tau_v-2h$, $N=32029$"); ax[0].set_title("mean hitting time of each curve",fontsize=8)
ax[1].fill_between(x,q10,q90,color=C[2],alpha=0.25,label="10% to 90%")
ax[1].plot(x,y,"o",color=C[1],label="median")
b=np.polyfit(x,y,1); xx=np.linspace(x.min()-0.3,x.max()+0.3,10)
ax[1].plot(xx,np.polyval(b,xx),"--",color=C[1],lw=1,label=f"fit, slope {b[0]:.2f}")
ax[1].plot(xx,-3*xx+np.mean(y+3*x),":",color="#2E1F27",lw=1.2,label="slope $-3$")
ax[1].set_xlabel(r"$\log_2 N$"); ax[1].legend(frameon=False,fontsize=6.3); ax[1].set_title("typical correction",fontsize=8)
plt.tight_layout(); plt.savefig(D+"vertex_hitting.pdf"); plt.close()
print("fit",b)
# 3 form factor
rm=json.load(open("ramp_model.json"))["ramp_model"]; kv=np.array(r["km_var"])
fig,ax=plt.subplots(figsize=(4.6,2.6))
for k,N in enumerate(("16033","32029","48049")):
    F=np.array(r[N]["F"]); h=r[N]["h"]; j=np.arange(1,len(F))
    ax.semilogy(j,F[1:]**2,"o",ms=2.3,color=C[k],alpha=0.8,label=f"$N={N}$")
    ax.plot([2*math.log2(h)-3,60],[h*2,h*2],color=C[k],lw=1)
jj=np.arange(1,40); ax.plot(jj,2.7*2.0**(jj/2),"--",color="#2E1F27",lw=1,label=r"ramp $2.7\cdot2^{j/2}$ (class numbers)")
ax.plot([t[0] for t in rm],[t[1] for t in rm],"-",color="#2E1F27",lw=1.4)
ax.set_ylim(1e-3,1e5); ax.set_xlabel("$j$"); ax.set_ylabel(r"$F_j(N)^2$"); ax.legend(frameon=False,fontsize=6.3,loc="lower right")
ax.set_title("arithmetic form factor: ramp, then Poisson plateau $2h$",fontsize=8)
plt.savefig(D+"form_factor.pdf"); plt.close()
# 4 spacings
s=np.load("spacings_iso.npy"); rr=np.load("ratios_iso.npy"); sp=json.load(open("spectra.json"))
fig,ax=plt.subplots(1,2,figsize=(6.2,2.3))
g=np.linspace(0,4,300)
ax[0].hist(s,bins=50,range=(0,4),density=True,color=C[4],alpha=0.7)
ax[0].plot(g,np.exp(-g),color="#2E1F27",lw=1.3,label="Poisson")
ax[0].plot(g,np.pi/2*g*np.exp(-np.pi*g*g/4),"--",color=C[1],lw=1.3,label="GOE surmise")
ax[0].set_xlabel("unfolded spacing"); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title("eigenvalues of $T_2$, three levels",fontsize=8)
t=np.linspace(0,1,200)
ax[1].hist(rr,bins=30,range=(0,1),density=True,color=C[5],alpha=0.6)
ax[1].plot(t,2/(1+t)**2,color="#2E1F27",lw=1.3,label="Poisson, mean 0.386")
ax[1].plot(t,2*27/8*(t+t*t)/(1+t+t*t)**2.5,"--",color=C[1],lw=1.3,label="GOE, mean 0.531")
ax[1].set_xlabel(r"ratio $\tilde r$"); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title(f"spacing ratios, observed mean {sp['pooled']['mean_r']:.3f}",fontsize=8)
plt.tight_layout(); plt.savefig(D+"isogeny_spacings.pdf"); plt.close()
# 5 Harper Hecke
hs=json.load(open("harper_st.json")); xs=sorted(int(k) for k in hs)
fig,ax=plt.subplots(figsize=(4.2,2.4))
ax.errorbar(xs,[hs[str(x)]["hecke"] for x in xs],yerr=[hs[str(x)]["hecke_sd_of_mean"] for x in xs],fmt="o-",color=C[1],ms=3,label="Sato Tate Hecke model")
ax.plot(xs,[hs[str(x)]["steinhaus"] for x in xs],"s-",color=C[0],ms=3,label="Steinhaus")
ll=np.array([hs[str(x)]["loglog^-1/4"] for x in xs])
ax.plot(xs,0.92*ll,"--",color=C[0],lw=0.9); ax.plot(xs,0.83*ll,"--",color=C[1],lw=0.9,label=r"$c(\log\log x)^{-1/4}$")
ax.set_xscale("log"); ax.set_xlabel("$x$"); ax.set_ylabel(r"$\mathbb{E}|S(x)|/\sqrt{x}$"); ax.legend(frameon=False,fontsize=7)
plt.savefig(D+"harper_hecke.pdf"); plt.close()
print("ok")
