import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"font.family":"serif"})
d=np.load("w_100_400.npz"); R,ks,dims,eps,p=d["R"],d["ks"],d["dims"],d["eps"],d["p"]
nb=40; edges=np.linspace(0,2,nb+1); mid=(edges[1:]+edges[:-1])/2
def dens(sel):
    Y=p[:,None]/ks[None,sel]**2; Z=eps[sel][None,:]*R[:,sel]*ks[None,sel]; W=np.broadcast_to(dims[sel][None,:],Y.shape).astype(float); m=Y<=2
    k=np.digitize(Y[m],edges)-1; return np.bincount(k,Z[m],nb)/np.bincount(k,W[m],nb)
a=dens((ks>=100)&(ks<200)); b=dens((ks>=200)&(ks<400))
fig,ax=plt.subplots(figsize=(3.8,2.3))
ax.step(mid,a,where="mid",color="#9C3D1F",lw=.9,label="$100\\leq k<200$"); ax.step(mid,b,where="mid",color="#1F3B63",lw=.9,label="$200\\leq k<400$")
ax.axhline(0,color="k",lw=.3); ax.axvline(.25,color="0.6",lw=.5,ls=":")
ax.set_xlabel("$y=p/k^2$"); ax.set_ylabel("$k$ times mean of $\\varepsilon_k\\lambda_f(p)$"); ax.legend(frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig("murm_weight.pdf")
out=dict(corr=float(np.corrcoef(a,b)[0,1]),nweights=int(len(ks)),nforms=int(dims.sum()),nprimes=int(len(p)),
 forms_lo=int(dims[(ks<200)].sum()),forms_hi=int(dims[ks>=200].sum()),first_bins=np.c_[a,b][:6].round(2).tolist())
json.dump(out,open("wfig.json","w"),indent=1); print(out)
