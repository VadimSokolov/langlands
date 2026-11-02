import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"font.family":"serif"})
d=np.load("w_100_400.npz"); R,ks,dims,eps,p=d["R"],d["ks"],d["dims"],d["eps"],d["p"]
edges=np.linspace(0,2,81); mid=(edges[1:]+edges[:-1])/2
def dens(sel,power):
    Y=p[:,None]/ks[None,sel]**2; Z=(eps[sel][None,:]*R[:,sel])*(p[:,None]**power) # sum over forms of eps*lambda*p^power
    W=np.broadcast_to(dims[sel][None,:],Y.shape).astype(float); m=Y<=2
    k=np.digitize(Y[m],edges)-1
    return np.bincount(k,Z[m],80)/np.bincount(k,W[m],80)
lo=(ks>=100)&(ks<200); hi=(ks>=200)&(ks<400)
out={}
for pw in [0,0.5]:
    a,b=dens(lo,pw),dens(hi,pw); s=(a@b)/(b@b)
    out[str(pw)]=dict(corr=float(np.corrcoef(a,b)[0,1]),rms_lo=float(np.sqrt((a**2).mean())),rms_hi=float(np.sqrt((b**2).mean())),scale_lo_over_hi=float(s))
print(json.dumps(out,indent=1))
a,b=dens(lo,0.5),dens(hi,0.5)
s=np.sign(b); print("zeros hi",mid[1:][s[1:]!=s[:-1]].round(3)); print("first",b[:6].round(3))
fig,ax=plt.subplots(figsize=(3.6,2.3))
ax.plot(mid,a,color="#9C3D1F",lw=.9,label="$100\\leq k<200$"); ax.plot(mid,b,color="#1F3B63",lw=.9,label="$200\\leq k<400$")
ax.axhline(0,color="k",lw=.3); ax.set_xlabel("$y=p/k^2$"); ax.set_ylabel("mean of $\\varepsilon_k\\,a_f(p)/p^{(k-2)/2}$"); ax.legend(frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig("murm_weight.pdf")
np.savetxt("murm_weight.csv",np.c_[mid,a,b],delimiter=",",header="y,mean_100_200,mean_200_400",fmt="%.5f")
json.dump(out,open("wanal.json","w"),indent=1)
