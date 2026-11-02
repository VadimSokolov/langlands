import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"font.family":"serif"})
ink,rust="#1F3B63","#9C3D1F"
fn="/root/ecdata_dl/ecdata/allcurves/allcurves.00000-09999"
def load(nm,lo,hi):
    d=np.load(nm); r=[int(l.split()[4]) for l in open(fn) if lo<=int(l.split()[0])<hi and l.split()[2]=='1']
    return d["A"].astype(float),d["eps"],d["N"],d["p"],np.array(r)
W={"a":load("m2k.npz",2500,5000),"b":load("m5k.npz",5000,10000)}
edges=np.linspace(0,3,121); mid=(edges[1:]+edges[:-1])/2
def D(A,w,N,p,I=None):
    if I is None: I=np.arange(len(N))
    Y=p[None,:]/N[I,None]; Z=w[I,None]*A[I]; m=Y<=3
    k=np.digitize(Y[m],edges)-1; c=np.bincount(k,None,120); s=np.bincount(k,Z[m],120); s2=np.bincount(k,Z[m]**2,120)
    mu=s/c; return mu, np.sqrt(s2/c-mu**2)/np.sqrt(c)
Da,sa=D(*W["a"][:4]); Db,sb=D(*W["b"][:4])
out={"corr":float(np.corrcoef(Da,Db)[0,1]),"rms_a":float(np.sqrt((Da**2).mean())),"rms_b":float(np.sqrt((Db**2).mean())),
     "maxse_b":float(sb.max()),"Db_first":Db[:4].round(3).tolist()}
s=np.sign(Db); out["zeros_b"]=mid[1:][s[1:]!=s[:-1]].round(3).tolist(); out["argmin_b"]=float(mid[Db.argmin()]); out["min_b"]=float(Db.min())
out["argmax_b_after0.3"]=float(mid[30:][Db[30:].argmax()]); out["max_b_after0.3"]=float(Db[30:].max())
np.savetxt("murm_D.csv",np.c_[mid,Da,sa,Db,sb],delimiter=",",header="y,D_2500,se_2500,D_5000,se_5000",fmt="%.5f")
# Fig A: raw by parity vs p (window b) and scale invariant overlay
A,eps,N,p,r=W["b"]
fig,ax=plt.subplots(1,2,figsize=(5.6,2.3))
def binp(y,nb=150):
    e=np.linspace(p.min(),p.max(),nb+1); i=np.clip(np.digitize(p,e)-1,0,nb-1)
    return np.array([p[i==k].mean() for k in range(nb)]),np.array([y[i==k].mean() for k in range(nb)])
for e,c,lab in [(1,ink,"root number $+1$"),(-1,rust,"root number $-1$")]:
    x,y=binp(A[eps==e].mean(0)); ax[0].plot(x,y,color=c,lw=.8,label=lab)
ax[0].axhline(0,color="k",lw=.3); ax[0].set_xlabel("$p$"); ax[0].set_ylabel("mean $a_p$"); ax[0].legend(frameon=False,fontsize=7)
ax[1].plot(mid,Da,color=rust,lw=.9,label="$2500\\leq N<5000$"); ax[1].plot(mid,Db,color=ink,lw=.9,label="$5000\\leq N<10000$")
ax[1].axhline(0,color="k",lw=.3); ax[1].set_xlabel("$y=p/N$"); ax[1].set_ylabel("$D_X(y)$"); ax[1].legend(frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig("murm_density.pdf")
# Fig B: by rank
fig,ax=plt.subplots(figsize=(3.6,2.3))
for rr,c in [(0,ink),(1,rust),(2,"0.45")]:
    I=np.where(r==rr)[0]; m,_=D(A,np.ones(len(N)),N,p,I); ax.plot(mid,m,color=c,lw=.8,label=f"rank {rr} ({len(I)})")
ax.axhline(0,color="k",lw=.3); ax.set_xlabel("$y=p/N$"); ax.set_ylabel("mean $a_p$"); ax.legend(frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig("murm_rank.pdf")
json.dump(out,open("figs_ch.json","w"),indent=1); print(out)
