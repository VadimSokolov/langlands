import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9})
fn="/root/ecdata_dl/ecdata/allcurves/allcurves.00000-09999"
rank={}
for line in open(fn):
    f=line.split()
    if f[2]=='1': rank[int(f[0]),f[1]]=int(f[4])
def load(nm,lo,hi):
    d=np.load(nm); # ranks in same order as file scan
    r=[]
    for line in open(fn):
        f=line.split(); N=int(f[0])
        if lo<=N<hi and f[2]=='1': r.append(int(f[4]))
    return d["A"].astype(float),d["eps"],d["N"],d["p"],np.array(r)
W={"2500":load("m2k.npz",2500,5000),"5000":load("m5k.npz",5000,10000)}
edges=np.linspace(0,3,121); mid=(edges[1:]+edges[:-1])/2
def dens(A,eps,N,p,I=None):
    if I is None: I=np.arange(len(N))
    Y=p[None,:]/N[I,None]; Z=eps[I,None]*A[I]/np.sqrt(p)[None,:]; m=Y<=3
    k=np.digitize(Y[m],edges)-1; return np.bincount(k,Z[m],120)/np.bincount(k,None,120)
out={}
M={k:dens(*W[k][:4]) for k in W}
for k in W:
    A,eps,N,p,r=W[k]; out[k]={"classes":len(N),"rank_counts":{int(x):int((r==x).sum()) for x in np.unique(r)}}
out["corr_windows"]=float(np.corrcoef(M["2500"],M["5000"])[0,1])
out["max_absdiff"]=float(np.abs(M["2500"]-M["5000"]).max())
# by rank, window 5000, mean a_p/sqrt p (no eps) in y bins
A,eps,N,p,r=W["5000"]
Mr={}
for rr in [0,1,2]:
    I=np.where(r==rr)[0]; Y=p[None,:]/N[I,None]; Z=A[I]/np.sqrt(p)[None,:]; m=Y<=3
    k=np.digitize(Y[m],edges)-1; Mr[rr]=np.bincount(k,Z[m],120)/np.bincount(k,None,120)
    out[f"rank{rr}_mean_y_le3"]=float(Z[m].mean())
# figure 1: two windows
fig,ax=plt.subplots(1,2,figsize=(6.2,2.6))
ax[0].plot(mid,M["2500"],color="#9C3D1F",lw=1,label="2500 <= N < 5000")
ax[0].plot(mid,M["5000"],color="#1F3B63",lw=1,label="5000 <= N < 10000")
ax[0].axhline(0,color="k",lw=.4); ax[0].set_xlabel("y = p/N"); ax[0].set_ylabel("M_X(y)"); ax[0].legend(fontsize=7,frameon=False)
for rr,c in zip([0,1,2],["#1F3B63","#9C3D1F","0.5"]):
    ax[1].plot(mid,Mr[rr],color=c,lw=1,label=f"rank {rr}")
ax[1].axhline(0,color="k",lw=.4); ax[1].set_xlabel("y = p/N"); ax[1].set_ylabel("mean a_p / sqrt p"); ax[1].legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig("murm_windows.pdf")
# classifier: train on half of 5000, test other half; transfer to 2500
rng=np.random.default_rng(1); idx=rng.permutation(len(N)); tr,te=idx[:len(N)//2],idx[len(N)//2:]
Mt=dens(A,eps,N,p,tr)
def score(A,N,p,I,Mx,ymax):
    Y=p[None,:]/N[I,None]; k=np.clip(np.digitize(Y,edges)-1,0,119); Wt=np.where(Y<=ymax,Mx[k],0)
    return (Wt*A[I]/np.sqrt(p)[None,:]).sum(1), (Wt**2).sum(1)
res=[]
A2,eps2,N2,p2,r2=W["2500"]
for ymax in [0.25,0.5,1,1.5,2,3]:
    S,V=score(A,N,p,te,Mt,ymax); acc=(np.sign(S)==eps[te]).mean()
    S2,_=score(A2,N2,p2,np.arange(len(N2)),Mt,ymax); acc2=(np.sign(S2)==eps2).mean()
    # separation: mean of eps*S over sd
    es=eps[te]*S; mu,sd=es.mean(),es.std()
    res.append(dict(y=ymax,acc_test=round(float(acc),4),acc_transfer=round(float(acc2),4),
      mu=round(float(mu),3),sd=round(float(sd),3),V=round(float(V.mean()),4),hoeff=float(np.exp(-V.mean()/8)),
      gauss_err=float(0.5*__import__("math").erfc(mu/sd/np.sqrt(2)))))
out["classifier"]=res
# per rank accuracy at y=3
S,_=score(A,N,p,te,Mt,3)
out["acc_by_rank"]={int(x):round(float((np.sign(S)==eps[te])[r[te]==x].mean()),4) for x in [0,1,2]}
# score histogram figure
fig,ax=plt.subplots(figsize=(3.2,2.4))
for e,c in [(1,"#1F3B63"),(-1,"#9C3D1F")]:
    ax.hist(S[eps[te]==e],bins=60,histtype="step",color=c,label=f"root number {e:+d}")
ax.axvline(0,color="k",lw=.4); ax.set_xlabel("score S_E (y = 3)"); ax.legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig("murm_score.pdf")
json.dump(out,open("analysis.json","w"),indent=1); print(json.dumps(out,indent=1))
