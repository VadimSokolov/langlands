import numpy as np, math
from scipy.stats import norm
rng=np.random.default_rng(1)
def run(logf, sample, reps=1000, n=5, mu0=1.3, sig0=0.7, level=0.9, half=6.0):
    mus=np.linspace(-half,half,401); ls=np.linspace(-3,3,301); sg=np.exp(ls)
    cov={1:[0,0],2:[0,0]}
    for r in range(reps):
        yv=mu0+sig0*sample(n); m=np.median(yv); s=np.std(yv)
        MU=m+s*mus[:,None]; SG=s*sg[None,:]
        ll=sum(logf((yi-MU)/SG) for yi in yv)-n*np.log(SG)
        for pw in (1,2):
            lp=ll-pw*np.log(SG)+np.log(SG); w=np.exp(lp-lp.max()); w/=w.sum()
            cs=np.cumsum(w.sum(0)); cm=np.cumsum(w.sum(1))
            a,b=SG[0,np.searchsorted(cs,(1-level)/2)],SG[0,np.searchsorted(cs,(1+level)/2)]
            c,d=MU[np.searchsorted(cm,(1-level)/2),0],MU[np.searchsorted(cm,(1+level)/2),0]
            cov[pw][0]+=a<=sig0<=b; cov[pw][1]+=c<=mu0<=d
    return {k:(v[0]/reps,v[1]/reps) for k,v in cov.items()}
print("normal", run(lambda z: -z*z/2, lambda n: rng.standard_normal(n)))
us,ph=np.load("phi.npy"); du=us[1]-us[0]; cdf=np.cumsum(ph)*du; cdf/=cdf[-1]
lf=lambda z: np.log(np.maximum(np.interp(z,us,ph,left=0,right=0),1e-300))
print("delta kernel", run(lf, lambda n: np.interp(rng.uniform(size=n),cdf,us)))
