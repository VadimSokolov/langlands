# SU(2) = unit quaternions S^3; von Mises Fisher noise (conjugation and left invariant). Class angle theta = arccos(g_0).
import numpy as np, json, math
rng=np.random.default_rng(7)
def vmf_s3(mu,kappa,n):
    # Wood (1994) for p=4
    p=4; b=(-2*kappa+math.sqrt(4*kappa**2+(p-1)**2))/(p-1); x0=(1-b)/(1+b); c=kappa*x0+(p-1)*math.log(1-x0**2)
    out=[]
    while len(out)<n:
        z=rng.beta((p-1)/2,(p-1)/2); w=(1-(1+b)*z)/(1-(1-b)*z); u=rng.uniform()
        if kappa*w+(p-1)*math.log(1-x0*w)-c>=math.log(u): out.append(w)
    w=np.array(out); v=rng.normal(size=(n,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
    X=np.column_stack([w,np.sqrt(1-w*w)[:,None]*v])   # mean direction e0
    # rotate e0 -> mu by left multiplication with quaternion mu (maps 1 to mu)
    def qmul(a,b):
        a0,a1,a2,a3=a.T if a.ndim>1 else a; b0,b1,b2,b3=b.T
        return np.column_stack([a0*b0-a1*b1-a2*b2-a3*b3,a0*b1+a1*b0+a2*b3-a3*b2,a0*b2-a1*b3+a2*b0+a3*b1,a0*b3+a1*b2-a2*b1+a3*b0])
    return qmul(mu,X)
tg=np.linspace(1e-4,math.pi-1e-4,4000)
def post_quantiles(R,kappa,w):
    r=np.linalg.norm(R[1:]); a=kappa*r*np.sin(tg)
    logs=np.log(w(tg))+kappa*R[0]*np.cos(tg)+np.where(a>1e-8,np.log(np.sinh(np.minimum(a,700))+1e-300)-np.log(a+1e-300),0.0)
    logs-=logs.max(); p=np.exp(logs); c=np.cumsum(p); c/=c[-1]
    return np.interp([0.05,0.95],c,tg)
res={}
for kappa,n in ((5,5),(20,5),(5,20)):
    for th0 in (0.1,0.3,math.pi/2,2.8,3.05):
        u=np.array([1.0,1.0,1.0])/math.sqrt(3); mu=np.r_[math.cos(th0),math.sin(th0)*u]
        cov={"reference":0,"haar":0}; R_=3000
        for r in range(R_):
            X=vmf_s3(mu,kappa,n); R=X.sum(0)
            for name,w in (("reference",lambda t: np.ones_like(t)),("haar",lambda t: np.sin(t)**2)):
                lo,hi=post_quantiles(R,kappa,w); cov[name]+=(lo<=th0<=hi)
        res[f"k{kappa}_n{n}_th{th0:.2f}"]={k:v/R_ for k,v in cov.items()}
        print(kappa,n,round(th0,2),res[f"k{kappa}_n{n}_th{th0:.2f}"],flush=True)
json.dump(res,open("refprior_sim.json","w"),indent=1)
