import numpy as np, json, math
from scipy.special import exp1
d=np.load('data.npz'); P=d['P'].astype(float); c1=d['c1'].copy(); X=1e7
c1[P==23]=3   # ramified prime excluded from the classes
haar=np.array([1/6,1/2,1/3]); out={}
# Mertens per class: sum_{p in C} 1/p - delta log log x ; sum_{p in C} log p/p - delta log x
ll=math.log(math.log(X)); lx=math.log(X)
out['mertens_inv']=[float(np.sum(1/P[c1==c])-haar[c]*ll) for c in range(3)]
out['mertens_log']=[float(np.sum(np.log(P[c1==c])/P[c1==c])-haar[c]*lx) for c in range(3)]
out['B_total']=float(np.sum(1/P)-ll); out['sum_inv_pp1']=float(np.sum(1/(P*(P-1))))
# predicted class shares for size biased pick ~ (delta log x + c_C)/(log x + sum c) and for distinct count ~ (delta loglog x + b_C)/(loglog x + B)
gam=0.5772156649015329
out['kappa']=[float(out['mertens_log'][c]+np.sum(np.log(P[c1==c])/(P[c1==c]*(P[c1==c]-1)))+gam*haar[c]) for c in range(3)]
out['pred_sb']=[float(haar[c]+out['kappa'][c]/(lx-1)) for c in range(3)]
out['sum_logp_pp1']=float(np.sum(np.log(P)/(P*(P-1))))
out['pred_count']=[float((haar[c]*ll+out['mertens_inv'][c])/(ll+sum(out['mertens_inv']))) for c in range(3)]
print(out)
# Markov Krein / Dirichlet mean check: Q = trace^2 on S3: values 4,0,1 with Haar weights 1/6,1/2,1/3
rng=np.random.default_rng(3); M=2000000
W=rng.dirichlet(haar,size=M); G=rng.gamma(1.0,size=M)
H=G*(4*W[:,0]+W[:,2])
lam=np.linspace(0,6,25)
mc=np.array([np.mean(np.exp(-l*H)) for l in lam]); th=np.exp(-(np.log(1+4*lam)/6+np.log(1+lam)/3))
# exact sigma=1.01 Euler product (Frobenius functional Xi_sigma(Q), zeta random integer, primes<1e7, PNT tail), as in Chapter 17
sig=1.01; Q=np.where(c1==0,4.0,np.where(c1==2,1.0,0.0)); mask=(P!=23)
def lapl(l):
    s2=sig+l*(sig-1)
    logf=np.sum(np.log1p(-P[mask]**-sig)-np.log1p(-P[mask]**(-(sig+l*(sig-1)*Q[mask]))))   # E exp(-l (sigma-1) e_p log p Q_p) with e_p geometric
    tail=sum(w*(exp1((sig+l*(sig-1)*q-1)*math.log(X))-exp1((sig-1)*math.log(X))) for w,q in [(1/6,4.0),(1/3,1.0)])
    return math.exp(logf+tail)
ex=np.array([lapl(l) for l in lam])
out['lam']=lam.tolist(); out['mc']=mc.tolist(); out['th']=th.tolist(); out['exact101']=ex.tolist()
out['mc_err']=float(np.max(np.abs(mc-th))); out['ex_err']=float(np.max(np.abs(ex-th)))
print('markov krein mc err',out['mc_err'],'sigma=1.01 vs limit',out['ex_err'])
# PD(1) facts for the chapter: Golomb Dickman, E V2, size biased uniform
V=np.sort(W,axis=1)  # not PD; instead simulate PD(1) by stick breaking
U=rng.uniform(size=(200000,60)); st=np.cumprod(1-U,axis=1); st=np.concatenate([np.ones((200000,1)),st[:,:-1]],1)*U
V=np.sort(st,axis=1)[:,::-1]
out['pd_EV1']=float(V[:,0].mean()); out['pd_EV2']=float(V[:,1].mean())
print('PD(1) E V1',out['pd_EV1'],'E V2',out['pd_EV2'])
json.dump(out,open('pd2.json','w'))
