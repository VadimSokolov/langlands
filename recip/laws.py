import numpy as np, json
from scipy.special import exp1
d=np.load('data.npz'); P=d['P'].astype(float); X=1e7
S={2,7,11,23,31,37}; keep=~np.isin(d['P'],list(S))
P=P[keep]; c1=d['c1'][keep]; c2=d['c2'][keep]; l7=d['leg7'][keep]; p7=d['p7'][keep]; x11=d['x11'][keep]; x37=d['x37'][keep]
chi=np.array([2.,0.,-1.]); mu=np.array([1/6,1/2,1/3])
# semicircle quadrature
th=np.linspace(0,np.pi,401)[1:-1]; w=np.sin(th)**2; w/=w.sum(); xs=2*np.cos(th)
QST=(xs[:,None]-xs[None,:])**2; WST=w[:,None]*w[None,:]
def logL_fin(sig,lam,Q):
    q=P**(-sig); t=P**(-(sig-1)*lam*Q)
    return np.sum(np.log1p(-q)-np.log1p(-q*t))
def tail(sig,lam,vals,probs):
    xX=(sig-1)*np.log(X)
    return -np.sum(probs*(exp1(xX)-exp1((1+lam*vals)*xX)))
def limit(lam,vals,probs): return -np.sum(probs*np.log1p(lam*vals))
cases={}
# S3 classes
for k,name in enumerate(['identity','transposition','threecycle']):
    Q=(c1==k).astype(float); cases[name]=(Q,np.array([1.]),np.array([mu[k]]))
Qp=(chi[c1]-chi[c2])**2
vals=np.array([(a-b)**2 for a in chi for b in chi]); probs=np.array([m1*m2 for m1 in mu for m2 in mu])
cases['rho_rho31']=(Qp,vals,probs)
cases['qr_forgot']=(((l7-p7)**2).astype(float),np.array([4.]),np.array([.5]))
cases['ell_pair']=((x11-x37)**2,QST.ravel(),WST.ravel())
sigs=[1.5,1.2,1.1,1.05,1.02,1.01]; lams=np.linspace(0,5,26)
out={'sigs':sigs,'lams':lams.tolist(),'fin':{},'lim':{}}
for name,(Q,vals,probs) in cases.items():
    out['lim'][name]=[limit(l,vals,probs) for l in lams]
    out['fin'][name]={str(s):[logL_fin(s,l,Q)+tail(s,l,vals,probs) for l in lams] for s in sigs}
    print(name,'lam=1 limit',round(out['lim'][name][5],5),[round(out['fin'][name][str(s)][5],5) for s in sigs])
# mean check (Thorin): E Q under limit
print('means',{n:float(np.sum(v[1]*v[2])) for n,v in cases.items()})
# zero probabilities
sz=np.logspace(-3,-0.3,30)+1; out['zero']={'s':sz.tolist()}
for name in ['rho_rho31','qr_forgot','ell_pair','transposition']:
    Q,vals,probs=cases[name]; m=Q>1e-12; mass=float(np.sum(probs[vals>1e-12]))
    lp=[float(np.sum(np.log1p(-P[m]**(-s)))-mass*exp1((s-1)*np.log(X))) for s in sz]
    slope=np.polyfit(np.log(sz[:10]-1),lp[:10],1)[0]
    out['zero'][name]={'logP':lp,'mass':mass,'slope':float(slope)}; print(name,'mass',mass,'slope',slope)
# reciprocity exact: eta product equals rho -> Q identically zero (checked in chapter); impostor at 2 excluded by S
json.dump(out,open('laws.json','w'))
