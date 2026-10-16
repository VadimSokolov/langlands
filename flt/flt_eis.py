# Eisenstein discrepancy Thorin shapes c(E,p): Haar formula, brute force check, and data for 11a1, 37a1
import numpy as np, json, itertools
from scipy.special import exp1
d=np.load('data.npz'); P=d['P']
a11=np.rint(d['x11']*np.sqrt(P)).astype(np.int64); a37=np.rint(d['x37']*np.sqrt(P)).astype(np.int64)
def cp(p): return 1-(p*p-2)/((p*p-1)*(p-1))
def brute(p):
    n=tot=0
    for a,b,c,e in itertools.product(range(p),repeat=4):
        det=(a*e-b*c)%p
        if det==0: continue
        tot+=1
        if (a+e-1-det)%p!=0: n+=1     # tr != 1+det  <=> no eigenvalue 1
    return n/tot
out={'formula':{},'brute':{},'share':{},'laplace':{}}
for p in [3,5,7,11,13]:
    out['formula'][p]=cp(p)
    if p<=7: out['brute'][p]=brute(p)
def share(a,N,p):
    m=(P!=p)&(N%P!=0)
    return float(np.mean(((a[m]-1-P[m])%p)!=0)), int(m.sum())
for name,a,N in [('11a1',a11,11),('37a1',a37,37)]:
    for p in [2,3,5,7,11,13]:
        out['share'][f'{name}_{p}']=share(a,N,p)
# Laplace transform of Xi_sigma(Q) at sigma=1.01 with PNT tail, vs (1+lam)^(-c)
X=1e7; lam=np.linspace(0,6,25)
def lap(a,N,p,sig):
    m=(P!=p)&(N%P!=0); D=P[m][((a[m]-1-P[m])%p)!=0].astype(float); c=len(D)/m.sum()
    res=[]
    for L in lam:
        s2=sig+L*(sig-1)
        logf=np.sum(np.log1p(-D**-sig)-np.log1p(-D**-s2))
        tail=c*(exp1((s2-1)*np.log(X))-exp1((sig-1)*np.log(X)))
        res.append(float(np.exp(logf+tail)))
    return res
for key,(a,N,p) in {'37a1_5':(a37,37,5),'37a1_7':(a37,37,7),'11a1_5':(a11,11,5),'11a1_7':(a11,11,7)}.items():
    for sig in [1.01,1.002]:
        out['laplace'][f'{key}_{sig}']=lap(a,N,p,sig)
out['lam']=lam.tolist()
# Mazur Eisenstein congruence check: 5 divides |11a1(F_l)| for all l != 11
m=P!=11; out['five_divides_11a1']=bool(np.all((P[m]+1-a11[m])%5==0))
print(json.dumps({k:v for k,v in out.items() if k!='laplace'},indent=1))
for k,v in out['laplace'].items(): print(k,[round(x,4) for x in v[::6]])
json.dump(out,open('flt_eis.json','w'))
