import numpy as np, json
from scipy.special import exp1
d=np.load('/home/claude/langlands/newrecip/data.npz'); P0=d['P']; X=1e7
S=[2,7,11,23,31,37]; keep=~np.isin(P0,S); P=P0[keep].astype(float); c1=d['c1'][keep]; l7=d['leg7'][keep]; p7=d['p7'][keep]
mu=[1/6,1/2,1/3]
out={'top':{},'law':{}}
sigs=[1.5,1.2,1.1,1.05,1.02,1.01,1.005]
for name,mask,m in [('id',c1==0,1/6),('tr',c1==1,1/2),('3c',c1==2,1/3),('qr',l7!=p7,1/2)]:
    res=[]
    for s in sigs:
        q=P**(-s); logsurv=np.cumsum(np.log1p(-q)[::-1])[::-1]  # sum_{q>=p}
        above=np.concatenate([logsurv[1:],[0.0]])               # sum_{q>p, q<=X}
        tail0=np.exp(-exp1((s-1)*np.log(X)))                     # P(no prime factor > X)
        pin=np.sum(q[mask]*np.exp(above[mask]))
        pnone=np.exp(logsurv[0])*tail0
        ptop=(m*(1-tail0)+tail0*pin)/(1-pnone)
        res.append(float(ptop))
    out['top'][name]=res; print(name,[round(r,4) for r in res],'mu',m)
# E1 law for largest prime factor in class C at sigma=1.01 and 1.002
for s in [1.05,1.01]:
    for name,mask,m in [('id',c1==0,1/6),('tr',c1==1,1/2)]:
        vs=np.linspace(0.05,3,40); ex=[]
        for v in vs:
            y=np.exp(v/(s-1)); sel=mask&(P>y)
            lp=np.sum(np.log1p(-P[sel]**(-s)))
            xX=(s-1)*np.log(X); lp+= -m*exp1(max(v,xX)) if True else 0
            ex.append(float(np.exp(lp)))
        out['law'][f'{name}_{s}']={'v':vs.tolist(),'exact':ex,'limit':np.exp(-m*exp1(vs)).tolist()}
        print(name,s,max(abs(np.array(ex)-np.exp(-m*exp1(vs)))))
out['sigs']=sigs
json.dump(out,open('hitrates.json','w'))
