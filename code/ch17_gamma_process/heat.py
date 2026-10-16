import numpy as np, mpmath as mp, json
mp.mp.dps=40
def res_law(eps,c):
    M=int(12/np.sqrt(eps))+5
    n=np.arange(-M,M+1); w=np.exp(-np.pi*eps*n.astype(float)**2); w/=w.sum()
    return np.array([w[(n%c)==r].sum() for r in range(c)])
out={}
for c in [5,7,12]:
    eps=np.logspace(-0.5,-3,25)/1.0
    tv=[0.5*np.abs(res_law(e,c)-1/c).sum() for e in eps]
    u=np.exp(-np.pi/(c*c*eps)); bound=2*u/(1-u)
    out[c]={'eps':eps.tolist(),'tv':tv,'bound':bound.tolist()}
    print(c,[f"{a:.2e}/{b:.2e}" for a,b in zip(tv[::6],bound[::6])])
# exact heat flow reciprocity identity check
def S(a,c,k): return sum(mp.expjpi((mp.mpf(a)*r*r+2*k*r)/c) for r in range(c))
def check(a,c,eps):
    eps=mp.mpf(eps); tau=mp.mpf(a)/c+1j*eps; kap=c*eps/a
    K=range(-40,41)
    L=mp.sqrt(tau/1j)*sum(mp.e**(-mp.pi*k*k/(c*c*eps))*S(a,c,k) for k in K)
    R=mp.sqrt(1+1j*kap)*sum(mp.e**(-mp.pi*k*k*(1+1j*kap)/(c*c*eps))*mp.conj(S(c,a,-k)) for k in K)
    return abs(L-R)/abs(L)
out['identity']=[float(check(a,c,e)) for (a,c,e) in [(2,5,0.3),(4,7,0.05),(3,8,1.0),(6,11,0.02)]]
print(out['identity'])
json.dump(out,open('heat.json','w'))
