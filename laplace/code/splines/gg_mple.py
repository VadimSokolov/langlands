# Good Gaskins maximum penalized likelihood for Frobenius angles, sine (symmetric power) basis on [0,pi]
import numpy as np, json, math, cypari2
from scipy.optimize import minimize
pari=cypari2.Pari(); pari.allocatemem(2*10**9)
d=np.load('data.npz'); P=d['P']
th11=np.arccos(np.clip(d['x11'][P!=11]/2,-1,1))
ap=[int(x) for x in pari("E=ellinit([0,0,0,-1,0]); vector(primepi(10^6)-1,k,ellap(E,prime(k+1)))")]   # 32a1, skip p=2
pr=[int(x) for x in pari("vector(primepi(10^6)-1,k,prime(k+1))")]
th32=np.arccos(np.clip(np.array(ap)/(2*np.sqrt(pr)),-1,1))
J=12
def fit(th,alpha):
    S=np.array([np.sin(j*th) for j in range(1,J+1)])          # J x n
    jj=np.arange(1,J+1)**2
    def negobj(v):
        b=v/np.linalg.norm(v); g=b@S
        ll=np.mean(np.log(g*g+1e-300))
        pen=alpha*np.sum(jj*b*b)
        # gradient
        gb=(2*(S/g).mean(1))-2*alpha*jj*b
        gv=(gb-(gb@b)*b)/np.linalg.norm(v)
        return -(ll-pen), -gv
    v0=np.zeros(J); v0[0]=1
    r=minimize(negobj,v0,jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':2000})
    b=r.x/np.linalg.norm(r.x); b*=np.sign(b[0])
    return b
out={}
for name,th in [('11a1',th11),('32a1',th32)]:
    Uj=np.array([np.mean(np.sin(j*th)/np.sin(th)) for j in range(1,J+1)])   # empirical Sym^{j-1} moments S_j
    res={'n':int(len(th)),'S':Uj.tolist()}
    for alpha in [0.01,0.1,1.0]:
        b=fit(th,alpha); first=Uj/(2+alpha*(np.arange(1,J+1)**2-1))
        res[str(alpha)]={'b':b.tolist(),'first_order':first.tolist(),'energy':float(np.sum(np.arange(1,J+1)**2*b*b))}
        print(name,alpha,'b2..b6',np.round(b[1:6],5),'first order',np.round(first[1:6],5),'energy',round(res[str(alpha)]['energy'],4))
    res['hist']=np.histogram(th,bins=90,range=(0,math.pi),density=True)[0].tolist()
    print(name,'S_j j=1..7',np.round(Uj[:7],5))
    out[name]=res
json.dump(out,open('gg_mple.json','w'))
