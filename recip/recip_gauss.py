import numpy as np, mpmath as mp, json
mp.mp.dps=30
def S(a,c):  # sum_{n mod |c|} exp(pi i a n^2 / c)
    return sum(mp.expjpi(mp.mpf(a)*n*n/c) for n in range(abs(c)))
out={}
# Gauss sum reciprocity S(a,c)=sqrt(c/a) e^{i pi/4} S(-c,a), ac even
err=0; cnt=0
for a in range(1,41):
    for c in range(1,41):
        if (a*c)%2: continue
        e=abs(S(a,c)-mp.sqrt(mp.mpf(c)/a)*mp.expjpi(0.25)*S(-c,a)); err=max(err,e); cnt+=1
out['LS_pairs']=cnt; out['LS_maxerr']=float(err); print('LS',cnt,err)
# Brownian limit sqrt(eps) theta(a/c + i eps) -> S(a,c)/c
def theta(tau,M=None):
    eps=float(mp.im(tau)); M=M or int(8/np.sqrt(eps))+10
    n=np.arange(-M,M+1); return complex(np.sum(np.exp(1j*np.pi*n*n*complex(tau))))
traj={}
for m in [5,7,11,13]:
    eps=np.logspace(0,-5,120)
    z=[np.sqrt(e)*theta(mp.mpc(2.0/m,e)) for e in eps]
    traj[m]={'re':[w.real for w in z],'im':[w.imag for w in z],'lim':[float(mp.re(S(2,m)/m)),float(mp.im(S(2,m)/m))]}
    print(m,z[-1],complex(S(2,m)/m))
out['traj']=traj
# Fejer route: Theta_N(x) -> sum_k e^{-x k^2}
x=20.0; exact=float(mp.jtheta(3,0,mp.e**(-x)))
fe=[]
for N in [1,2,3,5,10,20,50]:
    n=np.arange(-N+1,N); fe.append(float(np.sqrt(np.pi/x)*np.sum((1-abs(n)/N)*np.exp(-np.pi**2*n**2/x))))
out['fejer']={'x':x,'exact':exact,'N':[1,2,3,5,10,20,50],'vals':fe}; print('fejer',exact,fe)
# QR check via Gauss sums for primes < 200
import cypari2; pari=cypari2.Pari()
ps=[int(p) for p in pari('primes([3,2000])')]; bad=0
for p in ps:
    for q in ps:
        if p<q and int(pari(f'kronecker({p},{q})*kronecker({q},{p})'))!=(-1)**((p-1)*(q-1)//4): bad+=1
out['QR_bad']=bad
json.dump(out,open('recip_gauss.json','w'))
