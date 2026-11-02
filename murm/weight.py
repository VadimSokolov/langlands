# Weight aspect murmurations for level one: Eichler Selberg trace formula with Hurwitz class numbers.
import numpy as np, cypari2, json, sys
pari=cypari2.Pari()
kmin,kmax,ymax=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
P=int(ymax*kmax**2)
primes=np.array([int(x) for x in pari.primes(int(pari.primepi(P)))])
M=4*P
H=np.zeros(M+1)
for m in range(3,M+1):
    if m%4 in (0,3): H[m]=float(pari.qfbhclassno(m))
ks=np.arange(kmin,kmax,2)
def trace_norm(p,ks):
    # Tr T_p on S_k(1) divided by p^((k-1)/2)
    T=np.arange(-int(np.floor(2*np.sqrt(p))),int(np.floor(2*np.sqrt(p)))+1)
    T=T[T*T<4*p]
    th=np.arccos(T/(2*np.sqrt(p)))
    U=np.sin(np.outer(ks-1,th))/np.sin(th)          # P_k(t,p)/p^((k-2)/2)
    tr=-0.5*(U@H[4*p-T*T])/np.sqrt(p) - p**(-(ks-1)/2.0)
    return tr
# check: Delta, tau(2)=-24, tau(3)=252
for p,tau in [(2,-24),(3,252),(5,4830)]:
    print(p, trace_norm(p,np.array([12]))*p**5.5, tau)
dims=np.array([int(pari.mfdim([1,int(k)],1)) for k in ks])
eps=np.where((ks//2)%2==0,1,-1)
rows=[]
for p in primes:
    keep=(p/ks**2<=ymax)&(p<ks**2*ymax)
    tr=trace_norm(int(p),ks)
    rows.append(tr)
R=np.array(rows)  # primes x weights
np.savez(f"w_{kmin}_{kmax}.npz",R=R,ks=ks,dims=dims,eps=eps,p=primes)
print("weights",len(ks),"primes",len(primes))
