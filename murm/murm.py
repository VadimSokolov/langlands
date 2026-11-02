import sys, json, cypari2, numpy as np
pari = cypari2.Pari(); pari.allocatemem(2*10**9)
lo, hi, P = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
fn = sys.argv[4]
curves=[]
for line in open(fn):
    f=line.split()
    N=int(f[0]); 
    if not (lo<=N<hi) or f[2]!='1': continue
    ai=json.loads(f[3]); r=int(f[4]); curves.append((N,ai,r))
primes=[int(p) for p in pari.primes(int(pari.primepi(P)))]
pr=np.array(primes)
eps=np.array([(-1)**c[2] for c in curves]); Ns=np.array([c[0] for c in curves])
A=np.zeros((len(curves),len(primes)),dtype=np.int32)
for i,(N,ai,r) in enumerate(curves):
    E=pari.ellinit(ai); an=pari.ellan(E,P)
    A[i]=[int(an[p-1]) for p in primes]
np.savez(sys.argv[5],A=A,eps=eps,N=Ns,p=pr)
print(len(curves),"classes",(eps==1).sum(),"even",(eps==-1).sum(),"odd")
