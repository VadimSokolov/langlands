# A Frey type curve y^2=x(x-A)(x+B) from the abc triple 3+125=128, and the Eisenstein shapes
import cypari2, numpy as np, json
pari=cypari2.Pari(); pari.allocatemem(2*10**9)
E=pari('ellinit([0,253,0,16000,0])')   # x(x+125)(x+128): A=-125, B=128, A+B=3
red=pari.ellglobalred(E); N=int(red[0])
print('conductor',N,'factor',pari.factor(N),'disc',pari.factor(E[11]) if False else pari.factor(pari('ellinit([0,253,0,16000,0]).disc')))
P=[int(p) for p in pari('primes([2,2000000])')]
a=[int(pari.ellap(E,p)) if N%p else None for p in P]
out={'N':N}
for p in [2,3,5,7,11,13]:
    L=[(ap,q) for ap,q in zip(a,P) if ap is not None and q!=p]
    out[p]=sum(((ap-1-q)%p)!=0 for ap,q in L)/len(L)
print(out); json.dump(out,open('frey_like.json','w'))
