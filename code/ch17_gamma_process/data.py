import cypari2, numpy as np, time
pari=cypari2.Pari(); pari.allocatemem(8*10**9)
X=10**7
t=time.time()
P=np.array([int(p) for p in pari(f"primes([2,{X}])")],dtype=np.int64)
print(len(P),time.time()-t)
def frob(pol):
    v=pari(f"apply(p->#polrootsmod({pol},p),primes([2,{X}]))")
    return np.array([int(x) for x in v],dtype=np.int8)
t=time.time(); r1=frob('x^3-x-1'); r2=frob('x^3+x+1'); print('frob',time.time()-t)
# class: 0 identity (3 roots), 1 transposition (1 root), 2 three-cycle (0 roots)
cls=lambda r: np.where(r==3,0,np.where(r==1,1,2)).astype(np.int8)
c1=cls(r1); c2=cls(r2)
leg7=np.array([int(x) for x in pari(f"apply(p->kronecker(7,p),primes([2,{X}]))")],dtype=np.int8)
p7=np.array([int(x) for x in pari(f"apply(p->kronecker(p,7),primes([2,{X}]))")],dtype=np.int8)
t=time.time()
Y=10**7
pari(f"E1=ellan(ellinit([0,-1,1,-10,-20]),{Y});"); pari(f"E2=ellan(ellinit([0,0,1,-1,0]),{Y});")
e1=np.array([int(x) for x in pari("vector(#E1,i,E1[i])")],dtype=np.int64)
e2=np.array([int(x) for x in pari("vector(#E2,i,E2[i])")],dtype=np.int64)
print('ell',time.time()-t)
x11=e1[P-1]/np.sqrt(P); x37=e2[P-1]/np.sqrt(P)
np.savez_compressed('../work/data.npz',P=P,c1=c1,c2=c2,leg7=leg7,p7=p7,x11=x11,x37=x37)
print([np.mean(c1==k) for k in range(3)], np.mean(leg7!=p7))
