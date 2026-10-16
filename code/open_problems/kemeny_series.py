# Kemeny constant of supersingular l isogeny graphs as a class number series; random sign limit model
import cypari2, math, json, numpy as np, sys, time
from scipy import sparse
pari=cypari2.Pari(); pari.allocatemem(2*10**9)
l=int(sys.argv[1]); J=int(sys.argv[2])
t0=time.time()
def hw(d):
    h=int(pari.qfbclassno(d)); return h/2 if d==-4 else (h/3 if d==-3 else h)
def sigma1(n): return sum(l**i for i in range(n+1))
def lam(j): return sum(l**min(i,j-i) for i in range(j+1))
W={}; A=[]; const=-1.0  # j=0 term D_0 = h - 1 - M_N = -1 for N = 1 mod 12
for j in range(1,J+1):
    n=l**j; Aj=0.0; r=math.isqrt(4*n)
    for s in range(-r,r+1):
        D=s*s-4*n
        if D>=0: continue
        fa=pari.factor(-D); ps=[int(p) for p in fa[0]]; es=[int(e) for e in fa[1]]
        # divisors f with f^2 | D and D/f^2 a discriminant
        fs=[1]
        for p,e in zip(ps,es):
            fs=[f*p**k for f in fs for k in range(e//2+1)]
        for f in fs:
            d=D//(f*f)
            if d%4 not in (0,1): continue
            w=hw(d); Aj+=w
            d0=int(pari.coredisc(d))
            W[(j,d0)]=W.get((j,d0),0.0)+w
    A.append(Aj)
    const+=l**(-j)*(Aj/2-sigma1(j))
    print(j, Aj/2, sigma1(j)-lam(j)/2+(1/12 if j%2==0 else 0), round(time.time()-t0,1), flush=True)
c_full=(1+1/l)*const
# aggregate weights by fundamental discriminant: E_N = c + sum_d0 w_d0 chi_d0(N)
wd={}
for (j,d0),w in W.items(): wd[d0]=wd.get(d0,0.0)-(1+1/l)*0.5*l**(-j)*w
d0s=sorted(wd); wv=np.array([wd[d] for d in d0s])
perj={}
for (j,d0),w in W.items(): perj.setdefault(str(j),{})[str(d0)]=w
json.dump({"perj":perj,"l":l,"J":J,"c":c_full,"A":A,"d0":d0s,"w":wv.tolist(),"sum_w2":float((wv**2).sum())},open(f"kemeny_l{l}_J{J}.json","w"))
print("c",c_full,"n_d0",len(d0s),"sum w^2",(wv**2).sum(), "sum |w|",np.abs(wv).sum())
