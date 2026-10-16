import cypari2, math
from fractions import Fraction as Fr
pari=cypari2.Pari(); pari.allocatemem(10**9)
def psi_sum(m,d0):
    # sum_{f|m} psi(f), psi(f)=f prod_{p|f}(1-chi(p)/p)
    if m==1: return Fr(1)
    fa=pari.factor(m); tot=Fr(1)
    for p,e in zip(fa[0],fa[1]):
        p=int(p);e=int(e);c=int(pari.kronecker(d0,p))
        s=Fr(1)+sum(Fr(p**i)-c*Fr(p**(i-1)) for i in range(1,e+1))
        tot*=s
    return tot
def elements(l,j,d0):
    # elements pi=(s+m sqrt d0)/2 of O_K with norm l^j and m>=1, via enumeration over factorization
    out=set()
    if d0==-4:
        K=pari("bnfinit(y^2+1)")
    else:
        K=pari("bnfinit(y^2+y+1)")
    # brute force for small, otherwise use qfbsolve on all representations
    n=4*l**j; D=-d0
    # solve s^2 + D m^2 = n: use PARI qfbsolve with flag for all solutions
    sols=pari(f"qfbsolve(Qfb(1,0,{D}),{n},3)")
    for v in sols:
        s,m=int(v[0]),int(v[1])
        if m>0: out.add((s,m))
        if m<0: out.add((s,-m)); out.add((-s,-m))
    # also include symmetric s
    out2=set()
    for s,m in out: out2.add((s,m)); out2.add((-s,m))
    return [x for x in out2 if x[0]**2+D*x[1]**2==n]
def cm_weight(l,d0,J):
    u=2 if d0==-4 else 3; tot=Fr(0)
    for j in range(1,J+1):
        Wj=sum(psi_sum(m,d0) for s,m in elements(l,j,d0))/u
        tot+=Fr(1,l**j)*Wj
    return -Fr(l+1,2*l)*tot
def c_closed(l): return Fr(l+1,l)*(-1+Fr(1,12*(l*l-1))-Fr(3*l-1,2*(l-1)**2))
import sys
for l in (2,3,5,7,11,13,17,19,23,29,31,37):
    J=60 if l<10 else 40
    w4=cm_weight(l,-4,J); w3=cm_weight(l,-3,J)
    mu=c_closed(l)+w4+w3
    print(l, l%4, l%3, float(c_closed(l)), float(w4), float(w3), "mean", float(mu), flush=True)
