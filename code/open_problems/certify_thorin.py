# Certified central Taylor coefficients (Arb), Patie Hausdorff test and Thorin Hankel determinants
import flint, math, json, sys, cypari2
from flint import arb, acb, ctx
pari=cypari2.Pari()
which=sys.argv[1]; ctx.prec=int(sys.argv[2]); K=int(sys.argv[3])
def fact(n): return arb(math.factorial(n))
if which=="xi":
    X=220; M=14
    def psi(x):
        s=acb(0)
        for n in range(1,M+1): s+=(-arb.pi()*n*n*x).exp()
        return s
    I=[]
    for k in range(K+1):
        f=lambda x,an_: psi(x)*(x**(arb(-3)/4))*((x.log()/2)**(2*k))
        v=acb.integral(f,1,X).real
        lX=arb(X).log()/2; b=arb.pi()
        tail=2*lX**(2*k)*(-b*X).exp()/(b-arb(2*k)/(X*arb(X).log()))
        rem=2*(-b*(M+1)**2).exp()/(1-(-b).exp())*X*lX**(2*k)
        I.append(2*(v+arb(0,1)*(tail+rem))/fact(2*k))
    c=[arb(0.5)-I[0]/8]+[(I[k-1]-I[k]/4)/2 for k in range(1,K+1)]
elif which=="delta":
    X=160; M=210
    tau=[0]+[int(t) for t in pari(f"vector({M},n,ramanujantau(n))")]
    beta=2*arb.pi()
    def F(y):
        q=(-beta*y).exp(); s=acb(0)
        for t in reversed(tau[1:]): s=(s+t)*q
        return s
    c=[]
    for k in range(K+1):
        f=lambda y,an_: F(y)*y**5*(y.log())**(2*k)
        v=acb.integral(f,1,X).real
        lX=arb(X).log(); b=beta
        # |tau(n)| <= d(n) n^{11/2} <= 2 n^6; y^5 (log y)^{2k} e^{-b y} decreasing beyond X here
        tail=2*arb(X)**5*lX**(2*k)*(-b*X).exp()/(b-5/arb(X)-arb(2*k)/(X*lX))/(1-(-b*X).exp())**8*64
        rem=2*arb(M+1)**6*(-b*(M+1)).exp()/(1-(-b).exp())**7*5040*arb(X)**6*lX**(2*k)
        c.append(2*(v+arb(0,1)*(tail+rem))/fact(2*k))
# Patie test
Psi=[c[n-1]/c[n] for n in range(1,K+1)]
cur=[Psi[i+2]-2*Psi[i+1]+Psi[i] for i in range(len(Psi)-2)]
patie={"certified_positive_through_order":None,"minima":[],"violation":None}
kmax=len(cur); ok_all=True
for k in range(kmax):
    vals=[((-1)**k)*v for v in cur]
    allpos=all(v>0 for v in vals); anyneg=[i+1 for i,v in enumerate(vals) if v<0]
    m=min(vals,key=lambda v: float(v.mid())) if vals else None
    patie["minima"].append((k,len(vals),str(m)))
    if anyneg and patie["violation"] is None: patie["violation"]=(k,anyneg[0])
    if not allpos: ok_all=False
    if ok_all: patie["certified_positive_through_order"]=k
    cur=[cur[i+1]-cur[i] for i in range(len(cur)-1)]
# Thorin Hankel: S_n from log series in w=z^2: log(c(w)/c0)=sum (-1)^{n+1} S_n w^n / n
a=[ci/c[0] for ci in c]; L=[arb(0)]*(K+1)
for n in range(1,K+1):  # log of power series: L_n = a_n - (1/n) sum_{k=1}^{n-1} k L_k a_{n-k}
    L[n]=a[n]-sum(k*L[k]*a[n-k] for k in range(1,n))/n
S=[None]+[((-1)**(n+1))*n*L[n] for n in range(1,K+1)]
def det(Mx):
    import copy
    A=[row[:] for row in Mx]; n=len(A); d=arb(1)
    for i in range(n):
        p=A[i][i]; d*=p
        for r in range(i+1,n):
            f=A[r][i]/p
            for cc in range(i,n): A[r][cc]-=f*A[i][cc]
    return d
hank={}
for shift in (1,2):
    for m in range(1,(K-shift)//2+1):
        if m>20: break
        Mx=[[S[i+j+shift] for j in range(m)] for i in range(m)]
        dv=det(Mx); hank[f"shift{shift}_m{m}"]=(str(dv), bool(dv>0))
out={"L":which,"prec":ctx.prec,"K":K,"c0":str(c[0]),"S1":str(S[1]),"S2":str(S[2]),"Psi":[str(p) for p in Psi[:4]],"patie":patie,"hankel":hank}
json.dump(out,open(f"thorin_cert_{which}_K{K}.json","w"),indent=1)
print(which,"S1",S[1]); print("patie",patie["certified_positive_through_order"],patie["violation"]); print("hankel",{k:v[1] for k,v in hank.items()})
