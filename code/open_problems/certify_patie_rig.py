# Rigorous certification (ball arithmetic, python-flint/Arb) of violations of the Patie (Hausdorff) test
import flint, math, json, sys, cypari2
from flint import arb, acb, ctx
pari=cypari2.Pari()
ctx.prec=int(sys.argv[2]) if len(sys.argv)>2 else 320
K=int(sys.argv[3]) if len(sys.argv)>3 else 26
which=sys.argv[1]
def fact(n): return arb(math.factorial(n))
def ball(err): return arb(0,err)  # [0 +/- err]
def coeffs_ec(ainv,N,M=150,X=150):
    E=pari.ellinit(ainv); an=[int(a) for a in pari.ellan(E,M)]
    beta=2*arb.pi()/arb(N).sqrt()
    def F(y):  # sum a_n q^n, q=exp(-beta y), Horner
        q=(-beta*y).exp(); s=acb(0)
        for a in reversed(an): s=(s+a)*q
        return s
    out=[]
    for k in range(K):
        f=lambda y,an_: F(y)*(y.log())**(2*k)
        I=acb.integral(f,1,X).real
        lX=arb(X).log(); b=beta
        tail=2*lX**(2*k)*(-b*X).exp()/(b-arb(2*k)/(X*lX))/(1-(-b*X).exp())**2
        # sup_{y>=1} |sum_{n>M} a_n e^{-b n y}| <= sum_{n>M} 2n e^{-b n} <= 2(M+1)e^{-b(M+1)}/(1-e^{-b})^2
        rem=2*(M+1)*(-b*(M+1)).exp()/(1-(-b).exp())**2*X*lX**(2*k)
        I=I+arb(0,1)*(tail+rem)
        out.append(2*I/fact(2*k))
    return out
def coeffs_dir(q,par,M=60):
    chi=[int(pari.kronecker(q if q%4==1 else -q, n)) if n%q else 0 for n in range(M+1)]
    # use the real primitive character of conductor q: kronecker(d, n) with d = q or -q fundamental
    d=q if (q%4==1) else -q
    chi=[int(pari.kronecker(d,n)) for n in range(M+1)]
    a=par; X=int(220*q/math.pi)+10
    c=arb.pi()/q; ex=arb(-1)/4 if a==1 else arb(-3)/4
    def psi(x):
        r=(-c*x).exp(); s=acb(0)
        for n in range(1,M+1):
            if chi[n]: s+=chi[n]*(n**a)*r**(n*n)
        return s
    out=[]
    for k in range(K):
        f=lambda x,an_: psi(x)*(x**ex)*((x.log()/2)**(2*k))
        I=acb.integral(f,1,X).real
        lX=arb(X).log()/2; b=arb.pi()/q
        # |psi(x)| <= sum_n n^a e^{-b n^2 x} <= 2 e^{-b x} for x >= X (X large); x^ex <= 1
        tail=2*2*lX**(2*k)*(-b*X).exp()/(b-arb(2*k)/(X*arb(X).log()))
        rem=2*(M+1)**a*(-b*(M+1)**2).exp()/(1-(-b).exp())*X*lX**(2*k)
        out.append(2*(I+arb(0,1)*(tail+rem))/fact(2*k))
    return out
def conv(c1,c2): return [sum(c1[i]*c2[k-i] for i in range(k+1)) for k in range(len(c1))]
if which=="11a1": c=coeffs_ec([0,-1,1,-10,-20],11)
elif which=="chi-3": c=coeffs_dir(3,1)
elif which=="chi-4": c=coeffs_dir(4,1)
elif which=="chi5": c=coeffs_dir(5,0)
elif which=="chi-4chi5": c=conv(coeffs_dir(4,1),coeffs_dir(5,0))
Psi=[None]+[c[n-1]/c[n] for n in range(1,len(c))]
delta=[None]+[Psi[n+2]-2*Psi[n+1]+Psi[n] for n in range(1,len(c)-2)]
res={"L":which,"prec_bits":ctx.prec,"K":K,"Psi":[str(p) for p in Psi[1:6]],"violations":[],"first":None}
nmax=len(delta)-1
for k in range(0,nmax):
    for n in range(1,nmax-k+1):
        v=sum(((-1)**(k-i))*math.comb(k,i)*delta[n+i] for i in range(k+1))*((-1)**k)
        if v<0:   # arb comparison: True only if certainly negative
            res["violations"].append((k,n,str(v)))
            if res["first"] is None: res["first"]=(k,n,str(v))
print(which,"Psi(1..3)",res["Psi"][:3]); print("first certified violation (k,n,value):",res["first"]); print("n certified",len(res["violations"]))
json.dump(res,open(f"certify_rig_{which}.json","w"),indent=1)
