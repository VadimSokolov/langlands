from mpmath import mp, mpf, gamma, exp, pi, log, sqrt, qp, quad, inf, cos, diff, findroot, mpc, nstr, sin, polyroots
import sys
mp.dps=60
kap = pi*mpf(2)**(mpf(4)/3)*gamma(mpf(2)/3)/gamma(mpf(1)/3)**2
def eta0(t): return exp(-pi*t/12)*qp(exp(-2*pi*t))
def eta_i(t): return eta0(t) if t>=1 else eta0(1/t)/sqrt(t)
f=lambda t: kap*eta_i(t)**4
N=1500
c=[0]*(N+1); c[0]=1
for n in range(1,N//6+1):
    for _ in range(4):
        for k in range(N,6*n-1,-1): c[k]-=c[k-6*n]
a={k+1:c[k] for k in range(N) if c[k]!=0}
part=sys.argv[1]
if part=='hcm':
    mp.dps=40
    g=lambda w: (lambda v: v**2*f(v)**2)((w+sqrt(w*w-4))/2)
    for w in [mpf('2.2'),mpf(3)]:
        print(w,[nstr((-1)**k*diff(g,w,k),8) for k in range(1,7)])
        print(w,[nstr((-1)**k*diff(g,w,k,h=mpf('1e-6')),8) for k in range(1,7)])
if part=='phi':
    # Laplace transform continuation
    from functools import lru_cache
    nodes=None
    def I01(s): return quad(lambda t: exp(-s*t)*f(t),[0,mpf('0.25'),mpf('0.5'),1])
    def phi(s): return I01(s)+kap*sum(an*exp(-(s+pi*n/3))/(s+pi*n/3) for n,an in a.items())
    print('phi(0)',nstr(phi(mpf(0)),20),'phi(1) vs direct', nstr(phi(mpf(1)),15), nstr(quad(lambda t: exp(-t)*f(t),[0,1,5,inf]),15))
    poles=[n for n in sorted(a) if n<=400]
    print('poles n', poles[:30]); print('residue signs', [a[n] for n in poles[:30]])
    zeros=[]
    for i in range(len(poles)-1):
        lo=-pi*poles[i+1]/3; hi=-pi*poles[i]/3
        K=60
        pts=[lo+(hi-lo)*(j+mpf(1)/2)/K for j in range(K)]
        vals=[phi(p) for p in pts]
        for j in range(K-1):
            if vals[j]*vals[j+1]<0:
                z=findroot(phi,(pts[j],pts[j+1]),solver='anderson'); zeros.append((z, poles[i],poles[i+1]))
    for z in zeros: print('zero', nstr(z[0],15), 'n0=-3z/pi', nstr(-3*z[0]/pi,10), 'between poles', z[1], z[2])
    # argument principle on box Re in [-pi*B/3, 1], Im in [-H,H]
    B=mpf(poles[-1])-3; H=mpf(30)
    import mpmath
    def dlog(s): return mpmath.diff(phi,s)/phi(s)
    # count via sum of arg changes along rectangle
    def winding(path, M=400):
        tot=0; prev=phi(path(0))
        for j in range(1,M+1):
            cur=phi(path(mpf(j)/M)); tot+=mpmath.arg(cur/prev); prev=cur
        return tot/(2*pi)
    x0=-pi*B/3; x1=mpf(1)
    corners=[mpc(x1,-H),mpc(x1,H),mpc(x0,H),mpc(x0,-H)]
    W=0
    for i in range(4):
        A=corners[i];Bc=corners[(i+1)%4]
        W+=winding(lambda u: A+(Bc-A)*u, 600)
    npoles=sum(1 for n in a if pi*n/3< pi*B/3)
    nreal=sum(1 for z in zeros if z[0]>x0)
    print('winding',nstr(W,6),'= zeros - poles; poles in box',npoles,'real zeros in box',nreal)
