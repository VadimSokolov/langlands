from mpmath import mp, mpf, gamma, exp, pi, sqrt, qp, findroot, mpc, nstr, arg, quad, inf
from mpmath.calculus.quadrature import GaussLegendre
mp.dps=45
kap = pi*mpf(2)**(mpf(4)/3)*gamma(mpf(2)/3)/gamma(mpf(1)/3)**2
def eta0(t): return exp(-pi*t/12)*qp(exp(-2*pi*t))
def eta_i(t): return eta0(t) if t>=1 else eta0(1/t)/sqrt(t)
f=lambda t: kap*eta_i(t)**4
N=1500
c=[0]*(N+1); c[0]=1
for n in range(1,N//6+1):
    for _ in range(4):
        for k in range(N,6*n-1,-1): c[k]-=c[k-6*n]
a=[(k+1,c[k]) for k in range(N) if c[k]!=0]
t0=mpf('0.2')
gl=GaussLegendre(mp)
nodes=[]
for lo,hi in [(mpf('0.005'),mpf('0.03')),(mpf('0.03'),mpf('0.07')),(mpf('0.07'),mpf('0.13')),(mpf('0.13'),t0)]:
    nodes+=gl.get_nodes(lo,hi,7,mp.prec)
nodes=[(x,w*f(x)) for x,w in nodes]
def phi(s):
    s=mp.mpmathify(s)
    I=sum(w*exp(-s*x) for x,w in nodes)
    return I+kap*sum(an*exp(-(s+pi*n/3)*t0)/(s+pi*n/3) for n,an in a)
print('check', nstr(phi(1),20), nstr(quad(lambda t: exp(-t)*f(t),[0,1,5,inf]),20))
print('check neg', nstr(phi(mpf(-0.5)),15), nstr(quad(lambda t: exp(0.5*t)*f(t),[0,1,5,20,60]),15))
NP=230
poles=[n for n,_ in a if n<=NP]
zeros=[]
for i in range(len(poles)-1):
    lo=-pi*poles[i+1]/3; hi=-pi*poles[i]/3
    K=24
    pts=[lo+(hi-lo)*(j+mpf(1)/2)/K for j in range(K)]
    vals=[phi(p) for p in pts]
    for j in range(K-1):
        if vals[j]*vals[j+1]<0:
            z=findroot(phi,(pts[j],pts[j+1]),solver='anderson'); zeros.append((z,poles[i],poles[i+1]))
for z,p,q in zeros: print('zero s=',nstr(z,12),' -3s/pi=',nstr(-3*z/pi,10),' gap',p,q)
def winding(A,B,M):
    tot=0; prev=phi(A)
    for j in range(1,M+1):
        cur=phi(A+(B-A)*mpf(j)/M); tot+=arg(cur/prev); prev=cur
    return tot/(2*pi)
x0=-pi*(NP-2)/3; H=mpf(40)
C=[mpc(1,-H),mpc(1,H),mpc(x0,H),mpc(x0,-H)]
W=sum(winding(C[i],C[(i+1)%4],1500) for i in range(4))
npol=sum(1 for n in poles if -pi*n/3>x0)
print('winding',nstr(W,6),'poles',npol,'real zeros',sum(1 for z in zeros if z[0]>x0))
