import sys
exec(open(sys.argv[1]).read().split("print('check'")[0])   # reuse phi setup from op3
from mpmath import invertlaplace, cos, nstr, log
def dphi(s):
    s=mp.mpmathify(s)
    I=-sum(w*x*exp(-s*x) for x,w in nodes)
    return I+kap*sum(an*(-t0*exp(-(s+pi*n/3)*t0)/(s+pi*n/3)-exp(-(s+pi*n/3)*t0)/(s+pi*n/3)**2) for n,an in a)
mp.dps=45
g=lambda s: -dphi(s)/phi(s)
print('g(1)',nstr(g(1),12), ' g(400)*sqrt(400)', nstr(g(400)*sqrt(400),10), 'sqrt(pi/3)', nstr(sqrt(pi/3),10))
xs=[mpf(v) for v in ['0.05','0.1','0.2','0.3','0.5','0.75','1','1.5','2','2.5','3','4','5','6']]
prev=None
for x in xs:
    m=invertlaplace(g,x,method='dehoog')
    print('x',nstr(x,4),'m(x)',nstr(m,12),' m*exp(pi x/3)',nstr(m*exp(pi*x/3),10), ' decreasing' if prev is None or m<prev else ' INCREASE')
    prev=m
