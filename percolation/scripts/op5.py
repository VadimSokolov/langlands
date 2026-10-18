import sys
src=open(sys.argv[1]).read().split("print('check'")[0]
exec(src)
from mpmath import invertlaplace, cos, nstr, log, quad
part=sys.argv[2]
if part=='m':
    mp.dps=40
    exec(open(sys.argv[3]).read().split("mp.dps=45\ng=")[0].split("from mpmath import invertlaplace, cos, nstr, log\n")[1])
    g=lambda s: -dphi(s)/phi(s)
    xs=[mpf(k)/50 for k in range(1,11)]+[mpf(k)/10 for k in range(3,61,2)]
    ms=[invertlaplace(g,x,method='dehoog') for x in xs]
    print('min m', nstr(min(ms),8), 'monotone', all(ms[i+1]<ms[i] for i in range(len(ms)-1)))
    # x*m'(x) not needed; check log-convexity hint
    open('m_grid.txt','w').write('\n'.join(f"{float(x)} {float(m)}" for x,m in zip(xs,ms)))
if part=='W':
    mp.dps=25
    Phi=lambda u: kap*exp(u)*eta_i(exp(u))**4
    Mom=lambda y: quad(lambda x: exp(y*x)*Phi(x),[-7,-3,-1,0,1,3,7])
    h=mpf(1)/25; ys=[k*h for k in range(0,26*25)]
    R=[1/Mom(y) for y in ys]
    def pW(x): return (R[0]/2+sum(R[k]*cos(x*ys[k]) for k in range(1,len(ys))))*h/pi
    vals=[]
    for x in [0,0.25,0.5,1,1.5,2,2.5,3,3.5,4,5,6]:
        v=pW(mpf(x)); vals.append((x,v)); print('x',x,nstr(v,12))
    for i in range(1,len(vals)):
        (x0,v0),(x1,v1)=vals[i-1],vals[i]
        if x0>=2: print('rate',x0,x1, nstr(-log(v1/v0)/(x1-x0),8))
    xs=[mpf(k)/20 for k in range(0,161)]
    pv=[pW(x) for x in xs]
    print('min pW on [0,8]', nstr(min(pv),6), 'mass', nstr(2*sum(pv)*mpf(1)/20 - pv[0]/20,8))
    open('pW_grid.txt','w').write('\n'.join(f"{float(x)} {float(v)}" for x,v in zip(xs,pv)))
