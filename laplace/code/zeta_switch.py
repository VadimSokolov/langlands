# (D) Thorin densities of Xi at basepoint 1/2+a as Cauchy smoothings of the zeros; (E) Laplace inversion of the Thorin law
import mpmath as mp, numpy as np, json, time, math
mp.mp.dps=25
out={}
t0=time.time()
K=1000
try:
    G=json.load(open('zeros1000.json'))
except Exception:
    G=[float(mp.im(mp.zetazero(k))) for k in range(1,K+1)]; json.dump(G,open('zeros1000.json','w'))
G=np.array(G); print('zeros',len(G),G[0],G[-1],time.time()-t0)
def xi_logder(s):  # xi'/xi for xi(s)=1/2 s(s-1) pi^{-s/2} Gamma(s/2) zeta(s)
    return 1/s+1/(s-1)-mp.log(mp.pi)/2+mp.digamma(s/2)/2+mp.zeta(s,derivative=1)/mp.zeta(s)
TK=G[-1]
def tailsum(a,r):
    # sum_{gamma>TK} [P_a(r-g)+P_a(r+g)] approx int_TK^inf 2a/g^2 dN(g), dN=(1/2pi)log(g/2pi)dg  (r<<TK)
    return float(mp.quad(lambda g: (a/(a*a+(r-g)**2)+a/(a*a+(r+g)**2))*mp.log(g/(2*mp.pi))/(2*mp.pi),[TK,mp.inf]))
res={}
for a in [0.5,0.2,0.05]:
    rr=np.linspace(0.05,50,1000) if a>0.1 else np.linspace(0.02,50,4000)
    lhs=np.array([float(mp.re(xi_logder(mp.mpf(0.5)+a+1j*r)))/math.pi for r in rr])   # = V_a(r^2) 2r : Thorin density in r
    rhs=np.array([np.sum(a/(a*a+(r-G)**2)+a/(a*a+(r+G)**2))/math.pi for r in rr])
    tail=np.array([tailsum(a,r)/math.pi for r in rr[::50]])
    rhs_t=rhs+np.interp(rr,rr[::50],tail)
    err=float(np.max(np.abs(lhs-rhs_t))); print('a',a,'max |V - Cauchy sum|',err,'min V',lhs.min(),'max',lhs.max())
    res[str(a)]={'r':rr.tolist(),'V':lhs.tolist(),'C':rhs_t.tolist(),'err':err,'minV':float(lhs.min())}
out['cauchy']=res
# (E) Thorin variable H at alpha=2: E exp(-sH) = exp(b sqrt s) zeta(2)/zeta(2+sqrt s), b=zeta'/zeta(2)
al=mp.mpf(2); b=mp.zeta(al,derivative=1)/mp.zeta(al)
F=lambda s: mp.exp(b*mp.sqrt(s))*mp.zeta(al)/mp.zeta(al+mp.sqrt(s))
EH=float(mp.diff(lambda x: mp.log(mp.zeta(x)),al,2)/2)
print('b',b,'E H theory',EH)
ts=np.concatenate([np.linspace(0.002,0.05,25),np.linspace(0.06,1.5,73)])
dens=[float(mp.invertlaplace(F,t,method='talbot')) for t in ts]
dens=np.array(dens)
mass=float(np.trapezoid(dens,ts)); mean=float(np.trapezoid(ts*dens,ts))
print('Talbot density: mass on [0.002,1.5]',mass,'mean',mean,'theory mean',EH,'min',dens.min())
# Post Widder: P_k(t) = (-1)^k/k! (k/t)^{k+1} F^{(k)}(k/t)
pw={}
for k in [4,16,32]:
    vals=[]
    for t in [0.05,0.1,0.2,0.4]:
        s0=mp.mpf(k)/t
        dk=mp.diff(F,s0,k)
        vals.append(float((-1)**k/mp.factorial(k)*(s0)**(k+1)*dk))
    pw[str(k)]=vals; print('Post Widder k',k,vals)
pw['talbot']=[float(mp.invertlaplace(F,t,method='talbot')) for t in [0.05,0.1,0.2,0.4]]; print('talbot',pw['talbot'])
out['H']={'t':ts.tolist(),'dens':dens.tolist(),'mass':mass,'mean':mean,'EH':EH,'b':float(b),'pw':pw}
json.dump(out,open('zeta_switch.json','w'))
