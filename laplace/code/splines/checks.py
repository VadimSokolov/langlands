import sympy as sp, numpy as np, json, math, cypari2
out={}
# (1) Weyl denominators are Dirichlet ground states of the alcove: SU(2) and USp(4)
t,u=sp.symbols('t u',real=True)
g2=sp.sin(t)
out['su2']=sp.simplify(-sp.diff(g2,t,2)/g2)
g4=(sp.cos(t)-sp.cos(u))*sp.sin(t)*sp.sin(u)
lap=sp.diff(g4,t,2)+sp.diff(g4,u,2)
out['usp4']=sp.simplify(-lap/g4)
# normalised Sato Tate density for USp(4) on alcove 0<u<t<pi: f = c g4^2, check integral and Fisher energy
c=1/sp.integrate(sp.integrate(g4**2,(u,0,t)),(t,0,sp.pi))
E=sp.integrate(sp.integrate(sp.diff(g4,t)**2+sp.diff(g4,u)**2,(u,0,t)),(t,0,sp.pi))*c
out['usp4_norm']=c; out['usp4_energy']=sp.simplify(E)
print(out)
# (2) Casimir form of the penalty for SU(2): gamma = sum b_j sin(j t): int gamma'^2 = sum j^2 b_j^2 * pi/2
# (3) Hecke GGC H_N(l) as Gamma x B spline: N=67, l=2
pari=cypari2.Pari(); pari.allocatemem(10**9)
pari("MS=msinit(67,2,1); SS=mscuspidal(MS); TT=mshecke(MS,2,SS); cp=charpoly(TT);")
a=np.array([float(x) for x in pari("real(polroots(cp))")]); c=3-a
print('atoms',c,'J0',float(pari('subst(cp,x,3)')),np.prod(c))
rng=np.random.default_rng(1); M=400000
E_=rng.exponential(size=(M,len(c)))
H1=(E_/c).sum(1)                                  # sum of Exp(c_i)
W=rng.dirichlet(np.ones(len(c)),size=M); G=rng.gamma(len(c),size=M)
H2=G*(W/c).sum(1)                                 # Gamma(k+1) x Dirichlet mean of 1/c (B spline knots 1/c_i)
ts=np.linspace(0.05,4,40)
def pdf_exact(x):  # partial fractions
    return np.prod(c)*sum(np.exp(-ci*x)/np.prod([cj-ci for cj in c if cj!=ci]) for ci in c)
# divided difference form: prod(c) t^k/k! E exp(-t D), D = sum c_i W_i (B spline knots c_i)
D=(W*c).sum(1); k=len(c)-1
pdf_dd=np.array([np.prod(c)*x**k/math.factorial(k)*np.mean(np.exp(-x*D)) for x in ts])
hist,edges=np.histogram(H2,bins=200,range=(0,4)); hist=hist/(M*(edges[1]-edges[0])); mids=(edges[1:]+edges[:-1])/2
out2={'atoms':c.tolist(),'ks':float(np.max(np.abs(np.sort(H1)[::4000]-np.sort(H2)[::4000]))),
 'dd_err':float(np.max(np.abs(pdf_dd-pdf_exact(ts)))),'hist_err':float(np.max(np.abs(hist-pdf_exact(mids)))),
 'ts':ts.tolist(),'pdf':pdf_exact(ts).tolist(),'pdf_dd':pdf_dd.tolist(),'mids':mids.tolist(),'hist':hist.tolist(),
 'EH':float(np.sum(1/c)),'mean_H2':float(H2.mean())}
print({k:v for k,v in out2.items() if not isinstance(v,list)})
json.dump({'su2':str(out['su2']),'usp4':str(out['usp4']),'usp4_energy':str(out['usp4_energy']),'hecke':out2},open('checks.json','w'))
