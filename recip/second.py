import cypari2, numpy as np, json
from scipy.special import exp1
pari=cypari2.Pari(); pari.allocatemem(4*10**9); pari.set_real_precision(30)
d=np.load('data.npz'); Pall=d['P']; X=1e7
S=[2,7,11,23,31,37]
def LdL(Lstr):
    pari(f"L={Lstr};"); return float(pari("lfun(L,1,1)/lfun(L,1)"))
g=0.5772156649015329
gam_rho=LdL("lfundiv(lfuncreate(x^3-x-1),1)")      # L'/L(1,rho)
gam_sgn=LdL("lfuncreate(-23)")                     # L'/L(1,chi_{-23})
gam_m4=LdL("lfuncreate(-4)")
print('LdL rho',gam_rho,'chi-23',gam_sgn,'chi-4',gam_m4)
# local data at S for the S3 characters: classes of Frob_p, p in S unramified
cls_of={int(p):int(c) for p,c in zip(Pall,d['c1'])}
chiS3=np.array([[1,1,1],[1,-1,1],[2,0,-1]])  # rows trivial,sgn,rho ; cols id,transp,3cycle
def powcls(c,m):
    if c==0: return 0
    if c==1: return 0 if m%2==0 else 1
    return 0 if m%3==0 else 2
def Sterm(row):  # sum_{p in S} sum_m tr(Frob_p^m | V^I) log p p^-m
    t=0.0
    for p in S:
        for m in range(1,60):
            if p==23:
                v={0:1.0,1:0.0,2:1.0}[row]   # trivial:1, sgn ramified:0, rho: V^I one dim, eigenvalue a_23=1
            else: v=chiS3[row][powcls(cls_of[p],m)]
            t+=v*np.log(p)*p**(-m)
    return t
LS=[g+Sterm(0), gam_sgn+Sterm(1), gam_rho+Sterm(2)]   # gamma for trivial: d/ds log((s-1)zeta^S) at 1
mu=np.array([1/6,1/2,1/3])
keep=~np.isin(Pall,S); P=Pall[keep].astype(float); c1=d['c1'][keep]
def R(C):
    t=0.0
    for m in range(2,40):
        pc=np.array([powcls(c,m) for c in range(3)])
        diff=(c1==C).astype(float)-(pc[c1]==C).astype(float)
        t-=np.sum(diff*np.log(P)*P**(-m))
    return t
kappa=[]
for C in range(3):
    k=mu[C]*sum(np.conj(chiS3[r][C])*LS[r] for r in range(3))+R(C); kappa.append(float(k))
print('kappa',kappa,'sum',sum(kappa))
# Ihara Euler-Kronecker constant of splitting field: gamma_K = gamma + LdL(sgn) + 2 LdL(rho)
gK=g+gam_sgn+2*gam_rho; print('gamma_K (Ihara)',gK)
# numerical slopes from exact Euler products
def logL(sig,lam,Q,mu_vals):
    q=P**(-sig); t=P**(-(sig-1)*lam*Q)
    xX=(sig-1)*np.log(X)
    tailv=-np.sum(mu_vals[:,1]*(exp1(xX)-exp1((1+lam*mu_vals[:,0])*xX)))
    return np.sum(np.log1p(-q)-np.log1p(-q*t))+tailv
res={'kappa':kappa,'gammaK':gK,'LdL':{'rho':gam_rho,'sgn':gam_sgn,'m4':gam_m4}}
eps=np.array([0.04,0.02,0.01,0.005,0.0025,0.00125])
for C in range(3):
    Q=(c1==C).astype(float); mv=np.array([[1.,mu[C]]])
    dev=[(logL(1+e,1.0,Q,mv)+mu[C]*np.log(2))/e for e in eps]
    fit=np.polyfit(eps,dev,1)
    res[f'C{C}']={'eps':eps.tolist(),'dev_over_eps':dev,'intercept':float(fit[1])}
    print(C,'numerical slope -> ',fit[1],'predicted',kappa[C],[round(v,4) for v in dev])
# QR forgotten sign: Q=4 on p=3 mod 4 (union of classes). kappa_Q = 4*h'(1) for set {chi_-4=-1}
l7=d['leg7'][keep]; p7=d['p7'][keep]; Qq=((l7-p7)**2).astype(float)
Sq=[2,7,11,23,31,37]
# h'(1) for indicator (1-chi_-4)/2 with L^S: trivial part gamma+sum_S logp/(p-1); chi_-4 part
St0=sum(np.log(p)/(p-1) for p in Sq)
St4=sum(sum((((-1)**((p-1)//2))**m)*np.log(p)*p**(-m) for m in range(1,60)) for p in Sq if p!=2)
hprime=0.5*(g+St0)-0.5*(gam_m4+St4)
# prime power correction: indicator(p=3 mod4) - indicator(p^m=3 mod 4)
chi4=np.where(P%4==3,-1,1)
Rq=0.0
for m in range(2,40):
    diff=(chi4==-1).astype(float)-((chi4**m)==-1).astype(float); Rq-=np.sum(diff*np.log(P)*P**(-m))
kq=4*(hprime+Rq); print('QR kappa pred',kq)
devq=[(logL(1+e,1.0,Qq,np.array([[4.,.5]]))+0.5*np.log(5))/e for e in eps]
fq=np.polyfit(eps,devq,1); print('QR numerical',fq[1],[round(v,4) for v in devq])
res['qr']={'kappa':float(kq),'dev_over_eps':devq,'intercept':float(fq[1])}
json.dump(res,open('second.json','w'))
