# Open problems of Chapter 21: Mertens constants via L(1,chi); atom at zero of class shares; share means; Rankin Selberg colouring
import cypari2, numpy as np, json, math, mpmath as mp
pari=cypari2.Pari(); pari.allocatemem(4*10**9); pari.set_real_precision(30); mp.mp.dps=20
d=np.load('data.npz'); P=d['P'].astype(float); c1=d['c1'].copy(); c1[P==23]=3
haar=np.array([1/6,1/2,1/3]); gam=0.5772156649015329
# characters of S3 on classes (id, transp, 3cyc): chi0, sign, rho
chi={'0':np.array([1,1,1.]),'1':np.array([1,-1,1.]),'2':np.array([2,0,-1.])}
def cls_pow(c,m):  # class of g^m
    if c==0: return 0
    if c==1: return 0 if m%2==0 else 1
    return 0 if m%3==0 else 2
# L functions with the Euler factor at 23 removed
pari("K=nfinit(x^3-x-1); L1=lfuncreate(-23); Lr=lfundiv(lfuncreate(x^3-x-1),lfuncreate(1));")
L1_1=float(pari("lfun(L1,1)")); Lr_1=float(pari("lfun(Lr,1)"))*(1-1/23)
print('L(1,chi_-23)',L1_1,'L^(23)(1,rho)',Lr_1, '3pi/sqrt23=',3*math.pi/math.sqrt(23))
# prime power sums sum_{p!=23,m>=2} chi(Frob_p^m)/(m p^m)  (primes < 1e7 suffice)
def ppsum(ch,s=1.0):
    tot=0.0
    for c in range(3):
        Pc=P[c1==c]
        for m in range(2,40):
            v=np.sum(Pc**(-m*s))/m
            if v<1e-18: break
            tot+=chi[ch][cls_pow(c,m)]*v
    return tot
pp={ch:ppsum(ch) for ch in chi}
B=0.2614972128476428
b_pred=[]
for c in range(3):
    val=haar[c]*(B-1/23)
    for ch,L1v in [('1',L1_1),('2',Lr_1)]:
        val+=haar[c]*chi[ch][c]*(math.log(L1v)-pp[ch])
    b_pred.append(val)
ll=math.log(math.log(1e7)); b_data=[float(np.sum(1/P[c1==c])-haar[c]*ll) for c in range(3)]
print('b_C predicted',b_pred,'data',b_data)
out={'L1chi':L1_1,'L1rho23':Lr_1,'pp':pp,'b_pred':b_pred,'b_data':b_data}
# P_C(s) = sum_{p in C} sum_m p^{-ms}/m = sum_chi c_chi log L^S(s,chi) + R_C(s), S={23}
# R_C(s) = sum_{p,m>=2} [1_C(p) - 1_C(p^m)] p^{-ms}/m : precompute (n, weight) tables with n=p^m < 1e14
tabs={}
for c in range(3):
    ns=[];ws=[]
    for cc in range(3):
        Pcc=P[c1==cc]
        for m in range(2,48):
            w=(1.0 if cc==c else 0.0)-(1.0 if cls_pow(cc,m)==c else 0.0)
            if w==0: continue
            sel=Pcc**m<1e14
            if not sel.any(): break
            ns.append(Pcc[sel]**m); ws.append(np.full(sel.sum(),w/m))
    tabs[c]=(np.concatenate(ns),np.concatenate(ws))
zeta=lambda s: float(pari(f"zeta({s})")); L1f=lambda s: float(pari(f"lfun(L1,{s})")); Lrf=lambda s: float(pari(f"lfun(Lr,{s})"))
def PC(c,s):
    s=float(s)
    v=haar[c]*(math.log(zeta(s))+math.log(1-23**-s))
    v+=haar[c]*chi['1'][c]*math.log(L1f(s))
    v+=haar[c]*chi['2'][c]*(math.log(Lrf(s))+math.log(1-23**-s))
    n,w=tabs[c]; return v+float(np.sum(w*n**(-s)))
# atom at zero: P(L_C=0)=exp(-P_C(sigma)); constant h_C(1)=lim P_C(s)-delta log(1/(s-1))
hC=[PC(c,1+1e-6)-haar[c]*math.log(1/1e-6) for c in range(3)]
hC2=[PC(c,1+1e-5)-haar[c]*math.log(1/1e-5) for c in range(3)]
print('h_C(1)',hC,hC2)
atoms={}
for sig in [1.1,1.01,1.001,1.0001]:
    atoms[str(sig)]=[math.exp(-PC(c,sig)) for c in range(3)]
    pred=[math.exp(-hC[c])*(sig-1)**haar[c] for c in range(3)]
    print('sigma',sig,'P(L_C=0)',np.round(atoms[str(sig)],5),'pred',np.round(pred,5))
out['hC']=hC; out['atoms']=atoms
# share means E[L_C/L ; L>0] = int_0^inf (-phi_C'(t)) prod_{C' != C} phi_C'(t) dt,  phi_C(t)=exp(P_C(sigma+t(sigma-1))-P_C(sigma))
ptabs={}
for c in range(3):
    Pc=P[c1==c]; ns=[];ws=[]
    for m in range(1,48):
        sel=Pc**m<1e14
        if not sel.any(): break
        ns.append(Pc[sel]**m); ws.append(np.full(sel.sum(),1.0/m))
    ptabs[c]=(np.concatenate(ns),np.concatenate(ws))
def PCfull(c,s):
    if s<2.5: return PC(c,s)
    n,w=ptabs[c]; return float(np.sum(w*n**(-s)))
L1d=lambda s: float(pari(f"lfun(L1,{s},1)/lfun(L1,{s})")); Lrd=lambda s: float(pari(f"lfun(Lr,{s},1)/lfun(Lr,{s})")); zd=lambda s: float(mp.zeta(s,derivative=1)/mp.zeta(s))
def PCd(c,s):
    s=float(s)
    if s>=2.5:
        n,w=ptabs[c]; return -float(np.sum(w*np.log(n)*n**(-s)))
    e23=math.log(23)*23**-s/(1-23**-s)
    v=haar[c]*(zd(s)+e23)+haar[c]*chi['1'][c]*L1d(s)+haar[c]*chi['2'][c]*(Lrd(s)+e23)
    n,w=tabs[c]; return v-float(np.sum(w*np.log(n)*n**(-s)))
def share_means(sig):
    ts=np.concatenate([[0],np.logspace(-4,5,320)])
    base=[PC(c,sig) for c in range(3)]
    phi=np.array([[math.exp(PCfull(c,sig+t*(sig-1))-base[c]) for t in ts] for c in range(3)])
    dphi=np.array([[-(sig-1)*PCd(c,sig+t*(sig-1)) for t in ts] for c in range(3)])*phi
    res=[]
    for c in range(3):
        g=dphi[c]*np.prod([phi[cc] for cc in range(3) if cc!=c],axis=0)
        res.append(float(np.trapezoid(g,ts)))
    return res,float(np.prod(phi[:,-1]))
sm={}
for sig in [1.1,1.01,1.001]:
    r,p1=share_means(sig); sm[str(sig)]={'means':r,'P_all_zero':p1}
    print('sigma',sig,'E share',np.round(r,4),'sum',round(sum(r),4),'P(all zero)',round(p1,4),'deficit',np.round(haar*(1-p1)-np.array(r),4),'(sigma-1)^delta',np.round((sig-1)**haar,4),flush=True)
out['share_means']=sm
# Rankin Selberg colouring: weights |chi_rho(Frob_p^m)|^2 ; limit shape (2/3,0,1/3)
from scipy.special import exp1
rstab={}
for c in range(3):
    Pc=P[c1==c]; ns=[];ws=[]
    for m in range(1,48):
        w=abs(chi['2'][cls_pow(c,m)])**2/m
        if w==0: continue
        sel=Pc**m<1e14
        if not sel.any(): break
        ns.append(Pc[sel]**m); ws.append(np.full(sel.sum(),w))
    rstab[c]=(np.concatenate(ns),np.concatenate(ws))
def PRS2(c,s):
    n,w=rstab[c]; v=float(np.sum(w*n**(-s)))
    return v+abs(chi['2'][c])**2*haar[c]*exp1((s-1)*math.log(1e7))
rs={}
for sig in [1.01,1.001]:
    lam=np.array([0.5,1,2,4])
    ex=[[math.exp(PRS2(c,sig+l*(sig-1))-PRS2(c,sig)) for l in lam] for c in range(3)]
    lim=[(1+lam)**(-4/6),np.ones_like(lam),(1+lam)**(-1/3)]
    rs[str(sig)]={'lam':lam.tolist(),'exact':ex,'limit':[x.tolist() for x in lim]}
    print('RS sigma',sig); [print(' class',c,np.round(ex[c],4),np.round(lim[c],4)) for c in range(3)]
out['rs']=rs
json.dump(out,open('op_pd.json','w'))
