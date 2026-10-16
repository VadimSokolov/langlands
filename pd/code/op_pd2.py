import cypari2, numpy as np, json, math
pari=cypari2.Pari(); pari.allocatemem(2*10**9); pari.set_real_precision(30)
d=np.load('data.npz'); P=d['P'].astype(float); c1=d['c1'].copy(); c1[P==23]=3
haar=np.array([1/6,1/2,1/3]); gam=0.5772156649015329
chi={'1':np.array([1,-1,1.]),'2':np.array([2,0,-1.])}
def cls_pow(c,m):
    if c==0: return 0
    if c==1: return 0 if m%2==0 else 1
    return 0 if m%3==0 else 2
pari("L1=lfuncreate(-23); Lr=lfundiv(lfuncreate(x^3-x-1),lfuncreate(1));")
dl={'1':float(pari("lfun(L1,1,1)/lfun(L1,1)")),'2':float(pari("lfun(Lr,1,1)/lfun(Lr,1)"))+math.log(23)/22}   # L^(23) for rho: + log23 * 23^-1/(1-23^-1)
# c_C = delta(C)(c - log23/23) + delta(C) sum_{chi!=1} chibar(C) [ -L'/L^S(1,chi) - sum_{p notin S, m>=2} chi(Frob_p^m) log p / p^m ]
c_tot=-gam-float(np.sum(np.log(P)/(P*(P-1))))
def pplog(ch):
    tot=0.0
    for c in range(3):
        Pc=P[c1==c]
        for m in range(2,40):
            v=np.sum(np.log(Pc)*Pc**(-m)); 
            if v<1e-18: break
            tot+=chi[ch][cls_pow(c,m)]*v
    return tot
pl={ch:pplog(ch) for ch in chi}
c_pred=[haar[c]*(c_tot-math.log(23)/23)+sum(haar[c]*chi[ch][c]*(-dl[ch]-pl[ch]) for ch in chi) for c in range(3)]
lx=math.log(1e7); c_data=[float(np.sum(np.log(P[c1==c])/P[c1==c])-haar[c]*lx) for c in range(3)]
print('L\'/L^(23)(1,chi)',dl); print('c_C pred',c_pred,'data',c_data)
# uniform integers: fraction with no prime factor in class C, vs Wirsing  ~ K_C (log x)^{-delta}
S=np.load('pd_samples.npz'); cnt=S['counts']
frac=[float(np.mean(cnt[:,c]==0)) for c in range(3)]
# Wirsing constant: K_C = (1/Gamma(1-delta)) * lim prod_{p<=x,p in C}(1-1/p) (log x)^{delta}; prod = exp(-sum 1/p - sum_{m>=2} 1/(m p^m)) 
Kc=[]
for c in range(3):
    Pc=P[c1==c]; lp=np.sum(np.log1p(-1/Pc)); Kc.append(math.exp(lp+gam*haar[c])*lx**haar[c]/math.gamma(1-haar[c]))
pred=[Kc[c]*lx**(-haar[c]) for c in range(3)]
print('no factor in C: data',frac,'Wirsing',pred)
json.dump({'dl':dl,'c_pred':c_pred,'c_data':c_data,'nofactor':frac,'wirsing':pred,'K':Kc},open('op_pd2.json','w'))
