import cypari2, numpy as np, json, math, time
pari=cypari2.Pari(); pari.allocatemem(6*10**9)
X=10**6
P=np.array([int(p) for p in pari(f"primes([2,{X}])")])
def nroots(pol,p): return len(pari(f"polrootsmod({pol},{p})"))
tr={3:2,1:0,0:-1}
t=time.time()
aR=np.array([1.0 if p==23 else tr[nroots('x^3-x-1',p)] for p in P])
aR31=np.array([1.0 if p==31 else tr[nroots('x^3+x+1',p)] for p in P])  # ramified: a_31 for x^3+x+1 ; inertia fixed line, sign char of Q(sqrt(-31)) is 0 there; trace on V^I
print('frob',time.time()-t)
# eta(z)eta(23z) via pentagonal series
def pent(N,step):
    c=np.zeros(N+1); k=0
    while True:
        done=True
        for kk in ([0] if k==0 else [k,-k]):
            e=step*kk*(3*kk-1)//2
            if e<=N: c[e]+=(-1)**abs(kk); done=False
        if done and k>0: break
        k+=1
    return c
E1=pent(X,1); E23=pent(X,23)
nz=np.nonzero(E23)[0]
prod=np.zeros(X+1)
for e in nz: prod[e:]+=E23[e]*E1[:X+1-e]
aF=np.array([prod[p-1] for p in P])   # coefficient of q^p in q*prod
print('eta match (p!=23):',np.all(aF[P!=23]==aR[P!=23]), 'a_23',aF[P==23], aR[P==23])
chi=np.array([float(x) for x in pari(f"apply(p->kronecker(-23,p),primes([2,{X}]))")])
t=time.time()
pari(f"T=mfcoefs(mfDelta(),{X});"); Tl=[float(x) for x in pari("T")]; aD=np.array([Tl[p] for p in P])/P**5.5
print('tau',time.time()-t)
pari(f"E=ellan(ellinit([0,-1,1,-10,-20]),{X});"); El=[float(x) for x in pari("E")]; aE=np.array([El[p-1] for p in P])/np.sqrt(P)
aImp=aR.copy(); aImp[0]=2.0   # Frob_2 changed to identity
pairs={'rho_vs_f':(aR,aF),'rho_vs_impostor':(aR,aImp),'rho_vs_chi23':(aR,chi),
       'rho_vs_rho31':(aR,aR31),'rho_vs_Delta':(aR,aD),'Delta_vs_11a1':(aD,aE)}
grid=np.unique(np.logspace(1,6,60).astype(int))
idx=np.searchsorted(P,grid,side='right')
out={'grid':grid.tolist(),'run':{}}
for k,(a,b) in pairs.items():
    cs=np.cumsum((a-b)**2); out['run'][k]=(cs[idx-1]/idx).tolist(); print(k,out['run'][k][-1])
for k,a in {'rho':aR,'Delta':aD,'chi':chi}.items():
    print('second moment',k,np.mean(a**2))
    out['m2_'+k]=float(np.mean(a**2))
# joint law of (a_rho, a_rho31) vs Haar on S3 x S3
vals=[2,0,-1]; J=np.zeros((3,3))
m=(P!=23)&(P!=31)
for i,u in enumerate(vals):
    for j,v in enumerate(vals): J[i,j]=np.mean((aR[m]==u)&(aR31[m]==v))
out['joint']=J.tolist(); print(J)
# conditional means given p mod 23
cm={}
for name,a in {'rho':aR,'chi':chi,'Delta':aD,'rho31':aR31}.items():
    cm[name]=[float(np.mean(a[(P%23)==r])) for r in range(1,23)]
out['condmean23']=cm
print({k:np.round(v,3) for k,v in cm.items()})
json.dump(out,open('recip_law.json','w'))
