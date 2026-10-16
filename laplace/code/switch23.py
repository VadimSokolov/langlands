# The Laplace switch for rho_23 (x^3-x-1) <-> f = eta(z)eta(23z)
import cypari2, numpy as np, json, math
pari=cypari2.Pari(); pari.allocatemem(4*10**9); pari.set_real_precision(38)
out={}
X=200000
P=[int(p) for p in pari(f"primes([2,{X}])")]
# Galois side: Frobenius data from the splitting of primes in K=Q(x^3-x-1); a(p^m)=sum_{f|m} f - 1
pari("K=nfinit(x^3-x-1);")
gal={}
for p in P:
    fs=[int(pr[3]) for pr in pari(f"idealprimedec(K,{p})")]
    m=1
    while p**m<=X:
        gal[p**m]=(sum(f for f in fs if m%f==0)-1, p, m); m+=1
# Automorphic side: q expansion of eta(z)eta(23z)=q prod(1-q^n)(1-q^{23n}) by Euler's pentagonal series
N=X
e=np.zeros(N+1,dtype=np.int64); k=0
while True:
    done=True
    for kk in ([k,-k] if k else [0]):
        g=kk*(3*kk-1)//2
        if g<=N: e[g]+= (-1)**abs(kk); done=False
    if done and k>0: break
    k+=1
a=np.zeros(N+2,dtype=np.int64)
nz=np.nonzero(e)[0]
for j in nz:
    if 23*j>N: break
    a[1+23*j:N+1]+=e[j]*e[:N-23*j]
# a[n] coefficient of q^n of q*eta(q)eta(q^23)/q^{...}: f=q*sum e_i q^i * sum e_j q^{23 j}
a=np.roll(a,0)
coef=lambda n: int(a[n])
chi=lambda p: int(pari(f"kronecker(-23,{p})"))
aut={}
mism=0
for p in P:
    ap=coef(p); c=chi(p)
    # power sums alpha^m+beta^m with alpha+beta=a_p, alpha*beta=chi(p)
    s_prev,s=2,ap
    m=1
    while p**m<=X:
        val = s if p!=23 else 1
        if p==23: val=1  # a_23=1: alpha=1, beta=0
        aut[p**m]=val
        s_prev,s=s,ap*s-c*s_prev
        m+=1
    if gal[p][0]!=ap: mism+=1
pp=sorted(gal)
mism_pp=sum(1 for n in pp if gal[n][0]!=aut[n])
out['primes']=len(P); out['prime_mismatch']=mism; out['primepower_mismatch']=mism_pp; out['a23']=coef(23)
print('primes',len(P),'mismatch a_p',mism,'prime powers',len(pp),'mismatch',mism_pp,'a_23',coef(23))
# frequencies of Frobenius labels
lab={}
for p in P:
    if p!=23: lab[gal[p][0]]=lab.get(gal[p][0],0)+1
out['labels']={str(k):v/(len(P)-1) for k,v in lab.items()}
# Thorin density at alpha=1.5 from both sides: 2 pi r U(r^2) = sum c_n (1-cos(r log n)), c_n = Lambda(n) n^{-alpha}
alpha=1.5
def cvec(D):
    L=np.array([math.log(n) for n in pp]); C=np.array([D(n)*math.log(gal[n][1])/n**alpha for n in pp]); return L,C
Lg,Cg=cvec(lambda n: gal[n][0]); La,Ca=cvec(lambda n: aut[n])
rs=np.linspace(0.02,40,2000)
gG=np.array([np.sum(Cg*(1-np.cos(r*Lg))) for r in rs]); gA=np.array([np.sum(Ca*(1-np.cos(r*La))) for r in rs])
UG=gG/(2*np.pi*rs**2) ; UA=gA/(2*np.pi*rs**2)   # U(z) at z=r^2: g/(2 pi sqrt z)= g/(2 pi r); store density in r variable later
out['U_maxdiff']=float(np.max(np.abs(gG-gA)))
print('Thorin side max diff',out['U_maxdiff'])
# fake form: change a_59 from its value to a different value in {2,0,-1}
ap59=gal[59][0]; fake=0 if ap59!=0 else 2
def fakeD(n):
    p,m=gal[n][1],gal[n][2]
    if p!=59: return aut[n]
    # recompute power sums with fake a_59, chi(59)
    c=chi(59); s_prev,s=2,fake
    for _ in range(m-1): s_prev,s=s,fake*s-c*s_prev
    return s
Lf,Cf=cvec(fakeD)
gF=np.array([np.sum(Cf*(1-np.cos(r*Lf))) for r in rs])
out['a59']=ap59; out['fake59']=fake
out['rs']=rs.tolist(); out['gG']=gG.tolist(); out['gF']=gF.tolist()
json.dump(out,open('switch23_a.json','w'))
