import cypari2, numpy as np, json, math, itertools
pari=cypari2.Pari(); pari.allocatemem(4*10**9)
out={}
# 1. Wilton congruence: tau(p) = a_p(eta(z)eta(23z)) mod 23
X=300000
pari(f"T=mfcoefs(mfDelta(),{X});"); tau=[int(x) for x in pari("T")]
P=[int(p) for p in pari(f"primes([2,{X}])")]
def nroots(p): return len(pari(f"polrootsmod(x^3-x-1,{p})"))
w1={3:2,1:0,0:-1}
bad=0; cnt={2:0,0:0,-1:0}
for p in P:
    if p==23: continue
    a=w1[nroots(p)]; cnt[a]+=1
    if (tau[p]-a)%23: bad+=1
out['wilton']={'primes':len(P)-1,'violations':bad,'counts':cnt}
print('wilton',out['wilton'])
# 2. level raising 11 -> 77 mod 3
pari("E11=ellinit([0,-1,1,-10,-20]);")
res={}
for lab in ['77a1','77b1','77c1']:
    pari(f"E=ellinit(ellsearch(\"{lab}\")[2]);") if False else None
X2=20000
pari(f"mf=mfinit([77,2],0); B=mfeigenbasis(mf); nb=#B;")
nb=int(pari("nb")); print('newforms 77',nb, [int(pari(f'poldegree(mffields(mf)[{i}])')) for i in range(1,nb+1)])
P2=[int(p) for p in pari(f"primes([2,{X2}])")]
a11=[int(pari(f"ellap(E11,{p})")) for p in P2]
for i in range(1,nb+1):
    if int(pari(f"poldegree(mffields(mf)[{i}])"))!=1: continue
    co=[int(x) for x in pari(f"mfcoefs(B[{i}],{X2})")]
    a=[co[p] for p in P2]
    mism=[p for p,x,y in zip(P2,a,a11) if p not in (3,7,11) and (x-y)%3]
    mism5=[p for p,x,y in zip(P2,a,a11) if p not in (5,7,11) and (x-y)%5]
    res[i]={'a2':a[0],'a3':a[1],'a5':a[2],'a7':a[3],'mod3_mismatch':len(mism),'first':mism[:5],'n':len(P2)-3,'mod5_mismatch':len(mism5)}
    print(i,res[i])
out['level77']=res
out['a11']={'a2':a11[0],'a3':a11[1],'a7':a11[3]}
# 3. dimensions and supersingular counts (mass formula) for small primes
dims={N:int(pari(f"mfdim([{N},2],1)")) for N in [2,3,5,7,11,13,17,19,23,29,31,37,41,43]}
out['dims']=dims; print(dims)
# 4. Hurwitz quaternions of norm n vs 24*sigma_odd(n)
M=60
cnt=[0]*(M+1)
r=int(math.isqrt(4*M))+2
for A in range(-r,r+1):
    for Bq in range(-r,r+1):
        if (A-Bq)%2: continue
        for C in range(-r,r+1):
            if (A-C)%2: continue
            s3=A*A+Bq*Bq+C*C
            if s3>4*M: continue
            for D in range(-r,r+1):
                if (A-D)%2: continue
                s=s3+D*D
                if s<=4*M and s%4==0: cnt[s//4]+=1
pred=[0]+[24*sum(d for d in range(1,n+1) if n%d==0 and d%2) for n in range(1,M+1)]
out['hurwitz']={'max_n':M,'all_match':cnt[1:]==pred[1:],'first':cnt[1:9]}
print('hurwitz',out['hurwitz'])
# 5. Jacobi four squares check too (level 4, S_2(Gamma0(4))=0)
json.dump(out,open('flt.json','w'))
