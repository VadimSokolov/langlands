import cypari2, json, math
pari=cypari2.Pari(); pari.allocatemem(4*10**9)
pari('read("../work/ss_graph.gp")')
Ns=[int(p) for p in pari("select(p->p%12==1,primes([37,1300]))")]
out=[]
for l in [2,3,5,7]:
    for N in Ns:
        pari(f"A=ssgraph({N},{l}); n=#A; Lr=matrix(n-1,n-1,i,j,if(i==j,{l+1}-A[i,j],-A[i,j])); D=matsnf(Lr);")
        D=[int(x) for x in pari("D") if int(x)!=1]
        h=(N-1)//12
        pari(f"MS=msinit({N},2,1); SS=mscuspidal(MS); TT=mshecke(MS,{l},SS); cp=charpoly(TT);")
        J=int(pari(f"subst(cp,'x,{l}+1)"))
        # repeated eigenvalue structure mod p: cyclicity requires char poly of T_l to have no repeated factor (x-(l+1)) mod p beyond multiplicity one? record gcd data
        out.append({'N':N,'l':l,'h':h,'snf':D,'J0':J,'tau':math.prod(D) if D else 1})
    print(l,flush=True)
json.dump(out,open('sandpile.json','w'))
nc=[r for r in out if len(r['snf'])>1]
print('cases',len(out),'noncyclic',len(nc))
for r in nc[:15]: print(r['N'],r['l'],r['snf'],'J0/h',r['J0']//r['h'],r['J0']%r['h'])
print('all h | J0:',all(r['J0']%r['h']==0 for r in out),'tau ok', all(r['tau']*r['h']==r['J0'] for r in out))
