import cypari2, numpy as np, json, time
pari=cypari2.Pari(); pari.allocatemem(4*10**9)
pari('read("../work/ss_graph.gp")')
def mfdata(N,l):
    pari(f"MS=msinit({N},2,1); SS=mscuspidal(MS); TT=mshecke(MS,{l},SS); cp=charpoly(TT);")
    a=[float(x) for x in pari("real(polroots(cp))")] if int(pari("poldegree(cp)"))>0 else []
    J=int(pari(f"subst(cp,'x,{l}+1)")) if a else 1
    return np.array(a),J
out=[]
Ns=[int(p) for p in pari("select(p->p%12==1,primes([13,1300]))")]
others=[11,23,37,43,47,59,71,83,107,131]
for l in [2,3,5,7]:
    for N in Ns+[n for n in others if n%12!=1]:
        if N==l: continue
        pari(f"A=ssgraph({N},{l});")
        AA=pari("A"); n=len(AA); A=np.array([[int(AA[i,j]) for j in range(n)] for i in range(n)],float)
        P=A/(l+1); ev=np.sort(np.real(np.linalg.eigvals(P)))[::-1]
        Kg=float(np.sum(1/(1-ev[1:])))
        a,J=mfdata(N,l)
        Kmf=float(np.sum((l+1)/(l+1-a)))
        rec={'N':N,'l':l,'n':n,'g':len(a),'K_graph':Kg,'K_mf':Kmf,'J0':J}
        if N%12==1:
            Lap=(l+1)*np.eye(n)-A
            pari(f"Lr=matrix({n-1},{n-1},i,j,if(i==j,{l+1}-A[i,j],-A[i,j]))")
            tau=int(pari("matdet(Lr)")); rec['trees']=tau; rec['trees_times_n']=tau*n
            # Kirchhoff index from Laplacian pseudoinverse
            Jm=np.ones((n,n))/n; Lp=np.linalg.inv(Lap+Jm)-Jm; R=np.add.outer(np.diag(Lp),np.diag(Lp))-2*Lp
            rec['kirchhoff']=float(R[np.triu_indices(n,1)].sum()); rec['kirch_mf']=float(n*np.sum(1/(l+1-a)))
        out.append(rec)
    print(l,'done',flush=True)
json.dump(out,open('graphs.json','w'))
d=[r for r in out]
print('max K rel err',max(abs(r['K_graph']-r['K_mf'])/r['K_mf'] for r in d if r['g']>0))
print('trees exact', all(r['trees_times_n']==r['J0'] for r in d if 'trees' in r), sum('trees' in r for r in d))
print('kirchhoff max err',max(abs(r['kirchhoff']-r['kirch_mf'])/r['kirch_mf'] for r in d if 'trees' in r and r['g']>0))
for r in d:
    if r['N'] in (37,11,13,61) : print(r)
