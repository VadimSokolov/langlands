import cypari2, numpy as np, json, math
pari = cypari2.Pari(); pari.allocatemem(4*10**9)
out = {}
def U(n,x):
    t=np.arccos(np.clip(x/2,-1,1)); s=np.sin(t); return np.sin((n+1)*t)/np.where(s>1e-12,s,1e-12)
# vertical: T_2 and T_3 on S_2(Gamma_0(N)), N prime = 1 mod 12
Ns=[int(z) for z in pari("select(p->p%12==1, primes([1000,2600]))")]
pool={2:[],3:[]}; perN=[]
for N in Ns:
    pari(f"mf=mfinit([{N},2],1);")
    row={"N":N}
    for q in (2,3):
        ev=np.array([float(z.real()) for z in pari(f"polroots(charpoly(mfheckemat(mf,{q})))")])
        pool[q]+=list(ev/math.sqrt(q))
        if q==2:
            g=len(ev); h=g+1
            row.update(h=h, maxabs=float(np.max(np.abs(ev))), kemeny=float(np.sum(1/(1-ev/3))),
                       ihara_absu=[float(min(abs(r) for r in np.roots([2,-l,1]))) for l in ev][:3])
            row["iharadev"]=float(max(abs(abs(r)-1/math.sqrt(2)) for l in ev for r in np.roots([2,-l,1])))
    perN.append(row)
out["vertical"]={q:{"n":len(pool[q]),"U":[(n,float(np.mean(U(n,np.array(pool[q])))),q**(-n/2) if n%2==0 else 0) for n in (1,2,3,4,6,8)],
                    "maxabs":float(np.max(np.abs(pool[q])))} for q in (2,3)}
out["perN"]=[{k:v for k,v in r.items() if k!="ihara_absu"} for r in perN]
np.save("/home/claude/langlands/vert2.npy",np.array(pool[2])); np.save("/home/claude/langlands/vert3.npy",np.array(pool[3]))
# explicit supersingular isogeny graphs: spectrum and hitting times
pari("read(\"ss_graph.gp\")")
graphs=[]
for p in (1009,2017,3001):
    pari(f"A=ssgraph({p})"); n=int(pari("#A"))
    M=np.array([[int(pari(f"A[{i},{j}]")) for j in range(1,n+1)] for i in range(1,n+1)],dtype=float)
    ev=np.sort(np.linalg.eigvalsh(M))
    mf=np.sort([float(z.real()) for z in pari(f"polroots(charpoly(mfheckemat(mfinit([{p},2],1),2)))")])
    P=M/3; I=np.eye(n)
    # hitting times: for each target v solve (I-P) restricted
    H=np.zeros((n,n))
    for v in range(n):
        idx=[i for i in range(n) if i!=v]
        sol=np.linalg.solve((I-P)[np.ix_(idx,idx)],np.ones(n-1))
        H[idx,v]=sol
    kem_direct=float(np.mean(H.mean(0)))  # E_pi tau_v averaged over v (uniform pi), equals Kemeny
    kem_spec=float(np.sum(1/(1-ev[:-1]/3)))
    # return probability decay
    pts=[]
    for t in (10,20,40):
        r=np.mean(np.diag(np.linalg.matrix_power(P,t)))-1/n
        pts.append((t,float(r),float((2*math.sqrt(2)/3)**t)))
    graphs.append(dict(p=p,h=n,spec_match=float(np.max(np.abs(ev[:-1]-mf))),lam2=float(max(abs(ev[0]),ev[-2])),
        kem_direct=kem_direct,kem_spec=kem_spec,kem_over_h=kem_spec/n,maxhit=float(H.max()),ret=pts))
    if p==3001: np.save("/home/claude/langlands/H3001.npy",H.mean(0))
out["graphs"]=graphs
json.dump(out,open("/home/claude/langlands/stage2.json","w"),indent=1)
print(json.dumps(out["vertical"],indent=1)); 
for r in out["perN"][::6]: print(r)
for g in graphs: print(g)
