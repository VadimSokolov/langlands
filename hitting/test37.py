import cypari2, json, itertools, math
pari=cypari2.Pari(); pari.allocatemem(2*10**9); pari('read("/home/claude/langlands/ss_graph.gp")')
pari("Ea=ellinit([0,0,1,-1,0]); Eb=ellinit([0,1,1,-23,-50]);")  # 37a1, 37b1
def oddpart(invs):
    out=[]
    for d in invs:
        while d%2==0: d//=2
        if d>1: out.append(d)
    return out
def subgroups_quot_ok(G,target):
    # G: list of invariants of an abelian group (odd), test whether some cyclic subgroup of order 3 has quotient with invariants target
    # brute force on small groups via element orders: enumerate elements as tuples
    import itertools
    els=list(itertools.product(*[range(d) for d in G]))
    def order(e):
        o=1
        for x,d in zip(e,G):
            o=o*d//math.gcd(o*d//math.gcd(x,d)//1 if x else 1, o*d) if False else o
        return o
    res=set()
    for e in els:
        # order of e
        o=1
        for x,d in zip(e,G):
            oi=d//math.gcd(x,d); o=o*oi//math.gcd(o,oi)
        if o!=3: continue
        # quotient invariants via Smith form of relation matrix: generators e_i with relations d_i e_i=0 and e=0
        k=len(G); rows=[[d if i==j else 0 for j in range(k)] for i,d in enumerate(G)]+[list(e)]
        M=pari.matrix(len(rows),k,[v for r in rows for v in r]).mattranspose()
        snf=[int(x) for x in pari.matsnf(M) if int(x)!=1 and int(x)!=0]
        res.add(tuple(sorted(oddpart(snf),reverse=True)))
    return res
out=[]
for l in [3,5,7,11,13,17,19,23,29,31,41,43,47]:
    pari(f"A=ssgraph(37,{l}); Lr=matrix(2,2,i,j,if(i==j,{l+1}-A[i,j],-A[i,j]));")
    snf=[int(x) for x in pari("matsnf(Lr)") if int(x)!=1]
    ga=[int(x) for x in pari(f"ellgroup(Ea,{l})")]; gb=[int(x) for x in pari(f"ellgroup(Eb,{l})")]
    G=oddpart(ga)+oddpart(gb)
    tgt=tuple(sorted(oddpart(snf),reverse=True))
    if 3 in [math.gcd(g,3) for g in G] and any(g%3==0 for g in G):
        quots=subgroups_quot_ok(G,tgt) if math.prod(G)<5000 else None
        ok=None if quots is None else (tgt in quots)
    else: quots=None; ok='3 does not divide'
    print(l,'sandpile',snf,'37a',ga,'37b',gb,'target odd',tgt,'possible quotients',quots,'match',ok)
    out.append({'l':l,'sandpile':snf,'E37a':ga,'E37b':gb,'match':str(ok)})
json.dump(out,open('test37.json','w'))
