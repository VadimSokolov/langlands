import cypari2, json
pari=cypari2.Pari(); pari.allocatemem(4*10**9); pari('read("iso.gp")')
res=[]
for N in [37,61,73,97,109,157,181,193,229,241]:
  for l in [2,3,5]:
    r=pari(f"my(A=ssgraph({N},{l}),h=#A,L=({l}+1)*matid(h)-A); [matsnf(L[2..h,2..h]), h]")
    snf=[int(x) for x in r[0] if int(x)!=1]
    # structure of J0(N)(F_l) for elliptic factors only when all newforms rational: compute via ellinit of curves of conductor N
    res.append([N,l,int(r[1]),snf]); print(N,l,int(r[1]),snf)
json.dump(res,open('snf.json','w'))
