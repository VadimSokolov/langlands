import cypari2, mpmath as mp, json, math
pari=cypari2.Pari(); pari.allocatemem(2*10**9); pari.set_real_precision(40)
mp.mp.dps=30
out={}
xi=lambda s: s*(s-1)/2*mp.pi**(-s/2)*mp.gamma(s/2)*mp.zeta(s)
zeta_had=float(mp.diff(xi,0.5,2)/xi(0.5)/2)
for name,Lstr,c,T,d,N in [('zeta','lfuncreate(1)',0.5,400,1,1),('Delta','lfunmf(mfinit([1,12],1),mfDelta())',6,200,2,1),('rho23','lfundiv(lfuncreate(x^3-x-1),1)',0.5,200,2,23)]:
    pari(f"L={Lstr}; Li=lfuninit(L,[{c},{T+20}]);")
    if name=='zeta': had=zeta_had
    else: had=float(pari(f"real(lfunlambda(Li,{c},2)/lfunlambda(Li,{c}))"))/2
    zs=[float(x) for x in pari(f"lfunzeros(Li,{T})")]; zs=[z for z in zs if z>1e-9]
    partial=sum(1/z**2 for z in zs)
    tail=(math.log(N)+d*(math.log(T/(2*math.pi))+1))/(2*math.pi*T)
    out[name]={'hadamard':had,'zeros_partial':partial,'tail':tail,'zeros_total':partial+tail,'nzeros':len(zs),'T':T,'gamma1':zs[0]}
    print(name,had,partial+tail,partial,len(zs))
json.dump(out,open('explosion.json','w'))
