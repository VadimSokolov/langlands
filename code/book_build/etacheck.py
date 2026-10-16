import cypari2, mpmath as mp
pari=cypari2.Pari(); pari.allocatemem(10**9); mp.mp.dps=20
def EQ(rs, s, k):
    # f(iy) = prod eta(m i y)^{r_m}; F(y) = f(iy) e^{2 pi y}; L(s,f) = E[Q(s, 2 pi M)]
    def F(y):
        v=1.0
        for m,r in rs: v*=float(pari(f"real(eta(I*{m}*{y},1))"))**r
        return mp.mpf(v)*mp.e**(2*mp.pi*y)
    g=lambda y: F(y)*(2*mp.pi)**s*y**(s-1)*mp.e**(-2*mp.pi*y)/mp.gamma(s)
    return mp.quad(g,[0.005,0.05,0.3,1,3,15])
tests=[("Delta",[(1,24)],12,"mfDelta()",[6]),
       ("11a",[(1,2),(11,2)],2,None,[1,1.5]),
       ("eta(t)eta(23t)",[(1,1),(23,1)],1,None,[0.5,1]),
       ("eta(t)^4eta(5t)^4",[(1,4),(5,4)],4,None,[2])]
for name,rs,k,_,ss in tests:
    N=max(m for m,r in rs)
    # build the form via its q-expansion in PARI and its L function from coefficients
    expr="*".join(f"eta('x^{m}+O('x^4000))^{r}" for m,r in rs)
    pari(f"S=Vec('x*{expr});")
    for s in ss:
        # L function via lfuncreate with known level, weight, using coefficients
        pari(f"Lf=lfuncreate([n->vector(n,i,S[i]), 0, [0,1], {k}, {N}, 1]);")
        ex=float(pari(f"lfun(Lf,{s})"))
        print(name,s,float(EQ(rs,s,k)),ex, float(pari("lfuncheckfeq(Lf)")))
