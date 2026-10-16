import cypari2, numpy as np
pari = cypari2.Pari(); pari.allocatemem(10**9)
E = pari.ellinit([0,0,0,-1,0]); print(pari.ellglobalred(E)[0])
L2 = pari.lfunsympow(E, 2)
print(pari.lfunparams(L2) if hasattr(pari,'lfunparams') else "")
s = 6
val = float(pari.lfun(L2, s))
# Euler product over p != 2 up to 10^5 for L(s, Sym^2 E) motivic: local factor det(1 - Sym^2(Frob) p^{-s})^{-1}, eigenvalues alpha^2, alpha beta = p, beta^2
ps = [int(p) for p in pari.primes(10**4) if int(p) > 2]
prod = 1.0
for p in ps:
    ap = int(pari.ellap(E, p))
    # alpha^2 + beta^2 = ap^2 - 2p ; alpha^2 beta^2 = p^2 ; so Sym2 char poly: (1 - ap^2 X + ... ) : (1-a^2X)(1-pX)(1-b^2X) with a^2+b^2 = ap^2-2p
    X = p**(-s)
    prod *= 1.0/((1 - (ap*ap - 2*p)*X + p*p*X*X)*(1 - p*X))
print("lfun", val, "euler p>2", prod, "ratio", val/prod)
for name, f in [("1",1.0),("(1-2^-s)^-1",1/(1-2.**-s)),("(1+2^-s)^-1",1/(1+2.**-s)),("(1-2^(1-s))^-1",1/(1-2.**(1-s))),("(1+2^(1-s))^-1",1/(1+2.**(1-s)))]:
    print(name, f)
