"""Hecke angles of Gaussian primes: equidistribution of 4*theta_p mod 2pi for p = 1 mod 4,
and the CM curve y^2 = x^3 - x (32a2): a_p against 2a for the primary generator.
Writes gl1_data.npz with p, a, b, theta, ap."""
import numpy as np, cypari2, json, sys
pari = cypari2.Pari(); pari.allocatemem(2*10**9)
X = int(sys.argv[1]) if len(sys.argv) > 1 else 10**7
# sieve
s = np.ones(X+1, bool); s[:2] = False
for i in range(2, int(X**0.5)+1):
    if s[i]: s[i*i::i] = False
P = np.nonzero(s)[0]
P1 = P[P % 4 == 1]
# two squares via cornacchia (PARI qfbcornacchia) is slow in a loop; use sum of squares by Hermite Serret via sqrt(-1) mod p
# faster: brute: use numpy on x = sqrt(-1) mod p via pow with Euler criterion for a nonresidue
def two_squares(p):
    # find c with c^2 = -1 mod p
    for c in range(2, 200):
        r = pow(c, (p-1)//4, p)
        if (r*r) % p == p-1: break
    a, b = p, r
    while b*b > p:
        a, b = b, a % b
    return b, int(round((p - b*b)**0.5))
A = np.empty(len(P1), np.int64); B = np.empty(len(P1), np.int64)
for i, p in enumerate(P1):
    a, b = two_squares(int(p)); A[i] = a; B[i] = b
assert np.all(A*A + B*B == P1)
# primary normalisation: pi = a + b i with a odd, b even and a + b = 1 mod 4 (Hecke's primary choice)
odd = A % 2 == 1
a = np.where(odd, A, B).astype(np.int64); b = np.where(odd, B, A).astype(np.int64)
# choose signs: a odd; make a + b = 1 mod 4 using b -> -b and a -> -a
# options: (a,b),(a,-b),(-a,b),(-a,-b); pick the one with (a+b) % 4 == 1
cand = np.stack([a, b, a, -b, -a, b, -a, -b]).reshape(4, 2, -1)
ok = (cand[:, 0, :] + cand[:, 1, :]) % 4 == 1
idx = np.argmax(ok, axis=0)
a = cand[idx, 0, np.arange(len(P1))]; b = cand[idx, 1, np.arange(len(P1))]
assert np.all((a + b) % 4 == 1) and np.all(a % 2 == 1)
theta = np.arctan2(b, a)
# CM curve 32a2: y^2 = x^3 - x ; a_p via PARI
E = pari.ellinit([0, 0, 0, -1, 0])
ap = np.array([int(pari.ellap(E, int(p))) for p in P1], np.int64)
# test identities
match = np.mean(ap == 2*a)
print("primes", len(P), "p=1 mod 4", len(P1), "a_p = 2a (primary a+b=1 mod 4):", match)
# alternative normalisation used in Ireland Rosen: a odd, b even, a + b i = 1 mod (2+2i) i.e. a = 1 mod 4 if b = 0 mod 4 ... test a few
for name, cond in [("a=1 mod 4", (a % 4 == 1)), ("a=1 mod 4 and b=0 mod 4", (a % 4 == 1) & (b % 4 == 0))]:
    pass
# equidistribution of 4 theta mod 2pi
phi = (4*theta) % (2*np.pi)
# discrepancy against uniform
u = np.sort(phi/(2*np.pi)); n = len(u)
D = max(np.max(np.arange(1, n+1)/n - u), np.max(u - np.arange(n)/n))
print("discrepancy of 4theta/2pi:", D, "sqrt(n)D", D*np.sqrt(n))
# Weyl sums
for k in range(1, 6):
    print("k", k, "|mean e^{4ik theta}|", abs(np.mean(np.exp(4j*k*theta))))
# angle law of E: theta_E = arg(pi) for split p, and pi/2 for inert p; moments of cos
cosE = np.concatenate([ap/(2*np.sqrt(P1)), np.zeros(np.sum(P % 4 == 3))])
print("E[cos^2]", np.mean(cosE**2), "E[cos^4]", np.mean(cosE**4), "(CM prediction 1/4, 3/16)")
np.savez("gl1_data.npz", P=P, P1=P1, a=a, b=b, theta=theta, ap=ap)
json.dump(dict(X=X, nP=int(len(P)), nP1=int(len(P1)), match=float(match), D=float(D), sqrtnD=float(D*np.sqrt(n)),
               cos2=float(np.mean(cosE**2)), cos4=float(np.mean(cosE**4))), open("g1.json", "w"), indent=1)
