import numpy as np
from mpmath import mp, zeta, polygamma, euler, pi, mpf, quad, exp, log
from scipy.linalg import eigh_tridiagonal
mp.dps = 30

print("=== Problem 1: cumulants of log E and the Gumbel Thorin measure ===")
for n in range(2, 8):
    kappa = polygamma(n-1, 1)
    pred = (-1)**n * mp.factorial(n-1) * zeta(n)
    print(n, kappa, pred, abs(kappa-pred))
# Thorin measure U = sum_k delta_k : inverse moments give zeta
for n in (2,3,5):
    s = mp.nsum(lambda k: k**(-n), [1, mp.inf])
    print("sum k^-%d = %s  zeta = %s" % (n, s, zeta(n)))
# Gumbel representation check: -log E = gamma + sum (E_k - 1)/k
rng = np.random.default_rng(1)
N = 400000; K = 20000
Ek = rng.exponential(size=(N//40, K))
g = float(euler) + ((Ek - 1)/np.arange(1, K+1)).sum(axis=1)
print("sim cumulants of -log E (truncated series K=%d):" % K)
m = g - g.mean(); c2 = (m**2).mean(); c3 = (m**3).mean()
print(" kappa2 =", c2, " zeta(2) =", float(zeta(2)))
print(" kappa3 =", c3, " 2 zeta(3) =", float(2*zeta(3)))

print("\n=== Problem 1b: is W = G/(1-exp(-G)), G~Gamma(2), a GGC? necessary check ===")
# h(s) = -d/ds log E[exp(-sW)] must be completely monotone; check derivatives
def Wdens_moment(s, k):
    # E[W^k exp(-s W)] with G ~ Gamma(2): density g exp(-g)
    f = lambda g: (g/(1-exp(-g)))**k * exp(-s*g/(1-exp(-g))) * g*exp(-g)
    return quad(f, [0, 60])
def h(s):
    return Wdens_moment(s,1)/Wdens_moment(s,0)
ss = [mpf(x) for x in (0.1,0.5,1,2,4,8)]
hv = [h(s) for s in ss]
print(" h(s):", [float(v) for v in hv])
# finite difference derivatives sign check
for s in (0.5, 1, 2):
    s = mpf(s); d = mpf('1e-3')
    h0,h1,h2,h3,h4 = [h(s+j*d) for j in range(-2,3)]
    d1 = (h3-h1)/(2*d); d2 = (h3-2*h2+h1)/d**2; d3 = (h4-2*h3+2*h1-h0)/(2*d**3)
    print(" s=%s  h'=%.5f  h''=%.5f  h'''=%.5f" % (s, d1, d2, d3))
# cumulants of W
EW = Wdens_moment(0,1); EW2 = Wdens_moment(0,2); EW3 = Wdens_moment(0,3)
print(" E W =", EW, " 2 zeta(3) =", 2*zeta(3))
print(" Var W =", EW2-EW**2, " third cumulant =", EW3 - 3*EW*EW2 + 2*EW**3)

print("\n=== Problem 2: Apery Jacobi matrix and its spectral measure ===")
def apery_measure(N):
    n = np.arange(N, dtype=float)
    b = 34*n**3 + 51*n**2 + 27*n + 5
    a = (n[1:])**3
    lam, V = eigh_tridiagonal(b, a)
    w = V[0,:]**2
    return lam, w
for N in (10, 50, 200, 1000):
    lam, w = apery_measure(N)
    val = 6*np.sum(w/lam)
    print(" N=%5d  6*sum w/lam = %.15f   zeta(3) = %.15f  min lam = %.4f  w0=%.6f" % (N, val, float(zeta(3)), lam.min(), w[0]))
lam, w = apery_measure(1000)
print(" first eigenvalues:", lam[:6])
print(" first weights    :", w[:6])
print(" lam_k / (34 k^3) for k=5,20,100:", [lam[k]/(34*k**3) for k in (5,20,100)])
print(" log10 weights k=1..8:", np.log10(w[1:9]))
# Thorin test: is mu a Thorin measure (sum log(1+s/u) mu(du) finite)?  log moment
print(" int log(1+u) dmu  ~", np.sum(w*np.log1p(lam)))
# Carleman
print(" sum 1/a_k (k^-3) converges: ", np.sum(1/np.arange(1,1000)**3))

print("\n=== Problem 2b: zeta(2) continued fraction, Jacobi form ===")
def zeta2_measure(N):
    n = np.arange(N, dtype=float)
    b = 11*n**2 + 11*n + 3       # with sign flip x -> -x handled by plus numerators
    a = (n[1:])**2
    lam, V = eigh_tridiagonal(b, -a)  # partial numerators +k^4 : J with off diag i k^2; use -a (sign of a irrelevant to spectrum)
    return lam, V[0,:]**2
# For +k^4 numerators: 1/(b0 + a1^2/(b1 + ...)) = Stieltjes transform of a measure at z=0 after x -> -x. Equivalent: eigen of J(b, a) with a real gives 1/(b0 - a1^2/..). We need +, so use z = 0 with b -> -b:
n = np.arange(1000, dtype=float)
b = -(11*n**2 + 11*n + 3); a = n[1:]**2
lam, V = eigh_tridiagonal(b, a); w = V[0,:]**2
print(" 5*sum w/|lam| =", 5*np.sum(w/np.abs(lam)), " zeta(2) =", float(zeta(2)), " max lam =", lam.max())

print("\n=== Problem 3: 2-adic zeta_2(5) recurrence, real Jacobi matrix ===")
n = np.arange(400, dtype=float)
b = 32*(2*n+1)*(8*n**4+16*n**3+20*n**2+12*n+3)
a = 256*(n[1:])**5
lam, V = eigh_tridiagonal(b, a); w = V[0,:]**2
print(" min eigenvalue:", lam.min(), " positive definite:", lam.min()>0)
print(" diagonal dominance b_n - a_n - a_{n+1} for n=1,10,100:", [b[k]-a[k-1]-a[k] for k in (1,10,100)])
print(" sum w/lam =", np.sum(w/lam), " (real value of the formal CF; no claim about zeta_2(5))")

print("\n=== Problem 4: L(3,chi_4) as a difference of log Gamma cumulants ===")
beta3 = (zeta(3, mpf(1)/4) - zeta(3, mpf(3)/4))/64
print(" (zeta(3,1/4)-zeta(3,3/4))/64 =", beta3, "  pi^3/32 =", pi**3/32)
k3a = polygamma(2, mpf(1)/4); k3b = polygamma(2, mpf(3)/4)
print(" -(k3(logG_1/4) - k3(logG_3/4))/128 =", -(k3a-k3b)/128)
L3_12 = mp.nsum(lambda k: (1 if k%12 in (1,11) else (-1 if k%12 in (5,7) else 0))*k**-3, [1, mp.inf])
L3_12b = (zeta(3,mpf(1)/12)-zeta(3,mpf(5)/12)-zeta(3,mpf(7)/12)+zeta(3,mpf(11)/12))/12**3
print(" L(3,chi_12) direct =", L3_12, " Hurwitz form =", L3_12b)

print("\n=== Problem 5: Monte Carlo for zeta(3) with Legendre control variates ===")
from numpy.polynomial import legendre as L
rng = np.random.default_rng(7)
M = 2_000_000
U = rng.random(M); V = rng.random(M)
Y = 0.5*(-np.log(U*V))/(1-U*V)
print(" plain MC: mean=%.6f  se=%.6f  zeta(3)=%.6f" % (Y.mean(), Y.std()/np.sqrt(M), float(zeta(3))))
def shifted_P(n, x):
    return L.legval(2*x-1, [0]*n+[1])
for m in (1,2,3,4,6):
    X = np.column_stack([shifted_P(i,U)*shifted_P(j,V) for i in range(m+1) for j in range(m+1) if (i,j)!=(0,0)])
    beta_hat = np.linalg.lstsq(X, Y - Y.mean(), rcond=None)[0]
    R = Y - X@beta_hat
    print(" controls up to degree %d: mean=%.6f  se=%.6f  variance ratio=%.4f" % (m, R.mean(), R.std()/np.sqrt(M), R.var()/Y.var()))
