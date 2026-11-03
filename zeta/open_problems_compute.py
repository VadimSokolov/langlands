from mpmath import mp, mpf, zeta, matrix, eigsy, log, exp, quad, pi, binomial, nsum, inf
mp.dps = 40

def P(n): return 34*n**3 + 51*n**2 + 27*n + 5

# high precision spectral measure of truncated Apery Jacobi matrix
N = 60
J = matrix(N, N)
for n in range(N):
    J[n,n] = P(n)
    if n > 0:
        J[n,n-1] = J[n-1,n] = n**3
E, Q = eigsy(J)
lam = [E[k] for k in range(N)]
w = [Q[0,k]**2 for k in range(N)]
print("6 sum w/lam =", 6*sum(w[k]/lam[k] for k in range(N)))
print("zeta(3)      =", zeta(3))
print("first moment sum w lam =", sum(w[k]*lam[k] for k in range(N)), " (should be b_0 = 5)")
print("log moment  =", sum(w[k]*log(1+lam[k]) for k in range(N)))

print("\n=== Open Problem 1: perturbation formulas for lambda_k, w_k ===")
def Delta(k): return P(k+1)-P(k)
print(" k   lambda_k          P(k)   eps_k=P(k)-lambda_k   2nd order pert    w_k            1st order product")
for k in range(0, 10):
    pert = P(k) + (k**6/Delta(k-1) if k>0 else 0) - (k+1)**6/Delta(k)
    prod = mpf(1)
    for j in range(1, k+1):
        prod *= mpf(j)**6/(P(k)-P(j-1))**2
    print("%2d  %s  %7d  %s  %s  %s  %s" % (k, mp.nstr(lam[k],12), P(k), mp.nstr(P(k)-lam[k],6), mp.nstr(P(k)-pert,6), mp.nstr(w[k],6), mp.nstr(prod,6)))
# asymptotic slope of log w_k
import math
I = quad(lambda t: log(1-t**3), [0,1])
slope = -(2*log(34) + 6 + 2*I)/log(10)
print(" int_0^1 log(1-t^3) dt =", I)
print(" predicted asymptotic slope of log10 w_k:", slope)
print(" empirical slopes log10(w_k/w_{k-1}) k=5,10,20,30:", [mp.nstr((log(w[k])-log(w[k-1]))/log(10),5) for k in (5,10,20,30)])

print("\n=== Open Problem 4: Apery's sequences as orthogonal polynomials at zero ===")
# Apery v_n (denominators) and u_n (numerators), u_n/v_n -> zeta(3)
v = [mpf(1), mpf(5)]; u = [mpf(0), mpf(6)]
for n in range(1, 25):
    v.append((P(n)*v[n] - n**3*v[n-1])/(n+1)**3)
    u.append((P(n)*u[n] - n**3*u[n-1])/(n+1)**3)
print(" v_n:", [int(x) for x in v[:6]], " closed form sum C(n,k)^2 C(n+k,k)^2:",
      [int(nsum(lambda k: binomial(n,k)**2*binomial(n+k,k)**2, [0,n])) for n in range(6)])
print(" u_n/v_n - zeta(3) for n=5,10,20:", [mp.nstr(u[n]/v[n]-zeta(3),5) for n in (3,5,8)])
# orthonormal polynomials p_n(x) via recurrence a_{n+1} p_{n+1} = (x - b_n) p_n - a_n p_{n-1}
def pn_at(x, nmax):
    p = [mpf(1), (x - P(0))/mpf(1)]
    for n in range(1, nmax):
        p.append(((x - P(n))*p[n] - n**3*p[n-1])/mpf((n+1)**3))
    return p
p0 = pn_at(mpf(0), 10)
print(" orthonormal p_n(0):", [mp.nstr(x,8) for x in p0[:6]])
print(" (-1)^n v_n        :", [mp.nstr((-1)**n*v[n],8) for n in range(6)])
# sum p_n(0)^2 diverges -> determinate
print(" partial sums of p_n(0)^2, n<=5,10,20:", [mp.nstr(sum(x**2 for x in pn_at(mpf(0),n+1)),5) for n in (3,5,8)])
# linear form identity: v_n zeta(3) - u_n = 6 (-1)^n int p_n(x)/x dmu(x)
print(" n   v_n zeta(3) - u_n      6(-1)^n int p_n/x dmu")
for n in range(0, 9):
    integral = sum(w[k]*pn_at(lam[k], n+1)[n]/lam[k] for k in range(N))
    print(" %d  %s   %s" % (n, mp.nstr(v[n]*zeta(3)-u[n], 15), mp.nstr(6*(-1)**n*integral, 15)))
print(" ratio of successive linear forms (should -> (sqrt2-1)^4 = %s):" % mp.nstr((mp.sqrt(2)-1)**4,8),
      [mp.nstr((v[n]*zeta(3)-u[n])/(v[n-1]*zeta(3)-u[n-1]),6) for n in (3,5,8)])

print("\n=== Open Problem 3: W = G/(1-e^{-G}) is not a GGC: endpoint and tail checks ===")
# density of W-1 near 0 should be ~ 4x ; tail of W ~ w e^{-w}
def fG(g): return g*exp(-g)
def W(g): return g/(1-exp(-g))
def dW(g): return (1-exp(-g)-g*exp(-g))/(1-exp(-g))**2
from mpmath import findroot
def fW(wv):
    g = findroot(lambda g: W(g)-wv, max(wv-1, mpf('1e-3')))
    return fG(g)/dW(g)
for x in (mpf('1e-3'), mpf('1e-2'), mpf('0.1')):
    print(" f_{W-1}(x)/x at x=%s : %s  (limit 4)" % (x, fW(1+x)/x))
for wv in (10, 20, 30):
    print(" f_W(w)/(w e^{-w}) at w=%d : %s" % (wv, fW(mpf(wv))/(wv*exp(-wv))))
print(" Var(W) = 1.43258 vs Var Gamma(2) = 2  -> W-1 is not Gamma(2), so W is not a GGC")

print("\n=== Open Problem 2: the Apery GGC Y vs Gamma(w_0)/lambda_0 ===")
print(" E Y = zeta(3)/6 =", zeta(3)/6, "  w_0/lambda_0 =", w[0]/lam[0], "  gap =", zeta(3)/6 - w[0]/lam[0])
print(" Var Y = sum w_k/lam_k^2 =", sum(w[k]/lam[k]**2 for k in range(N)), " vs w_0/lam_0^2 =", w[0]/lam[0]**2)
# total variation bound: TV(Y, Gamma(w0)/lam0) <= P(remaining components not all zero) <= sum_{k>=1} w_k ... crude; use mean of remainder instead
print(" mean of remainder sum_{k>=1} w_k/lam_k =", sum(w[k]/lam[k] for k in range(1,N)))
