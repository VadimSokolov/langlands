from mpmath import mp, mpf, gamma, exp, pi, sqrt, qp, quad, factorial, matrix, det, polyroots, nstr, lu_solve, mpc
mp.dps=80
kap = pi*mpf(2)**(mpf(4)/3)*gamma(mpf(2)/3)/gamma(mpf(1)/3)**2
def eta0(t): return exp(-pi*t/12)*qp(exp(-2*pi*t))
def eta_i(t): return eta0(t) if t>=1 else eta0(1/t)/sqrt(t)
Phi=lambda u: kap*exp(u)*eta_i(exp(u))**4
M=18
# Phi even: moments over [0,inf)
mu=[2*quad(lambda x: x**(2*m)*Phi(x),[0,0.5,1,1.5,2,3,4,5,6.5]) for m in range(M+1)]
print('mass',nstr(mu[0],40))
a=[mu[m]*(-1)**m/factorial(2*m) for m in range(M+1)]
L=[mpf(0)]*(M+1)
for n in range(1,M+1):
    L[n]=a[n]-sum(k*L[k]*a[n-k] for k in range(1,n))/n
Z=[None]+[-m*L[m] for m in range(1,M+1)]
for m in range(1,M+1): print('Z',m,nstr(Z[m],25))
# Hankel determinants H_n=[Z_{i+j+1}]
for n in range(1,9):
    H=matrix(n,n)
    for i in range(n):
        for j in range(n): H[i,j]=Z[i+j+1]
    print('Delta',n,nstr(det(H),12))
# Prony with n atoms: w_i roots; moments Z_{k+1}=sum c w^{k+1}
zeros=[mpf(x) for x in "3.443343367909476878929937586 5.802689552546195901316320248 7.266467310821318522722657751 9.113424945499136957715665264 10.17441103098667470227930586 11.77437667375267836950691611".split()]
for n in [2,3,4,5,6,7,8]:
    # use moments m0..m_{2n-1} with m_k = Z_{k+1+s}, shift s to emphasise small gamma
    s=M-2*n
    mm=[Z[k+1+s] for k in range(2*n)]
    A=matrix(n,n); b=matrix(n,1)
    for i in range(n):
        for j in range(n): A[i,j]=mm[i+j]
        b[i]=-mm[i+n]
    c=lu_solve(A,b)
    coeffs=[1]+[c[n-1-k] for k in range(n)]
    ws=polyroots(coeffs,maxsteps=200,extraprec=200)
    gs=sorted([1/sqrt(w) if abs(mp.im(w))<1e-30 else 1/sqrt(w) for w in ws],key=lambda g: abs(g))
    print('Prony n',n,[nstr(g,12) for g in gs[:4]])
print('true',[nstr(g,12) for g in zeros[:4]])
