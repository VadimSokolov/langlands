import mpmath as mp, sys
mp.mp.dps=110
NT=22; KMAX=16
def test_coeffs(c):
    psi=[c[n-1]/c[n] for n in range(1,len(c))]
    cur=[psi[i+2]-2*psi[i+1]+psi[i] for i in range(len(psi)-2)]
    for k in range(KMAX):
        neg=[i+1 for i,v in enumerate(cur) if (-1)**k*v<0]
        if neg: return (k,neg[0])
        cur=[cur[i+1]-cur[i] for i in range(len(cur)-1)]
    return None
def coeffs(terms,a,X):
    # terms: list of (weight, n); kernel sum w n^a exp(-pi n^2 x / q)
    e=mp.mpf(-1)/4 if a==1 else mp.mpf(-3)/4
    out=[]
    for k in range(NT+1):
        f=lambda x: sum(w*mp.mpf(n)**a*mp.e**(-mp.pi*n*n*x/X) for w,n in terms)*x**e*(mp.log(x)/2)**(2*k)
        out.append(2*mp.quad(f,[1,5,20,80,mp.inf])/mp.factorial(2*k))
    return out
for q in (3,8,43,163):
    for a in (0,1):
        r1=test_coeffs(coeffs([(1,1)],a,q)); r2p=test_coeffs(coeffs([(1,1),(1,2)],a,q)); r2m=test_coeffs(coeffs([(1,1),(-1,2)],a,q))
        print(f"q={q} a={a}: single term {r1}; with +n=2 {r2p}; with -n=2 {r2m}",flush=True)
