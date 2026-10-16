import mpmath as mp, json
mp.mp.dps=30
def chars():
    out={}
    for q,f in [(3,lambda r:[0,1,-1][r%3]),(4,lambda r:[0,1,0,-1][r%4]),(7,lambda r:[0,1,1,-1,1,-1,-1][r%7])]:
        out[q]=f
    # odd complex character mod 5: chi(2)=i
    lg={1:0,2:1,4:2,3:3}
    out[5]=lambda r: 0 if r%5==0 else mp.mpc(0,1)**lg[r%5]
    return out
def exit_split(r,q,t,K=60):
    # images: density of exiting at 0 and at q for BM (variance t) from r in (0,q)
    f0=sum((r+2*n*q)/mp.sqrt(2*mp.pi*t**3)*mp.e**(-(r+2*n*q)**2/(2*t)) for n in range(-K,K+1))
    fq=sum((q-r+2*n*q)/mp.sqrt(2*mp.pi*t**3)*mp.e**(-(q-r+2*n*q)**2/(2*t)) for n in range(-K,K+1))
    return f0,fq
def exit_eigen(r,q,t,K=400):
    f0=sum(k*mp.pi/q**2*mp.sin(k*mp.pi*r/q)*mp.e**(-k*k*mp.pi**2*t/(2*q*q)) for k in range(1,K))
    fq=sum(k*mp.pi/q**2*(-1)**(k+1)*mp.sin(k*mp.pi*r/q)*mp.e**(-k*k*mp.pi**2*t/(2*q*q)) for k in range(1,K))
    return f0,fq
res={}
for q,chi in chars().items():
    tau=sum(chi(r)*mp.expjpi(2*mp.mpf(r)/q) for r in range(1,q))
    errs=[]
    for t in [0.3,1.0,3.0]:
        L=sum(chi(r)*(lambda a:a[0]-a[1])(exit_split(r,q,t)) for r in range(1,q))
        E=sum(chi(r)*(lambda a:a[0]-a[1])(exit_eigen(r,q,t)) for r in range(1,q))
        img=2/mp.sqrt(2*mp.pi*t**3)*sum(chi(m)*m*mp.e**(-m*m/(2*t)) for m in range(1,200))
        eig=(-4*mp.pi*1j*tau/q**2)*sum(mp.conj(chi(j))*j*mp.e**(-2*mp.pi**2*j*j*t/q**2) for j in range(1,200))
        errs.append([float(abs(L-E)),float(abs(L-img)),float(abs(E-eig)),float(abs(L))])
    res[q]={'tau':[float(mp.re(tau)),float(mp.im(tau))],'errs':errs}
    print(q,errs)
json.dump(res,open('exitcheck.json','w'))
