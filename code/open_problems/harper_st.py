# Hecke (Sato Tate x Steinhaus) random multiplicative function: f(p^k) = e^{ik theta_p} U_k(cos phi_p)
import numpy as np, math, json, time
N=10**6; rng=np.random.default_rng(11); t0=time.time()
spf=np.zeros(N+1,dtype=np.int64)
s=np.ones(N+1,bool); s[:2]=False
for i in range(2,int(N**.5)+1):
    if s[i]: s[i*i::i]=False
P=np.nonzero(s)[0]; pidx=np.full(N+1,-1); pidx[P]=np.arange(len(P))
for p in P[::-1]: spf[p::p]=p
rows=[];pi=[];ks=[]
for n in range(2,N+1):
    m=n
    while m>1:
        p=spf[m]; k=0
        while m%p==0: m//=p; k+=1
        rows.append(n); pi.append(pidx[p]); ks.append(k)
rows=np.array(rows); pi=np.array(pi); ks=np.array(ks)
starts=np.r_[0,np.nonzero(np.diff(rows))[0]+1]
print("built",len(rows),round(time.time()-t0,1),flush=True)
def st(k):
    out=np.empty(0)
    while len(out)<k:
        th=rng.uniform(0,np.pi,2*k); u=rng.uniform(size=2*k); out=np.r_[out,th[u<np.sin(th)**2]]
    return out[:k]
xs=[10**3,3*10**3,10**4,3*10**4,10**5,3*10**5,10**6]; R=300; acc={x:[] for x in xs}; accS={x:[] for x in xs}
for r in range(R):
    th=rng.uniform(0,2*np.pi,len(P)); ph=st(len(P))
    U=np.sin((ks+1)*ph[pi])/np.sin(ph[pi])
    lg=np.add.reduceat(np.log(np.abs(U)),starts); ng=np.add.reduceat((U<0).astype(np.int64),starts)%2
    phs=np.add.reduceat(ks*th[pi],starts)
    f=np.zeros(N+1,complex); f[1]=1; f[2:]=(1-2*ng)*np.exp(lg+1j*phs)
    S=np.cumsum(f)
    # Steinhaus comparison with the same theta: completely multiplicative e^{i k theta}
    g=np.zeros(N+1,complex); g[1]=1; g[2:]=np.exp(1j*phs); Sg=np.cumsum(g)
    for x in xs: acc[x].append(abs(S[x])); accS[x].append(abs(Sg[x]))
    if r%50==0: print(r,round(time.time()-t0),flush=True)
out={str(x):{"hecke":float(np.mean(acc[x])/math.sqrt(x)),"steinhaus":float(np.mean(accS[x])/math.sqrt(x)),"loglog^-1/4":math.log(math.log(x))**-0.25,
             "hecke_sd_of_mean":float(np.std(acc[x])/math.sqrt(x)/math.sqrt(R))} for x in xs}
json.dump(out,open("harper_st.json","w"),indent=1)
for x in xs: print(x,out[str(x)])
