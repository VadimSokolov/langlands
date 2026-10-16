import cypari2, numpy as np, json, math, time, sys
pari=cypari2.Pari(); pari.allocatemem(3*10**9)
pari('read("../work/ss_graph.gp")')
l=2; out={}; t0=time.time()
def U(j,x): t=np.arccos(np.clip(x,-1,1)); return np.sin((j+1)*t)/np.sin(t)
for X in (3000,6000,12000,24000):
    Ns=[int(z) for z in pari(f"select(p->p%12==1,primes([{X//2},{X}]))")]
    rng=np.random.default_rng(X); Ns=sorted(rng.choice(Ns,12,replace=False).tolist())
    for N in Ns:
        pari(f"A=ssgraph({N},2);"); n=int(pari("#A"))
        A=np.array([[int(pari(f"A[{i},{j}]")) for j in range(1,n+1)] for i in range(1,n+1)],float)
        ev,V=np.linalg.eigh((A+A.T)/2); ev=ev[:-1]; Vn=V[:,:-1]; x=ev/(2*math.sqrt(l)); M=(N-1)/12
        F=[float(np.sum(U(j,x))-(M*l**(-j/2) if j%2==0 else 0)) for j in range(0,70)]
        # per vertex: E_pi tau_v = h * Z_vv, Z_vv = sum_i phi_i(v)^2/(1-lam_i/3)
        W2=Vn**2
        Et=n*(W2@(1/(1-ev/3)))
        # B(l^j)_vv for j<=16 via eigen: (1/h)*sigma1 + sum_i phi^2 * l^{j/2} U_j
        Bvv=[ (2**(j+1)-1)/n + W2@(l**(j/2)*U(j,x)) for j in range(0,17)]
        rstar=[np.rint(2*b-2*(j%2==0)).astype(int) for j,b in enumerate(Bvv)]
        out[N]={"X":X,"h":n,"F":F,"Et":Et.tolist(),"rstar":[r.tolist() for r in rstar]}
        print(X,N,n,round(time.time()-t0),flush=True)
json.dump(out,open("window_test.json","w"))
