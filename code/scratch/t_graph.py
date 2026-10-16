import cypari2, numpy as np, time
pari=cypari2.Pari(); pari.allocatemem(2*10**9)
pari('read("ss_graph.gp")')
def graph(p,l):
    pari(f"A=ssgraph({p},{l})"); n=int(pari("#A"))
    return np.array([[int(pari(f"A[{i},{j}]")) for j in range(1,n+1)] for i in range(1,n+1)])
t=time.time(); M=graph(1009,3); print(M.shape, set(M.sum(1)), np.abs(M-M.T).max(), time.time()-t)
ev=np.sort(np.linalg.eigvalsh(M))[:-1]
pari("mf=mfinit([1009,2],1); H=mfheckemat(mf,3);"); g=int(pari("#H"))
H=np.array([[float(pari(f"H[{i},{j}]")) for j in range(1,g+1)] for i in range(1,g+1)])
print("spec match", np.max(np.abs(np.sort(np.linalg.eigvals(H).real)-ev)))
t=time.time(); M=graph(5101,2); print(M.shape, time.time()-t)
t=time.time(); pari("f=x^5-x+1; P=select(p->poldisc(f)%p, primes([3,3000])); V=vector(#P,i,-polcoef(hyperellcharpoly(Mod(1,P[i])*f),3));"); print("g2",time.time()-t)
