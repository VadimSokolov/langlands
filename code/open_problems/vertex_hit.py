import cypari2, numpy as np, json, math, time, sys
pari=cypari2.Pari(); pari.allocatemem(3*10**9)
pari('read("../work/ss_graph.gp")')
out={}
for N in [int(x) for x in sys.argv[1:]]:
    t0=time.time(); pari(f"A=ssgraph({N},2);"); n=int(pari("#A"))
    A=np.array([[int(pari(f"A[{i},{j}]")) for j in range(1,n+1)] for i in range(1,n+1)],float)
    P=A/3; w,V=np.linalg.eig(P.T); pi=np.real(V[:,np.argmin(abs(w-1))]); pi/=pi.sum()
    Z=np.linalg.inv(np.eye(n)-P+np.outer(np.ones(n),pi))-np.outer(np.ones(n),pi)
    Et=np.diag(Z)/pi; M=(N-1)/12
    out[N]={"h":n,"M":M,"median_minus_2M":float(np.median(Et)-2*M),"mean_minus_2M":float(Et.mean()-2*M),"q10":float(np.percentile(Et,10)-2*M),"q90":float(np.percentile(Et,90)-2*M),"max_minus_2M":float(Et.max()-2*M),"pred_typical":-1.5*math.log2(N)}
    np.save(f"../work/Etau{N}.npy",Et)
    print(N,out[N],round(time.time()-t0),flush=True)
json.dump(out,open(f"../work/vertex_hit_{sys.argv[1]}.json","w"),indent=1)
