import cypari2, numpy as np, json, math
pari=cypari2.Pari(); pari.allocatemem(3*10**9)
pari('read("../work/ss_graph.gp")')
out={}
def F_km(x,q):  # CDF of Kesten McKay for (q+1) regular, numerically
    g=np.linspace(-2*math.sqrt(q),2*math.sqrt(q),200001)
    rho=(q+1)*np.sqrt(np.clip(4*q-g*g,0,None))/(2*math.pi*((q+1)**2-g*g))
    c=np.concatenate([[0],np.cumsum((rho[1:]+rho[:-1])/2*np.diff(g))]); c/=c[-1]
    return np.interp(x,g,c)
allr=[]; alls=[]
for N in (16033,32029,64033 if False else 48049):
    if not int(pari(f"isprime({N})")) or N%12!=1:
        N=int(pari(f"x={N};while(!(isprime(x)&&x%12==1),x++);x"))
    pari(f"A=ssgraph({N},2);"); n=int(pari("#A"))
    A=np.array([[int(pari(f"A[{i},{j}]")) for j in range(1,n+1)] for i in range(1,n+1)],float)
    ev=np.sort(np.linalg.eigvalsh((A+A.T)/2))[:-1]
    u=n*F_km(ev,2); bulk=np.abs(ev)<0.8*2*math.sqrt(2)
    s=np.diff(u)[bulk[:-1]&bulk[1:]]; s/=s.mean()
    d=np.diff(ev); r=np.minimum(d[1:],d[:-1])/np.maximum(d[1:],d[:-1]); r=r[(np.abs(ev[1:-1])<0.8*2*math.sqrt(2))]
    out[N]={"h":n,"n_spacings":int(len(s)),"var_s":float(s.var()),"mean_r":float(r.mean()),"min_s":float(s.min())}
    np.save(f"ev{N}.npy",ev); allr+=list(r); alls+=list(s)
    print(N,out[N],flush=True)
np.save("spacings_iso.npy",np.array(alls)); np.save("ratios_iso.npy",np.array(allr))
out["pooled"]={"n":len(alls),"var_s":float(np.var(alls)),"mean_r":float(np.mean(allr)),"goe_mean_r":0.5307,"poisson_mean_r":0.3863,"goe_var_s_surmise":4/math.pi-1}
json.dump(out,open("spectra.json","w"),indent=1); print(out["pooled"])
