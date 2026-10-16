import cypari2, numpy as np, json, math, time
from multiprocessing import Pool
def work(rng_):
    a,b=rng_
    pari=cypari2.Pari(); pari.allocatemem(2*10**9); pari.set_real_precision(19)
    pari("L=lfunmf(mfinit([1,12]),mfDelta()); Li=lfuninit(L,[6,0.01,2100]);")
    ts=np.arange(a,b,0.04); Z=np.array([float(pari(f"lfunhardy(Li,{t})")) for t in ts])
    return ts,Z
if __name__=="__main__":
    t0=time.time()
    edges=np.linspace(500,2000,9)
    with Pool(8) as P: res=P.map(work,[(edges[i],edges[i+1]) for i in range(8)])
    ts=np.concatenate([r[0] for r in res]); Z=np.concatenate([r[1] for r in res])
    idx=np.nonzero(np.sign(Z[:-1])!=np.sign(Z[1:]))[0]
    zs=ts[idx]-Z[idx]*(ts[idx+1]-ts[idx])/(Z[idx+1]-Z[idx])
    np.save("delta_hardy_zeros.npy",zs)
    out=json.load(open("stage9a.json")) if False else {}
    sel={'n': 50000, 'mean_logabs': 0.0020028136670439413, 'var_logabs': 1.5264941506710852, 'pred_var_mid': 0.9948418610898151, 'skew': -1.134658980305732, 'excess_kurtosis': 2.4932997960704046}
    fhk={'n_windows': 2000, 'mean_max': 1.1009830816151969, 'sd_max': 0.5674260935012289, 'mean_pred': 1.4654812656334482, 'mean_max_minus_pred': -0.3644981840182512, 'sd_max_minus_pred': 0.5657404314401111}
    out={"selberg":sel,"fhk":fhk,"model_variance_primes_upto_t":json.load(open("stage9a_var.json"))["model_variance"]}
    Nsm=lambda t:(t/math.pi)*(np.log(t/(2*math.pi))-1)
    u=Nsm(zs); sp=np.diff(u); sp=sp/sp.mean(); np.save("delta_spacings.npy",sp)
    count_pred=Nsm(2000)-Nsm(500)
    d=[]
    for i in range(len(u)):
        j=i+1
        while j<len(u) and u[j]-u[i]<3: d.append(u[j]-u[i]); j+=1
    d=np.array(d); dens=len(u)/(u[-1]-u[0])
    h,e=np.histogram(d,bins=np.linspace(0,3,31)); c=(e[:-1]+e[1:])/2
    R2=h/(len(u)*dens*(e[1]-e[0])); mont=1-(np.sin(np.pi*c)/(np.pi*c))**2
    from scipy.integrate import quad
    F=lambda s: quad(lambda x:32/math.pi**2*x*x*math.exp(-4*x*x/math.pi),0,s)[0]
    srt=np.sort(sp); ks=max(abs(np.searchsorted(srt,g)/len(srt)-F(g)) for g in np.linspace(0.05,3,60))
    out["zeros"]={"method":"sign changes of Hardy Z on a 0.04 grid, t in [500,2000]","n_zeros":int(len(zs)),"smooth_count_prediction":float(count_pred)}
    out["pair_correlation"]={"bins":c.tolist(),"empirical":R2.tolist(),"montgomery":mont.tolist(),"max_abs_dev":float(np.max(np.abs(R2-mont)))}
    out["spacings"]={"n":int(len(sp)),"var":float(sp.var()),"gue_surmise_var":float(quad(lambda x:x*x*32/math.pi**2*x*x*math.exp(-4*x*x/math.pi),0,10)[0]-1),"ks_vs_wigner_surmise":float(ks),"min":float(sp.min())}
    json.dump(out,open("stage9a.json","w"),indent=1); print(out["zeros"],out["pair_correlation"]["max_abs_dev"],out["spacings"],time.time()-t0)
