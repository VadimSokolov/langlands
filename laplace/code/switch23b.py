import cypari2, numpy as np, json, math, time
pari=cypari2.Pari(); pari.allocatemem(4*10**9); pari.set_real_precision(30)
out={}
# (B) Bohr recovery of Frobenius data from the Thorin density of the modular form eta(z)eta(23z) at alpha=2.
#     L and L' are summed from the q-expansion coefficients a_n, n<=Nmax (automorphic data only).
Nmax=30000
co=np.array([float(c) for c in pari(f"Vec(lfunan(lfunetaquo([1,1;23,1]),{Nmax}))")])
n=np.arange(1,Nmax+1,dtype=float); logn=np.log(n)
alpha=2.0; T=450.0; h=0.05; R=6*T
rs=np.arange(0,R+h/2,h)
t0=time.time()
g=np.empty_like(rs)
w0=co*n**(-alpha)
for i0 in range(0,len(rs),400):
    r=rs[i0:i0+400][:,None]
    E=np.exp(-1j*r*logn[None,:])
    Ls=(E*w0).sum(1); Lp=-(E*(w0*logn)).sum(1)
    g[i0:i0+400]=np.real(Lp/Ls)
print('time',time.time()-t0)
b=g[0]; g=g-b     # g(r) = sum_n Lambda(n) n^-alpha (1-cos(r log n)) = 2 pi r U(r^2)
w=np.exp(-rs**2/(2*T**2)); w[0]*=0.5
pari("K=nfinit(x^3-x-1);")
tab=[]
for m in range(2,65):
    c=-2*np.sum(g*np.cos(rs*math.log(m))*w)/np.sum(w)
    fa=pari(f"factor({m})")
    if len(fa[0])==1:
        p=int(fa[0][0]); e=int(fa[1][0])
        fs=[int(pr[3]) for pr in pari(f"idealprimedec(K,{p})")]
        lam=(sum(f for f in fs if e%f==0)-1)*math.log(p)
    else: lam=0.0
    tab.append([m,c*m**alpha,lam])
err=max(abs(r-l) for _,r,l in tab); print('max error recovering Lambda(n), n<=64:',err)
for m,r,l in tab: 
    if l!=0 or abs(r)>1e-3: print(m,round(r,6),round(l,6), 'label' , round(r/math.log(int(pari(f'factor({m})')[0][0])),4))
out['bohr']=tab; out['bohr_err']=err; out['T']=T; out['alpha']=alpha; out['Nmax']=Nmax
out["g_r"]=rs[:1201].tolist(); out["g"]=g[:1201].tolist()
# (C) Polya kernel from the Artin L function on the critical line (Galois analytic object)
pari("La=lfundiv(lfuncreate(x^3-x-1),lfuncreate(1)); Lai=lfuninit(La,[1/2,0,61]);")
tmax=60; dt=0.005
ts=np.arange(0,tmax+dt/2,dt)
Lam=np.array([float(v) for v in pari(f"vector({len(ts)},k,real(lfunlambda(Lai,1/2+I*(k-1)*{dt})))")])
out['Lambda_imag']=float(pari("vecmax(vector(60,k,abs(imag(lfunlambda(Lai,1/2+I*k)))))"))
us=np.linspace(-3,3,121)
wt=np.ones_like(ts); wt[0]=0.5; wt[-1]=0.5
Phi_inv=np.array([np.sum(Lam*np.cos(ts*u)*wt)*dt/np.pi for u in us])
co2=co[:4000]; n2=np.arange(1,4001)
Phi_aut=np.array([2*np.sum(co2*np.exp(-2*np.pi*n2*math.exp(u)/math.sqrt(23)))*math.exp(u/2) for u in us])
d=float(np.max(np.abs(Phi_inv-Phi_aut))); print('Polya kernel: inverse transform of Artin Lambda vs eta product, max diff',d,'max',Phi_aut.max(),'min',Phi_aut.min())
out['us']=us.tolist(); out['Phi_inv']=Phi_inv.tolist(); out['Phi_aut']=Phi_aut.tolist(); out['Phi_diff']=d
out['ts']=ts[::20].tolist(); out['Lam']=Lam[::20].tolist()
out['Lambda_half']=float(Lam[0])
json.dump(out,open('switch23_b.json','w'))
