import cypari2, numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D","#3B8B5A"]
pari=cypari2.Pari(); pari.allocatemem(2*10**9)
X=200000
P=[int(p) for p in pari(f"primes([2,{X}])")]
pari(f"E=ellan(ellinit([0,-1,1,-10,-20]),{X});"); an=[int(x) for x in pari("E")]
x11=np.array([an[p-1]/np.sqrt(p) for p in P if p!=11])
w1={3:2,1:0,0:-1}
a1=np.array([w1[len(pari(f"polrootsmod(x^3-x-1,{p})"))] for p in P if p!=23])
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
vals=[-1,0,2]; freq=[np.mean(a1==v) for v in vals]; haar=[1/3,1/2,1/6]
ax[0].bar(np.array(vals)-0.15,freq,0.3,color=C[0],label="frequency, $p<2\\cdot10^5$")
ax[0].bar(np.array(vals)+0.15,haar,0.3,color=C[2],label="Haar measure on $S_3$")
ax[0].set_xticks(vals); ax[0].set_xlabel(r"$a_p$ of $\eta(z)\eta(23z)$, weight one")
ax[0].legend(frameon=False,fontsize=6.5); ax[0].set_title("weight one: an atomic label law",fontsize=8)
ax[1].hist(x11,bins=60,density=True,color=C[0],alpha=0.6,label=r"$a_p/\sqrt{p}$ for 11a1")
t=np.linspace(-2,2,300); ax[1].plot(t,np.sqrt(4-t*t)/(2*np.pi),color=C[1],lw=1.4,label="semicircle")
ax[1].set_xlabel(r"$a_p/\sqrt{p}$, weight two"); ax[1].legend(frameon=False,fontsize=6.5)
ax[1].set_title("weight two: a continuous label law",fontsize=8)
plt.tight_layout(); plt.savefig("figs/flt_labels.pdf"); plt.close()
# running mismatch mod 3 for the three rational level 77 newforms vs 11a1
X2=20000; P2=[int(p) for p in pari(f"primes([2,{X2}])")]
a11=[int(pari(f"ellap(ellinit([0,-1,1,-10,-20]),{p})")) for p in P2]
pari(f"mf=mfinit([77,2],0); B=mfeigenbasis(mf);")
fig,ax=plt.subplots(1,1,figsize=(4.6,2.5))
k=0
for i in range(1,5):
    if int(pari(f"poldegree(mffields(mf)[{i}])"))!=1: continue
    co=[int(x) for x in pari(f"mfcoefs(B[{i}],{X2})")]
    sel=[(p,co[p],a) for p,a in zip(P2,a11) if p not in (3,7,11)]
    mm=np.cumsum([(c-a)%3!=0 for p,c,a in sel])/np.arange(1,len(sel)+1)
    ax.semilogx([s[0] for s in sel],mm,color=C[k],lw=1.3,label=f"newform {k+1} of level 77"); k+=1
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"share of $\ell\leq x$ with $a_\ell\neq a_\ell(11a1)$ mod 3")
ax.legend(frameon=False,fontsize=7); ax.set_title("Ribet's congruence: zero discrepancy mass at level 77",fontsize=8)
plt.tight_layout(); plt.savefig("figs/flt_congruence.pdf"); plt.close()
