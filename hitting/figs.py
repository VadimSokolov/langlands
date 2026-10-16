import numpy as np, json, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.dpi":220})
C=["#2E86AB","#C73E1D","#F18F01","#5C4D7D","#3B8B5A","#A23B72"]
d=json.load(open('graphs.json'))
fig,ax=plt.subplots(1,2,figsize=(6.6,2.7))
for a,ls in zip(ax,[(2,3),(5,7)]):
    for c,l in zip(C,ls):
        r=[x for x in d if x['l']==l and 'trees' in x and x['g']>=2]
        N=np.array([x['N'] for x in r]); y=np.array([math.log(x['J0'])/x['g'] for x in r])
        a.plot(N,y,'o',ms=2.5,color=c,label=rf"$\ell={l}$")
        lyons=(l*math.log(l)-0.5*(l-1)*math.log(l*l-1))*1.0
        a.axhline(lyons,color=c,lw=1,ls='--')
        a.axhline(2*math.log(math.sqrt(l)+1),color=c,lw=0.6,ls=':'); a.axhline(2*math.log(math.sqrt(l)-1),color=c,lw=0.6,ls=':')
    a.set_xlabel("$N$"); a.legend(frameon=False,fontsize=7,loc="center right")
ax[0].set_ylabel(r"$g^{-1}\log\,|J_0(N)(F_\ell)|$")
ax[0].set_title("spanning trees: Lyons entropy (dashed), Weil bounds (dotted)",fontsize=7.5)
ax[1].set_title(r"$\ell=5,7$",fontsize=8)
plt.tight_layout(); plt.savefig('figs/hit_isogeny.pdf'); plt.close()
o=json.load(open('hitrates.json')); s=np.array(o['sigs'])
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
for c,(k,lab,m) in zip(C,[('id','split completely',1/6),('tr','transpositions',1/2),('3c','three cycles',1/3),('qr',r'$(\frac{7}{p})\ne(\frac{p}{7})$',1/2)]):
    ax[0].semilogx(s-1,o['top'][k],'o-',ms=3,color=c,lw=1,label=lab); ax[0].axhline(m,color=c,lw=0.6,ls=':')
ax[0].invert_xaxis(); ax[0].set_xlabel(r"$\sigma-1$"); ax[0].set_ylabel("P(top prime factor in class)")
ax[0].legend(frameon=False,fontsize=6.3,loc="upper center",bbox_to_anchor=(0.5,-0.3),ncol=2); ax[0].set_title("Frobenius of the largest prime factor",fontsize=8)
for c,(k,lab) in zip(C,[('id_1.01','split completely'),('tr_1.01','transpositions')]):
    L=o['law'][k]; ax[1].plot(L['v'],L['exact'],'o',ms=2.5,color=c,label=lab+r", $\sigma=1.01$"); ax[1].plot(L['v'],L['limit'],'-',color=c,lw=1)
ax[1].set_xlabel(r"$v$"); ax[1].set_ylabel(r"P$(V_C\leq v)$"); ax[1].legend(frameon=False,fontsize=6.5,loc="lower right")
ax[1].set_title(r"limit $\exp(-\mu(C)E_1(v))$ (lines)",fontsize=8)
plt.tight_layout(); plt.savefig('figs/hit_rates.pdf'); plt.close()
# exit identity figure
def chars():
    lg={1:0,2:1,4:2,3:3}
    return [(3,lambda r:[0,1,-1][r%3],r'$\chi_{-3}$'),(4,lambda r:[0,1,0,-1][r%4],r'$\chi_{-4}$'),(7,lambda r:[0,1,1,-1,1,-1,-1][r%7],r'$\chi_{-7}$'),(5,lambda r:0 if r%5==0 else 1j**lg[r%5],r'$\chi_5$, imaginary part')]
t=np.linspace(0.08,6,300)
fig,ax=plt.subplots(1,1,figsize=(4.6,2.6))
for c,(q,chi,lab) in zip(C,chars()):
    tau=sum(chi(r)*np.exp(2j*np.pi*r/q) for r in range(1,q))
    img=2/np.sqrt(2*np.pi*t**3)*sum(chi(m)*m*np.exp(-m*m/(2*t)) for m in range(1,80))
    eig=(-4*np.pi*1j*tau/q**2)*sum(np.conj(chi(j))*j*np.exp(-2*np.pi**2*j*j*t/q**2) for j in range(1,80))
    f=(lambda z: z.imag) if q==5 else (lambda z: np.real(z))
    ax.plot(t,f(eig),'-',color=c,lw=1.3,label=lab); ax.plot(t[::12],f(img)[::12],'o',ms=3,color=c)
ax.set_xlabel("$t$"); ax.set_ylabel("signed exit density"); ax.legend(frameon=False,fontsize=7)
ax.set_title("images (dots) against eigenfunctions (lines)",fontsize=8)
plt.tight_layout(); plt.savefig('figs/hit_exit.pdf'); plt.close()
