import numpy as np, json, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight"})
D="/home/claude/langlands/book/figs/"; C={"ink":"#1F3B63","rust":"#9C3D1F","teal":"#1B7F79","gold":"#C99A2E","plum":"#6B3F7A","sky":"#8FB3DE","sand":"#E2C9A6"}
xs=np.linspace(-2,2,801)
def km(q,x): return (q+1)/((math.sqrt(q)+1/math.sqrt(q))**2-x**2)*np.sqrt(np.clip(4-x**2,0,None))/(2*np.pi)
v2=np.load("vert2.npy"); v3=np.load("vert3.npy")
fig,ax=plt.subplots(1,2,figsize=(4.4,2.1))
for a,v,q,c,cc in [(ax[0],v2,2,C["sky"],C["ink"]),(ax[1],v3,3,C["sand"],C["rust"])]:
    a.hist(v,bins=70,density=True,color=c,edgecolor="none"); a.plot(xs,km(q,xs),color=cc,lw=1.3); a.set_title(f"$T_{q}$ on $S_2(\\Gamma_0(N))$",fontsize=9); a.set_xlabel(r"$\lambda/\sqrt{%d}$"%q)
plt.tight_layout(); plt.savefig(D+"vertical.pdf"); plt.close()
s2=json.load(open("stage2.json")); R=s2["rows"]
h=np.array([r["h"] for r in R]); k2=np.array([r["K_over_h2"] for r in R]); k3=np.array([r["K_over_h3"] for r in R])
fig,ax=plt.subplots(figsize=(4.4,2.2))
ax.plot(h,k2,"o",ms=3,color=C["ink"],label=r"$\ell=2$"); ax.axhline(2,color=C["ink"],ls=":",lw=1)
ax.plot(h,k3,"s",ms=3,color=C["rust"],label=r"$\ell=3$"); ax.axhline(1.5,color=C["rust"],ls=":",lw=1)
ax.set_xlabel(r"vertices $h=(N-1)/12$"); ax.set_ylabel(r"$K_N/h$"); ax.legend(frameon=False,fontsize=7)
plt.savefig(D+"kemeny.pdf"); plt.close()
import cypari2; pari=cypari2.Pari(); pari.allocatemem(10**9); pari('read("ss_graph.gp")')
pari("A=ssgraph(1009,2)"); n=int(pari("#A")); M=np.array([[int(pari(f"A[{i},{j}]")) for j in range(1,n+1)] for i in range(1,n+1)])
G=nx.from_numpy_array(M); P=M/3; I=np.eye(n); H=np.zeros((n,n))
for v in range(n):
    idx=[i for i in range(n) if i!=v]; H[idx,v]=np.linalg.solve((I-P)[np.ix_(idx,idx)],np.ones(n-1))
pos=nx.kamada_kawai_layout(G)
fig,ax=plt.subplots(figsize=(4.4,3.2)); ax.axis("off")
nx.draw_networkx_edges(G,pos,ax=ax,width=0.6,edge_color="#9AA5B1")
sc=nx.draw_networkx_nodes(G,pos,ax=ax,node_size=28,node_color=H.mean(0)/n,cmap="viridis")
cb=plt.colorbar(sc,ax=ax,fraction=0.04); cb.set_label(r"mean hitting time$/h$",fontsize=8)
plt.savefig(D+"isogeny.pdf"); plt.close()
g=np.load("g2dens.npy"); mid=g[0]; names=["USp(4)","SU(2)xSU(2)","SU(2) diagonal"]; cols=[C["ink"],C["teal"],C["rust"]]
fig,ax=plt.subplots(1,3,figsize=(4.6,1.9),sharey=True)
for a,f,t in zip(ax,["C1","C2","C3"],["$y^2=x^5-x+1$","$y^2=x^6+3x^4+x^2+1$","$y^2=x^6+x^3+1$"]):
    x=np.load(f"g2_{f}.npy"); a.hist(x,bins=60,density=True,color="#D9D9D9",edgecolor="none")
    for k,c in zip(range(3),cols): a.plot(mid,g[k+1],color=c,lw=1)
    a.set_title(t,fontsize=7); a.set_xlabel(r"$a_p/\sqrt{p}$",fontsize=7); a.set_xlim(-4,4)
ax[0].legend(names,frameon=False,fontsize=5.5,loc="upper left"); plt.tight_layout(); plt.savefig(D+"genus2.pdf"); plt.close()
print("ok")
