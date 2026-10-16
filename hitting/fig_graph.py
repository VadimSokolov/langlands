import cypari2, numpy as np, json, random
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams.update({"font.family":"serif","font.size":9,"savefig.bbox":"tight","savefig.dpi":220})
pari=cypari2.Pari(); pari.allocatemem(10**9); pari('read("/home/claude/langlands/ss_graph.gp")')
N,l=157,2
pari(f"A=ssgraph({N},{l});"); AA=pari("A"); n=len(AA)
A=np.array([[int(AA[i,j]) for j in range(n)] for i in range(n)])
P=A/(l+1); pi=np.ones(n)/n
Z=np.linalg.inv(np.eye(n)-P+np.outer(np.ones(n),pi))-np.outer(np.ones(n),pi)
Et=np.diag(Z)/pi
# Wilson's algorithm
random.seed(3)
def step(v):
    return random.choices(range(n),weights=A[v])[0]
intree={0}; nxt={}
for s in range(n):
    u=s
    while u not in intree:
        nxt[u]=step(u); u=nxt[u]
    u=s
    while u not in intree:
        intree.add(u); u=nxt[u]
tree=[(u,nxt[u]) for u in nxt]
G=nx.Graph()
for i in range(n):
    for j in range(i+1,n):
        if A[i,j]>0: G.add_edge(i,j,w=A[i,j])
pos=nx.kamada_kawai_layout(G)
fig,ax=plt.subplots(figsize=(4.4,3.4))
for (i,j,dd) in G.edges(data=True):
    x=[pos[i][0],pos[j][0]]; y=[pos[i][1],pos[j][1]]
    ax.plot(x,y,color="#999999",lw=0.8+0.8*(dd['w']-1),zorder=1)
for (i,j) in tree:
    ax.plot([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]],color="#C73E1D",lw=2.2,zorder=2)
loops=[i for i in range(n) if A[i,i]>0]
sc=ax.scatter([pos[i][0] for i in range(n)],[pos[i][1] for i in range(n)],c=Et/n,cmap="viridis",s=120,zorder=3,edgecolors="k",linewidths=0.6)
for i in loops: ax.scatter([pos[i][0]],[pos[i][1]],s=260,facecolors='none',edgecolors="#F18F01",lw=1.2,zorder=2)
cb=plt.colorbar(sc,ax=ax,shrink=0.8); cb.set_label(r"$\mathbb{E}_\pi\tau_y/h$",fontsize=8)
ax.set_axis_off(); ax.set_title(f"$G_{{{N}}}({l})$: {n} curves, a Wilson spanning tree (red)",fontsize=8)
plt.savefig('figs/hit_graph.pdf'); plt.close()
pari(f"Lr=matrix({n-1},{n-1},i,j,if(i==j,{l+1}-A[i,j],-A[i,j]));"); tau=int(pari("matdet(Lr)"))
print(n,tau,Et.min()/n,Et.max()/n,len(loops), int(pari("matsnf(Lr)")[0]) if True else 0)
json.dump({'N':N,'l':l,'n':n,'tau':tau,'Et_over_h':(Et/n).tolist(),'loops':loops},open('graph157.json','w'))
