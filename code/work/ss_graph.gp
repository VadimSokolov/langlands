\\ supersingular 2 isogeny graph over F_{p^2}, adjacency with multiplicities
ssgraph(p,l)={
  my(T=ffinit(p,2,'a), g=ffgen(T,'a), D, H, f, j0, todo, seen, idx, A, P2=polmodular(l), n, fa, nb, cnt);
  D=3; while(kronecker(-D,p)!=-1 || !isfundamental(-D), D++);
  H=polclass(-D); f=subst(H,'x,'y)*g^0; fa=factor(f);
  j0=0; for(i=1,#fa[,1], if(poldegree(fa[i,1])==1, j0=-polcoef(fa[i,1],0)/polcoef(fa[i,1],1); break));
  seen=Map(); todo=List([j0]); mapput(seen,j0,1); n=1; edges=List();
  while(#todo>0, my(j=todo[#todo]); listpop(todo);
    fa=factor(subst(subst(P2,'x,j),'y,'Y)*g^0);
    for(i=1,#fa[,1], my(r=-polcoef(fa[i,1],0)/polcoef(fa[i,1],1));
      if(!mapisdefined(seen,r), n++; mapput(seen,r,n); listput(todo,r));
      listput(edges,[mapget(seen,j),mapget(seen,r),fa[i,2]])));
  A=matrix(n,n); for(k=1,#edges, my(e=edges[k]); A[e[1],e[2]]+=e[3]); A
}
