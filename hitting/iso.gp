ssgraph(p,l)={
  my(T=ffinit(p,2,'a), g=ffgen(T,'a), D=3, f, fa, j0,
     seen=Map(), todo, edges=List(), n=1, A,
     P=polmodular(l));
  while(kronecker(-D,p)!=-1 || !isfundamental(-D), D++);
  fa=factor(subst(polclass(-D),'x,'y)*g^0);
  for(i=1,#fa[,1], if(poldegree(fa[i,1])==1,
     j0=-polcoef(fa[i,1],0)/polcoef(fa[i,1],1); break));
  mapput(seen,j0,1); todo=List([j0]);
  while(#todo>0, my(j=todo[#todo]); listpop(todo);
    fa=factor(subst(subst(P,'x,j),'y,'Y)*g^0);
    for(i=1,#fa[,1],
      my(r=-polcoef(fa[i,1],0)/polcoef(fa[i,1],1));
      if(!mapisdefined(seen,r),
         n++; mapput(seen,r,n); listput(todo,r));
      listput(edges,[mapget(seen,j),mapget(seen,r),fa[i,2]])));
  A=matrix(n,n);
  for(k=1,#edges, my(e=edges[k]); A[e[1],e[2]]+=e[3]);
  A
}
check(p,l)={
  my(A=ssgraph(p,l), h=#A, L=(l+1)*matid(h)-A, J=matrix(h,h,i,j,1/h),
     Kg=(l+1)*(trace((L+J)^-1)-1), tau=matdet(L[2..h,2..h]),
     mf=mfinit([p,2],1), M=mfheckemat(mf,l), P=charpoly(M), d=poldegree(P),
     Km=(l+1)*subst(deriv(P),'x,l+1)/subst(P,'x,l+1), Jl=subst(P,'x,l+1));
  [p,l,h,d,Kg,Km,tau,Jl,Jl/h==tau, Kg==Km, A==A~]
}
