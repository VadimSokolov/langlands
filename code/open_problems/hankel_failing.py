import sys, json
src=open('certify_patie_rig.py').read().split("if which==")[0]
sys.argv=['x','none',sys.argv[1],sys.argv[2]]
exec(src)
from flint import arb
def hankel(c,K):
    a=[ci/c[0] for ci in c]; L=[arb(0)]*len(c)
    for n in range(1,len(c)):
        L[n]=a[n]-sum(k*L[k]*a[n-k] for k in range(1,n))/n
    S=[None]+[((-1)**(n+1))*n*L[n] for n in range(1,len(c))]
    def det(M):
        A=[r[:] for r in M]; d=arb(1)
        for i in range(len(A)):
            p=A[i][i]; d*=p
            for r in range(i+1,len(A)):
                f=A[r][i]/p
                for cc in range(i,len(A)): A[r][cc]-=f*A[i][cc]
        return d
    res={}
    for shift in (1,2):
        for m in range(1,7):
            if 2*m-2+shift>=len(c): break
            d=det([[S[i+j+shift] for j in range(m)] for i in range(m)]); res[f"s{shift}m{m}"]=(bool(d>0),str(d)[:40])
    return S,res
out={}
for name,fn in (("11a1",lambda: coeffs_ec([0,-1,1,-10,-20],11)),("chi-3",lambda: coeffs_dir(3,1)),("chi5",lambda: coeffs_dir(5,0)),("chi-4chi5",lambda: conv(coeffs_dir(4,1),coeffs_dir(5,0)))):
    c=fn(); S,res=hankel(c,K); out[name]={"S1":str(S[1])[:40],"hankel":res}
    print(name,"S1",str(S[1])[:30],{k:v[0] for k,v in res.items()},flush=True)
json.dump(out,open("hankel_failing.json","w"),indent=1)
