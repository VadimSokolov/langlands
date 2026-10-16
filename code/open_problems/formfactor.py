import numpy as np, json, math
l=2; q=2
g=np.linspace(-1,1,400001)[1:-1]; lam=2*math.sqrt(q)*g
rho=(q+1)*np.sqrt(4*q-lam**2)/(2*math.pi*((q+1)**2-lam**2))*2*math.sqrt(q); rho/=np.trapezoid(rho,g)
th=np.arccos(g)
def U(j,x): t=np.arccos(np.clip(x,-1,1)); return np.sin((j+1)*t)/np.sin(t)
J=60
km_var=[]
for j in range(J+1):
    u=np.sin((j+1)*th)/np.sin(th); m1=np.trapezoid(u*rho,g); m2=np.trapezoid(u*u*rho,g); km_var.append(m2-m1*m1)
# ramp model from class number weights
o=json.load(open("kemeny_l2_J26.json"))
res={"km_var":km_var}
for N in (16033,32029,48049):
    ev=np.load(f"ev{N}.npy"); h=len(ev)+1; x=ev/(2*math.sqrt(l)); M=(N-1)/12
    F=[float(np.sum(U(j,x))-(M*l**(-j/2) if j%2==0 else 0)) for j in range(J+1)]
    res[str(N)]={"h":h,"F":F}
json.dump(res,open("formfactor.json","w"))
for N in (16033,32029,48049):
    F=np.array(res[str(N)]["F"]); h=res[str(N)]["h"]
    print(N,h,[round(F[j]**2/(l**(j/2)),2) for j in range(1,30,2)])
    print("  plateau ratio F^2/(h var)",[round(F[j]**2/(h*km_var[j]),2) for j in range(30,60,3)])
