import mpmath as mp
mp.mp.dps=20
g=[mp.mpf('14.134725'),mp.mpf('21.022040'),mp.mpf('25.010858'),mp.mpf('30.424876'),mp.mpf('32.935062')]
def atoms(delta):
    a=[]
    for k,gk in enumerate(g):
        if k==2 and delta!=0:
            a+= [(gk+1j*delta)**2,(gk-1j*delta)**2]
        else: a.append(gk**2)
    return a
def u(tau,eps,delta):
    s=mp.mpc(-tau,eps)
    GG=sum(1/(s+a) for a in atoms(delta))
    return -mp.im(GG)/mp.pi
for delta in [0, 0.3]:
    for eps in [2, 20]:
        taus=[mp.mpf(k) for k in range(1,1300)]
        us=[u(t,eps,delta) for t in taus]
        m=min(us); i=us.index(m)
        print(f"delta={delta} eps={eps} min u = {float(m):.5f} at tau={float(taus[i]):.0f} (gamma_3^2-delta^2={float(g[2]**2-delta**2):.1f}, B=2 gamma delta={float(2*g[2]*delta):.2f})")
