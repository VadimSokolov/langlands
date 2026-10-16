import cypari2, numpy as np, json, math
from scipy.special import digamma
from scipy.integrate import quad
pari=cypari2.Pari(); pari.allocatemem(3*10**9)
pari("L=lfunmf(mfinit([1,12]),mfDelta());")
zD=[float(z) for z in pari("lfunzeros(L,120)") if float(z)>1e-6]
pari("T=mfcoefs(mfDelta(),170000); P=primes([2,170000]);")
P=[int(p) for p in pari("P")]
tau={p:float(pari(f"T[{p}+1]")) for p in P}
wid=1.0
def arch_density(t): return 2*np.real(-math.log(2*math.pi)+digamma(6+1j*t))/(2*math.pi)
def bump_prime(t0):
    h=lambda t: math.exp(-(t-t0)**2/(2*wid*wid))+math.exp(-(t+t0)**2/(2*wid*wid))
    arch=quad(lambda t:h(t)*arch_density(t),-60,60,points=[-t0,t0],limit=400)[0]
    umax=math.sqrt(2*60)/wid+1; pr=0.0
    for p in P:
        Lp=math.log(p)
        if Lp>umax: break
        xp=tau[p]/p**5.5; cs=[2.0,xp]
        for m in range(2,40): cs.append(xp*cs[-1]-cs[-2])
        m=1
        while m*Lp<=umax:
            u=m*Lp; H=2*wid*math.sqrt(2*math.pi)*math.exp(-wid*wid*u*u/2)*math.cos(t0*u)
            pr+=cs[m]*Lp*p**(-m/2)*H; m+=1
    return arch-pr/math.pi
def bump_zero(t0): return sum(math.exp(-(g-t0)**2/2)+math.exp(-(g+t0)**2/2) for g in zD)*2
grid=np.linspace(0.5,40,160)
bp=np.array([bump_prime(t) for t in grid]); bz=np.array([bump_zero(t) for t in grid])
np.save("bump.npy",np.vstack([grid,bp,bz]))
eps=0.05
def U(z):
    r=math.sqrt(z); tot=0.0
    for g in zD[:60]: tot+=2*(eps/(eps**2+g**2)-0.5*(eps/(eps**2+(r-g)**2)+eps/(eps**2+(r+g)**2)))
    return -tot/(2*math.pi*r)
zz=np.linspace(4,900,4000); Uv=np.array([U(z) for z in zz]); np.save("transport.npy",np.vstack([zz,Uv]))
g1=zD[0]**2; m=np.abs(zz-g1)<30
out={"bump":{"width":wid,"max_abs_diff":float(np.max(np.abs(bp-bz))),"max_value":float(bz.max())},
     "transport":{"eps":eps,"gamma1_sq":g1,"mass_near_first_atom":float(np.trapezoid(Uv[m],zz[m])),"min_U":float(Uv.min())},
     "vandantzig":{"density_at_0":1.2154575693664642,"min_density":0.00024405331726812664,"total_mass":0.9999458104471022,"logconcave":True}}
print(out); json.dump(out,open("stage6.json","w"),indent=1)
