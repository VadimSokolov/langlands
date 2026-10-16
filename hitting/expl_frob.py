import numpy as np, mpmath as mp, json, cypari2
mp.mp.dps=20
d=np.load('/home/claude/langlands/newrecip/data.npz'); P=d['P'].astype(float); c1=d['c1']; X=1e7
logs=[]; coef=[]
for m in range(1,24):
    mask=P**m<=X
    if not mask.any(): break
    pp=P[mask]; cl=c1[mask]
    a=np.where(cl==0,2.0,np.where(cl==1,2.0 if m%2==0 else 0.0, 2.0 if m%3==0 else -1.0)); a=np.where(pp==23,1.0,a)
    logs.append(m*np.log(pp)); coef.append(a*np.log(pp)/pp**(m/2))
L=np.concatenate(logs); Cf=np.concatenate(coef)
g=lambda t: mp.log(23)+2*mp.re(-mp.log(2*mp.pi)+mp.digamma(mp.mpf('0.5')+1j*t))
def arch(x):
    return float(mp.quad(lambda t: mp.e**(-x*t*t)*g(t),[0,1,5,20,mp.inf])/mp.pi)  # (1/2pi)*2*int_0^inf
def W(x):
    prime=(1/np.pi)*np.sum(Cf*np.sqrt(np.pi/x)*np.exp(-L**2/(4*x)))
    return 0.5*(arch(x)-prime)
pari=cypari2.Pari(); pari.allocatemem(10**9)
pari("L=lfundiv(lfuncreate(x^3-x-1),1); Z=lfunzeros(L,400);"); zs=np.array([float(z) for z in pari("Z")])
for x in [0.001,0.005,0.02,0.1,0.3]:
    print(x,W(x),np.sum(np.exp(-x*zs**2)))

I=float(mp.quad(lambda y: W(float(mp.e**(-y)))*mp.e**(-y),[-mp.log(1.1),2,6,12,30,60]))
print('E T from Frobenius', I)
json.dump({'ET_frobenius':I},open('expl_frob.json','w'))
