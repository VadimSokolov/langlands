from mpmath import mp, mpf, gamma, exp, pi, log, sqrt, qp, quad, inf, cos, diff, factorial, findroot
import sys
mp.dps=30
kap = pi*mpf(2)**(mpf(4)/3)*gamma(mpf(2)/3)/gamma(mpf(1)/3)**2
def eta0(t): return exp(-pi*t/12)*qp(exp(-2*pi*t))
def eta_i(t): return eta0(t) if t>=1 else eta0(1/t)/sqrt(t)
f=lambda t: kap*eta_i(t)**4
Phi=lambda u: kap*exp(u)*eta_i(exp(u))**4
part=sys.argv[1]
if part=='hcm':
    for u in [mpf('0.3'),mpf('0.6'),1,2,mpf('3.5')]:
        bad=[]
        for w in [mpf('2.2'),3,4,6,9]:
            g=lambda w: f(u*(w+sqrt(w*w-4))/2)*f(u*2/(w+sqrt(w*w-4)))
            ds=[(-1)**k*diff(g,w,k) for k in range(1,7)]
            if min(ds)<=0: bad.append((float(w),[float(d) for d in ds]))
        print('u',float(u),'bad',bad[:2])
if part=='cum':
    mu=[quad(lambda x: x**(2*m)*Phi(x),[-6,-3,-1,0,1,3,6]) for m in range(0,9)]
    print('mass',mu[0])
    # moment series in z=t^2 : E cos(tX)=sum mu_{2m}(-1)^m t^{2m}/(2m)!
    a=[mu[m]*(-1)**m/factorial(2*m) for m in range(9)]
    # log series: L=log(sum a_m z^m)
    L=[mpf(0)]*9
    for n in range(1,9):
        L[n]=a[n]-sum(k*L[k]*a[n-k] for k in range(1,n))/n
    # log Xi = sum log(1 - z/rho^2) = -sum_m Z_m z^m/m  -> Z_m = -m L_m
    Z=[None]+[-m*L[m] for m in range(1,9)]
    zeros=[mpf(x) for x in "3.443343367909476878929937586 5.802689552546195901316320248 7.266467310821318522722657751 9.113424945499136957715665264 10.17441103098667470227930586 11.77437667375267836950691611 13.01055982622439917603548235 13.99634105119279079569568395 15.69696813163500519905091039 16.25038600345248605266882135 17.68732582410033881966766579 18.85850998760573807432087802 19.74858543777126986291158008 20.79845361945188374092813878 22.1898009582103842508305165 22.75894922961026437740236755 23.9110860307330405403237802".split()]
    for m in range(1,9):
        zs=sum(g**(-2*m) for g in zeros)
        print(m, 'Z_m from cumulants', mp.nstr(Z[m],15), 'from 17 zeros', mp.nstr(zs,15), 'gamma1 est', mp.nstr((Z[m]/Z[m+1])**0.5,12) if m<8 else '', 'Z_m^(-1/2m)', mp.nstr(Z[m]**(-mpf(1)/(2*m)),12))
    print('var logT', mu[1])
if part=='W':
    Mom=lambda y: quad(lambda x: exp(y*x)*Phi(x),[-7,-3,-1,0,1,3,7])  # E T^y
    mp.dps=20
    zeros=[mpf(x) for x in "3.443343367909476878929937586 5.802689552546195901316320248 7.266467310821318522722657751 9.113424945499136957715665264 10.17441103098667470227930586 11.77437667375267836950691611 13.01055982622439917603548235 13.99634105119279079569568395 15.69696813163500519905091039 16.25038600345248605266882135 17.68732582410033881966766579 18.85850998760573807432087802 19.74858543777126986291158008 20.79845361945188374092813878 22.1898009582103842508305165 22.75894922961026437740236755 23.9110860307330405403237802".split()]
    # cache recip on grid using even symmetry
    import mpmath
    ys=[mpf(k)/20 for k in range(0,401)]  # y up to 20
    R=[1/Mom(y) for y in ys]
    print('recip at 0,5,10,20', R[0],R[100],R[200],R[400])
    h=mpf(1)/20
    def pW(x):  # (1/pi) int_0^inf cos(xy) R(y) dy, Simpson
        s=R[0]+R[-1]*cos(x*ys[-1])
        for k in range(1,400): s+= (4 if k%2 else 2)*R[k]*cos(x*ys[k])
        return s*h/3/pi
    def pPF(x):
        tot=0
        for k,g in enumerate(zeros):
            c=1
            for j,gj in enumerate(zeros):
                if j!=k: c*=1/(1-g**2/gj**2)
            tot+=c*g/2*exp(-g*abs(x))
        return tot
    for x in [0,0.05,0.1,0.2,0.3,0.5,0.8,1.2,1.6,2.0]:
        print('x',x, mp.nstr(pW(x),10), mp.nstr(pPF(x),10))
