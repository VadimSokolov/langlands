import json, cypari2
exec(open("cm_mean.py").read().split("import sys")[0])
out={}
for l in [int(p) for p in pari("primes([2,101])")]:
    J=60 if l<5 else (30 if l<20 else 20)
    w4=cm_weight(l,-4,J); w3=cm_weight(l,-3,J); c=c_closed(l)
    out[l]={"c":float(c),"w4":float(w4),"w3":float(w3),"mean":float(c+w4+w3),"l_mod_4":l%4,"l_mod_3":l%3,
            "inert_formula":-(12*l*l+13*l+13)/(12*(l-1)**2)}
    print(l,out[l]["mean"],out[l]["inert_formula"] if (l%12==11) else "",flush=True)
json.dump(out,open("limit_means.json","w"),indent=1)
