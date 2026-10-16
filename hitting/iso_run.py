import cypari2, json, time
pari=cypari2.Pari(); pari.allocatemem(4*10**9)
pari('read("iso.gp")')
out=[]
Ns=[p for p in range(13,700) if pari.isprime(p) and p%12==1]
for l in [2,3,5,7]:
    for N in Ns:
        if l==N: continue
        t=time.time(); r=pari(f"check({N},{l})")
        row=[int(r[0]),int(r[1]),int(r[2]),int(r[3]),str(r[4]),str(r[5]),int(r[6]),int(r[7]),bool(r[8]==1),bool(r[9]==1),bool(r[10]==1)]
        out.append(row)
        if N<200 or not (row[8] and row[9]): print(row, round(time.time()-t,1), flush=True)
json.dump(out,open('iso.json','w'))
print('all trees ok',all(r[8] for r in out),'all kemeny ok',all(r[9] for r in out),'symmetric',all(r[10] for r in out),len(out))
