# Chebotarev statistics for f = x^3 - x - 1 (splitting field S_3, disc -23)
import numpy as np, json
N = 10**7
s = np.ones(N+1, bool); s[:2] = False
for i in range(2, int(N**0.5)+1):
    if s[i]: s[i*i::i] = False
primes = np.nonzero(s)[0]

def mulmod(a, b, p):
    # polynomials of degree <=2 as tuples (c0,c1,c2), reduce mod x^3 = x + 1
    c = [0]*5
    for i in range(3):
        for j in range(3):
            c[i+j] += a[i]*b[j]
    for k in (4, 3):
        t = c[k]; c[k] = 0
        c[k-3] += t; c[k-2] += t
    return (c[0] % p, c[1] % p, c[2] % p)

def xpow(p):
    r = (1, 0, 0); b = (0, 1, 0); e = p
    while e:
        if e & 1: r = mulmod(r, b, p)
        b = mulmod(b, b, p); e >>= 1
    return r

def leg(a, p):
    return pow(a % p, (p-1)//2, p)

counts = {'id': 0, 'tr': 0, 'c3': 0}
marks = [10**4, 10**5, 10**6, 10**7]
out = []
path = []
for p in primes:
    p = int(p)
    while marks and p > marks[0]:
        tot = sum(counts.values())
        out.append((marks[0], tot, dict(counts)))
        marks.pop(0)
    if p in (2, 23):
        if p == 2:
            # x^3-x-1 mod 2 has no roots, irreducible: 3-cycle
            counts['c3'] += 1
        pass
    elif leg(-23, p) == p-1:
        counts['tr'] += 1
    else:
        counts['id' if xpow(p) == (0, 1, 0) else 'c3'] += 1
    path.append((p, counts['id'], counts['tr'], counts['c3']))
tot = sum(counts.values()); out.append((N, tot, dict(counts)))
for m, tot, c in out:
    print(m, tot, c, {k: round(v/tot, 5) for k, v in c.items()})
json.dump(out, open('/tmp/claude-0/ch03_cheb.json', 'w'))
np.save('/tmp/claude-0/ch03_path.npy', np.array(path, dtype=np.int64))
