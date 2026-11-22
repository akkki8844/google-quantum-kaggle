"""Which wires of an invsqrt .kmx hold the normalisation shift k (x = n / 4^k, n in [2^20, 2^22)) or simple functions of it,
and over which lines? Our own sim, sampled x."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse, Sim, edge_and_random, unpack, pack
path = sys.argv[1]
nq, body = parse(path); xs = edge_and_random(3000); sim = Sim(nq, xs, seed=61)
k = np.zeros(len(xs), np.int64); n = xs.astype(np.int64).copy()
for _ in range(11):
    sh = n < (1 << 20); n = np.where(sh, n << 2, n); k += sh
feats = {f"k{b}": ((k >> b) & 1) == 1 for b in range(4)}
for t in range(1, 11): feats[f"k>={t}"] = k >= t
F = {nm: pack(v[None])[0] for nm, v in feats.items()}
hold = {}
step = 50
def hook(i, s):
    if i % step: return
    for w in range(nq):
        for nm, v in F.items():
            if (s.q[w] == v).all() or (s.q[w] == ~v).all():
                hold.setdefault((w, nm), []).append(i)
sim.run(body, hook=hook)
L = len(body)
for (w, nm), ls in sorted(hold.items(), key=lambda kv: -len(kv[1])):
    print(f"q{w} = {nm}: {len(ls) * step} lines (first {ls[0]}, last {ls[-1]})")
