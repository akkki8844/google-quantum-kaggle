"""GF(2) affine rank of the 37 wire values (over sampled x) at every line of an invsqrt .kmx.
deficiency(t) = 38 - rank([wires; 1]). Zero wires count as dependencies. Our own sim."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse, Sim, edge_and_random, unpack

def gf2_rank(M):
    """M: (k, W) uint64 rows (bit-packed). Rank over GF(2)."""
    M = M.copy(); k, W = M.shape; r = 0
    for w in range(W):
        for bit in range(64):
            if r == k: return r
            sel = (M[r:, w] >> np.uint64(bit)) & np.uint64(1)
            nz = np.flatnonzero(sel) + r
            if len(nz) == 0: continue
            p = nz[0]
            if p != r: M[[r, p]] = M[[p, r]]
            rows = np.flatnonzero((M[:, w] >> np.uint64(bit)) & np.uint64(1)); rows = rows[rows != r]
            M[rows] ^= M[r]; r += 1
    return r

path = sys.argv[1]; nshots = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
nq, body = parse(path); xs = edge_and_random(nshots); sim = Sim(nq, xs, seed=11)
busy = np.load(path + ".busy.npy")
full = busy.sum(axis=1) == nq
ones = np.full((1, sim.W), np.uint64(0xFFFFFFFFFFFFFFFF))
defi = np.full(len(body) + 1, -1)
def hook(i, s):
    if full[i] and i % step == 0:
        defi[i] = nq + 1 - gf2_rank(np.vstack([s.q, ones]))
sim.run(body, hook=hook)
d = defi[defi >= 0]
print("full-width lines checked", len(d), "deficiency histogram", np.unique(d, return_counts=True))
np.save(path + ".defi.npy", defi)
zero = np.flatnonzero(defi == 0)
print("lines with NO redundancy at full width:", len(zero), zero[:60].tolist())
