"""Affine relocation: (1) G1's T-dagger bit v in the REAL span of the rotation firing bits (then shift angles: free);
(2) for each CS gadget, lines where b743 matches and a&b is in the GF(2) affine span of the 37 wires on the b743=1 shots
(then i^{ab} = Clifford S/CZ network there)."""
import sys, numpy as np
from fractions import Fraction
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv36")
from bitsim import parse, Sim, edge_and_random, unpack
from quad import in_span
path = sys.argv[1]
nq, body = parse(path); xs = edge_and_random(2000); sim = Sim(nq, xs, seed=41)
G1 = 16639
CS = [(32661, 13, 14), (34340, 20, 26), (35288, 26, 13)]
rot = []; tgt = {}; hits = {g[0]: [] for g in CS}
ones = np.full(sim.W, np.uint64(0xFFFFFFFFFFFFFFFF))
def hook(i, s):
    op, a, c = body[i]
    if op == "Z_POW" and (Fraction(a[1]) * 4).denominator != 1:
        rot.append(unpack((s.q[a[0]] & s.mask(c))[None])[0].astype(float))
    if i == G1: tgt["g1"] = unpack(s.q[0][None])[0].astype(float)
    for (L, qa, qb) in CS:
        if i == L: tgt[L] = (s.bit("b743").copy(), (s.q[qa] & s.q[qb]).copy())
    if op in ("CX", "CCX", "X", "SWAP", "CSWAP", "HMR") or i in [g[0] for g in CS]:
        for (L, qa, qb) in CS:
            if L in tgt and i % 1 == 0:
                cg, p = tgt[L]
                if True:
                    rows = np.vstack([s.q & cg[None], cg[None]])
                    if in_span(rows, p & cg): hits[L].append(i)
sim.run(body, hook=hook)
B = np.array(rot).T; v = tgt["g1"]
for name, M in (("rot", B), ("rot+const", np.hstack([B, np.ones((B.shape[0], 1))]))):
    y, res, rk, _ = np.linalg.lstsq(M, v, rcond=None)
    print("G1 in real span of", name, "max residual", np.abs(M @ y - v).max())
for L, hh in hits.items():
    print("CS gadget", L, "affine-span hit lines:", len(hh), hh[:10], hh[-5:])
