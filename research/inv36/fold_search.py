"""Phase relocation search for the remaining T gadgets of an invsqrt .kmx (our own sim, sampled shots).
G1: unconditioned T-dagger on q0 at body 16639 -> is its bit equal to some rotation's firing bit (then fold -1/8 turn
    into that rotation's angle)? Also complement / equal to sums of two disjoint rotation bits.
G2..G4: CS gadgets (i^{ab} if b743) -> at which lines does b743 hold the same value as at the gadget and some wire hold a&b
    (on the shots where b743 = 1)? Then an S on that wire under b743 replaces the gadget's 3 T."""
import sys, numpy as np
from fractions import Fraction
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse, Sim, edge_and_random, unpack
path = sys.argv[1]
nq, body = parse(path); xs = edge_and_random(3000); sim = Sim(nq, xs, seed=31)
G1 = 16639
CS = [(32661, 13, 14), (34340, 20, 26), (35288, 26, 13)]
rot = []; tgt = {}; hits = {g[0]: [] for g in CS}
def hook(i, s):
    op, a, c = body[i]
    if op == "Z_POW" and (Fraction(a[1]) * 4).denominator != 1:
        rot.append((i, (s.q[a[0]] & s.mask(c)).copy()))
    if i == G1:
        tgt["g1"] = s.q[0].copy()
    for (L, qa, qb) in CS:
        if i == L:
            tgt[L] = (s.bit("b743").copy(), (s.q[qa] & s.q[qb]).copy())
    for (L, qa, qb) in CS:
        if L in tgt:
            cg, p = tgt[L]
            if (s.bit("b743") == cg).all():
                for w in range(nq):
                    if ((s.q[w] ^ p) & cg == 0).all():
                        hits[L].append((i, w))
sim.run(body, hook=hook)
g1 = tgt["g1"]
print("G1 bit mean", unpack(g1[None])[0].mean())
for k, (i, b) in enumerate(rot):
    if (b == g1).all(): print("G1 == rotation", k, i)
    if (b == ~g1).all(): print("G1 == NOT rotation", k, i)
for k1 in range(len(rot)):
    for k2 in range(k1 + 1, len(rot)):
        b1, b2 = rot[k1][1], rot[k2][1]
        if not (b1 & b2).any() and ((b1 | b2) == g1).all(): print("G1 == rot", k1, "+ rot", k2)
for L, hh in hits.items():
    seg = {}
    for i, w in hh: seg.setdefault(w, []).append(i)
    print("CS gadget", L, {w: (min(v), max(v), len(v)) for w, v in seg.items()})
