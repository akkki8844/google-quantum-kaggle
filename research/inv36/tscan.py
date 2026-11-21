"""Mean executed T count of an invsqrt .kmx (Z_POW with odd multiple of 1/4 half-turn) and, for each T,
whether the controlled-S product it belongs to is affine / zero on acting shots (our own sim)."""
import sys, numpy as np
from fractions import Fraction
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse, Sim, edge_and_random, unpack
nq, body = parse(sys.argv[1]); xs = edge_and_random(4000); sim = Sim(nq, xs, seed=9)
tcount = 0; rows = []
def hook(i, s):
    global tcount
    op, a, c = body[i]
    if op != "Z_POW": return
    e = Fraction(a[1])
    if (e * 4).denominator == 1 and (e * 4).numerator % 2 == 1:
        m = unpack(s.mask(c)[None])[0]
        tcount += m.sum()
        q = unpack(s.q[a[0]][None])[0]
        rows.append((i, body[i], round(m.mean(), 4), round((q & m).mean(), 4)))
sim.run(body, hook=hook)
print("mean T per shot", tcount / sim.S)
for r in rows: print(r)
