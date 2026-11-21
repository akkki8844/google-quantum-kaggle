"""Real-linear dependency among the fine-rotation firing patterns b_k(x) (and the constant) of an invsqrt .kmx."""
import sys, numpy as np
from fractions import Fraction
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse, Sim, edge_and_random, unpack
nq, body = parse(sys.argv[1]); xs = edge_and_random(int(sys.argv[2]) if len(sys.argv) > 2 else 6000); sim = Sim(nq, xs, seed=5)
F, names = [], []
def hook(i, s):
    op, a, c = body[i]
    if op == "Z_POW" and (Fraction(a[1]) * 4).denominator != 1:
        F.append(unpack((s.q[a[0]] & s.mask(c))[None])[0].astype(float)); names.append((i, a[0], c))
sim.run(body, hook=hook)
B = np.array(F).T
print("fine rotations", B.shape[1], "shots", B.shape[0])
for k in range(B.shape[1]): print(k, names[k], "mean", round(B[:, k].mean(), 4))
A = np.hstack([B, np.ones((B.shape[0], 1))])
s = np.linalg.svd(A, compute_uv=False)
print("rank with const", np.linalg.matrix_rank(A), "of", A.shape[1], "smallest sv", s[-4:])
# exact duplicates / complements
for i in range(B.shape[1]):
    for j in range(i + 1, B.shape[1]):
        if (B[:, i] == B[:, j]).all(): print("duplicate", i, j)
        if (B[:, i] == 1 - B[:, j]).all(): print("complement", i, j)
