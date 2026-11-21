"""Free-Toffoli scans on an invsqrt .kmx with our numpy sim (no third-party code).
For every CCX/CCZ/CSWAP (top level or conditioned), over the shots where it acts:
  A  CCX a b t with t == a&b before (or == a&b ^ 1): MBU erase, 0 Toffolis.
  B  product a&b affine in the current qubits (CCX -> CNOTs, CCZ -> CZs): 0 Toffolis.
  C  CCZ / CCX product identically 0 on acting shots: delete.
  D  CCX a b t where t_after is affine in other qubits -> MBU + CNOT rebuild.
Sampled evidence only; any hit must pass the exact all-x check."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse, Sim, edge_and_random, unpack

def rank_solve(A, y):
    """A: (k, S) bool basis rows; y: (S,) bool. Return coefficient vector or None (GF(2))."""
    M = np.vstack([A, y[None]]).T.astype(np.uint8).copy()
    k = A.shape[0]; piv = []; r = 0
    for c in range(k):
        rows = np.flatnonzero(M[r:, c]) + r
        if len(rows) == 0: continue
        M[[r, rows[0]]] = M[[rows[0], r]]
        hit = np.flatnonzero(M[:, c]); hit = hit[hit != r]
        M[hit] ^= M[r]; piv.append(c); r += 1
        if r == M.shape[0]: break
    if M[r:, k].any(): return None
    x = np.zeros(k, dtype=np.uint8)
    for i, c in enumerate(piv): x[c] = M[i, k]
    return x

path = sys.argv[1]; nshots = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
nq, body = parse(path)
xs = edge_and_random(nshots)
sim = Sim(nq, xs, seed=7)
hits = []
nested = [0]
def hook(i, s):
    op, a, c = body[i]
    if op not in ("CCX", "CCZ", "CSWAP"):
        return
    m = unpack(s.mask(c)[None])[0]
    if not m.any(): hits.append((i, op, "NEVER-ACTS")); return
    Q = unpack(s.q)[:, m]                     # (nq, S_act)
    if op == "CSWAP":
        return
    pa, pb = Q[a[0]], Q[a[1]]
    prod = pa & pb
    if not prod.any(): hits.append((i, op, "C:zero")); return
    if op == "CCX":
        t = Q[a[2]]
        if (t == prod).all(): hits.append((i, op, "A:t==ab")); return
        if (t == ~prod).all(): hits.append((i, op, "A:t==ab^1")); return
    basis = np.vstack([Q[[j for j in range(nq) if not (op == "CCX" and j == a[2])]], np.ones((1, Q.shape[1]), bool)])
    sol = rank_solve(basis[:, :4096], prod[:4096])
    if sol is not None:
        # confirm on all acting shots
        pred = (basis[sol.astype(bool)].sum(axis=0) & 1).astype(bool) if sol.any() else np.zeros_like(prod)
        if (pred == prod).all():
            hits.append((i, op, "B:affine " + str(np.flatnonzero(sol).tolist())))
sim.run(body, hook=hook)
print("acting-shot scan done; hits", len(hits))
for h in hits: print(h, body[h[0]])
