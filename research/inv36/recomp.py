"""For candidate (wire w, untouched interval (s, e)): is w's value at line e an affine / quadratic GF(2) function of the
other wires' values at some line t in [s+1, e] (sampled shots)? Reports the latest t with an affine solution and the
number of quadratic terms needed at e (greedy-free: rank of the solution support)."""
import sys, json, numpy as np
from itertools import combinations
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv36")
from bitsim import parse, Sim, edge_and_random, unpack
from quad import in_span
path = sys.argv[1]; cands = json.loads(sys.argv[2])
nq, body = parse(path); xs = edge_and_random(1500); sim = Sim(nq, xs, seed=51)
ones = np.full(sim.W, np.uint64(0xFFFFFFFFFFFFFFFF))
val = {}; res = {}
checkpoints = {}
for k, (w, s, e) in enumerate(cands):
    for t in sorted({s + 1, (s + e) // 2, e - 20, e}):
        checkpoints.setdefault(t, []).append((k, w))
def hook(i, st):
    for k, w in checkpoints.get(i, []):
        others = [j for j in range(nq) if j != w]
        v = st.q[w].copy()
        aff = in_span(np.vstack([ones] + [st.q[j] for j in others]), v)
        quad = None
        if not aff:
            feats = [ones] + [st.q[j] for j in others] + [st.q[a] & st.q[b] for a, b in combinations(others, 2)]
            quad = in_span(np.array(feats), v)
        res.setdefault(k, []).append((i, "affine" if aff else ("quadratic" if quad else "no")))
sim.run(body, hook=hook)
for k, (w, s, e) in enumerate(cands): print((w, s, e), res.get(k))
