"""At the start of sampled full-width windows of an invsqrt .kmx: for each wire w, is its value a GF(2) quadratic
function of the other wires' values (span of 1, q_i, q_i q_j over i, j != w)? Our own sim, sampled shots."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse, Sim, edge_and_random
from itertools import combinations

def in_span(rows, v):
    """rows (k, W) uint64, v (W,) uint64: is v in the GF(2) row span? Gaussian elimination on bit columns."""
    M = np.vstack([rows, v[None]]).copy(); k = M.shape[0] - 1
    r = 0; W = M.shape[1]
    piv_rows = []
    for w in range(W):
        for bt in range(64):
            if r >= k: break
            col = (M[r:k, w] >> np.uint64(bt)) & np.uint64(1)
            nz = np.flatnonzero(col)
            if len(nz) == 0: continue
            p = nz[0] + r
            if p != r: M[[r, p]] = M[[p, r]]
            hit = np.flatnonzero((M[:, w] >> np.uint64(bt)) & np.uint64(1)); hit = hit[hit != r]
            M[hit] ^= M[r]; r += 1
    return not M[k].any()

if __name__ == "__main__":
    path = sys.argv[1]; nwin = int(sys.argv[2]); nshots = int(sys.argv[3])
    nq, body = parse(path); xs = edge_and_random(nshots); sim = Sim(nq, xs, seed=21)
    busy = np.load(path + ".busy.npy"); full = busy.sum(1) == nq
    starts = [i for i in range(1, len(full)) if full[i] and not full[i - 1]]
    pick = set(starts[:: max(1, len(starts) // nwin)][:nwin])
    ones = np.full(sim.W, np.uint64(0xFFFFFFFFFFFFFFFF))
    out = []
    def hook(i, s):
        if i not in pick: return
        Q = s.q.copy()
        hits = []
        for w in range(nq):
            others = [j for j in range(nq) if j != w]
            feats = [ones] + [Q[j] for j in others] + [Q[a] & Q[b] for a, b in combinations(others, 2)]
            if in_span(np.array(feats), Q[w]): hits.append(w)
        out.append((i, hits)); print(i, body[i], "quadratic wires:", hits, flush=True)
    sim.run(body, hook=hook)
    print("windows with at least one quadratic wire:", sum(1 for _, h in out if h), "of", len(out))
