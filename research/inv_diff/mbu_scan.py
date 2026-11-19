"""Scan an invsqrt circuit for unconditional Toffolis that only undo an AND.

For each top-level CCX a b t: if, just before it, t == a & b on every sampled shot (so the gate
returns t to 0), it can be replaced by measurement-based uncompute (HMR t -> m; if m: CZ a b), which
costs 0 Toffolis instead of 1. Also reports CCX where t == a & b XOR 1 (t ends at 1) and CCZ-free
variants. Sampled evidence only: every candidate must then pass the exact all-x check.

Usage: python -I research/inv_diff/mbu_scan.py CIRCUIT.kmx [--shots 2048]
"""
import argparse, json, re
import numpy as np
import kickmix as km

MAX_X = 3 * (2**10 - 2) ** 2


def gf2_solve(A, y):
    """Solve sum_i x_i A[i] = y over GF(2) (rows of A are vectors over shots). Return x or None."""
    A = A.copy(); y = y.copy(); k = A.shape[0]
    M = np.vstack([A, y[None, :]]).T.astype(np.uint8)  # shots x (k+1)
    piv_cols, r = [], 0
    for c in range(k):
        rows = np.flatnonzero(M[r:, c]) + r
        if len(rows) == 0:
            continue
        M[[r, rows[0]]] = M[[rows[0], r]]
        hit = np.flatnonzero(M[:, c]); hit = hit[hit != r]
        M[hit] ^= M[r]
        piv_cols.append(c); r += 1
        if r == M.shape[0]:
            break
    if M[r:, k].any():
        return None
    x = np.zeros(k, dtype=bool)
    for i, c in enumerate(piv_cols):
        x[c] = bool(M[i, k])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--shots", type=int, default=2048)
    a = ap.parse_args()
    c = km.Circuit.from_file(a.circuit)
    n = c.num_qubits
    lines = [l for l in str(c).splitlines() if l.strip() and not l.lstrip().startswith("#")]
    header = [l for l in lines if l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
    body = [l for l in lines if not l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
    rng = np.random.default_rng(3)
    p2 = sorted({v for k in range(23) for v in (2**k - 1, 2**k, 2**k + 1) if 1 <= v <= MAX_X} | {MAX_X, MAX_X - 1})
    edge = np.array(sorted(set(range(1, 4097)) | set(p2) | {4 * v for v in range(1, 1024)}), dtype=np.uint64)
    edge = edge[edge <= MAX_X]
    xs = np.concatenate([edge, rng.integers(1, MAX_X + 1, size=max(0, a.shots - len(edge)), dtype=np.uint64)])
    a.shots = len(xs)
    sim = km.Simulator(batch_size=a.shots, debug_print=False, seed=11)
    sim.use_same_registers_as(km.Circuit("\n".join(header)))
    sim.write_across_shots(c.register_data["x"], (xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1 == 1)
    allq = km.array([km.q(i) for i in range(n)])
    cands = []
    depth, group = 0, []
    pat = re.compile(r"^CCX q(\d+) q(\d+) q(\d+)$")
    for idx, line in enumerate(body):
        if depth == 0 and not group:
            m = pat.match(line.strip())
            if m:
                st = sim.read_across_shots(allq)
                qa, qb, qt = (int(v) for v in m.groups())
                ab = st[qa] & st[qb]
                if np.array_equal(st[qt], ab):
                    cands.append({"line": idx, "gate": line.strip(), "kind": "t==ab (free MBU)"})
                elif np.array_equal(st[qt], ~ab):
                    cands.append({"line": idx, "gate": line.strip(), "kind": "t==~ab"})
                else:
                    after = st[qt] ^ ab                    # value t holds after the gate
                    others = [q for q in range(n) if q != qt]
                    sol = gf2_solve(np.vstack([st[others], np.ones(st.shape[1], dtype=bool)]), after)
                    if sol is not None:
                        used = [others[i] for i in range(len(others)) if sol[i]]
                        cands.append({"line": idx, "gate": line.strip(), "kind": "after = affine",
                                      "terms": used, "const": bool(sol[-1])})
        group.append(line)
        if line.startswith("PUSH_CONDITION"):
            depth += 1
        elif line.startswith("POP_CONDITION"):
            depth -= 1
        if depth == 0:
            sim.do(km.Circuit("\n".join(group)))
            group = []
    print(json.dumps({"instructions": len(body), "candidates": len(cands)}))
    for cd in cands:
        print(cd)


if __name__ == "__main__":
    main()
