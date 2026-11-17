"""Find redundant qubits in an invsqrt circuit: GF(2) linear dependencies among live qubit values.

Simulates the circuit one balanced instruction group at a time on a batch of inputs and, after each
group, takes the matrix V (shots x qubits) of qubit values. Over GF(2) with an all-ones column
appended (constants), rank(V_live) < number of live qubits means some live qubit equals an XOR of
others (plus a constant) on every sampled input: it could be erased and rebuilt with CNOTs only.

Reports, for the steps where the live count equals the circuit's peak, the GF(2) rank deficiency,
and the qubits involved in a dependency. A qubit freed across every peak window lowers Q by one.

Usage: python -I tools/invsqrt_dependency.py CIRCUIT.kmb [--shots 4096] [--out deps.json]
"""
import argparse
import json

import numpy as np
import kickmix as km

MAX_X = 3 * (2**10 - 2) ** 2


def gf2_rank_and_null(M: np.ndarray) -> tuple[int, list[np.ndarray]]:
    """Rank of a 0/1 matrix (rows = samples, cols = variables) and a basis of column dependencies."""
    A = M.T.copy().astype(np.uint8)  # variables x samples: find dependencies among rows of A
    nvar = A.shape[0]
    # Track combinations: augment with identity.
    comb = np.eye(nvar, dtype=np.uint8)
    rank = 0
    pivots = []
    rows = list(range(nvar))
    A = A.copy()
    col = 0
    r = 0
    ncols = A.shape[1]
    while r < nvar and col < ncols:
        piv = None
        for i in range(r, nvar):
            if A[i, col]:
                piv = i
                break
        if piv is None:
            col += 1
            continue
        A[[r, piv]] = A[[piv, r]]
        comb[[r, piv]] = comb[[piv, r]]
        for i in range(nvar):
            if i != r and A[i, col]:
                A[i] ^= A[r]
                comb[i] ^= comb[r]
        r += 1
        col += 1
    rank = r
    deps = [comb[i] for i in range(rank, nvar)]  # rows reduced to zero -> dependencies
    return rank, deps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    c = km.Circuit.from_file(a.circuit)
    n = c.num_qubits
    lines = [l for l in str(c).splitlines() if l.strip()]
    header = [l for l in lines if l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
    body = [l for l in lines if not l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
    rng = np.random.default_rng(5)
    xs = np.concatenate([np.array([1, 2, 3, 231, MAX_X], dtype=np.uint64),
                         rng.integers(1, MAX_X + 1, size=a.shots - 5, dtype=np.uint64)])
    sim = km.Simulator(batch_size=a.shots, debug_print=False, seed=11)
    sim.use_same_registers_as(km.Circuit("\n".join(header)))
    sim.write_across_shots(c.register_data["x"], (xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1 == 1)
    allq = km.array([km.q(i) for i in range(n)])
    records = []
    depth = 0
    group = []
    gi = 0
    for idx, line in enumerate(body):
        group.append(line)
        depth += line.startswith("PUSH_CONDITION") - line.startswith("POP_CONDITION")
        if depth:
            continue
        sim.do(km.Circuit("\n".join(group)))
        group = []
        V = sim.read_across_shots(allq).T  # shots x qubits (bool)
        live = np.flatnonzero(V.any(axis=0))
        if len(live) >= n - 1:  # only analyse near-peak steps (cost)
            Vl = np.hstack([V[:, live].astype(np.uint8), np.ones((V.shape[0], 1), dtype=np.uint8)])
            rank, deps = gf2_rank_and_null(Vl)
            deficiency = Vl.shape[1] - rank
            involved = sorted({int(live[j]) for d in deps for j in np.flatnonzero(d[:-1])})
            records.append({"step": idx, "live": int(len(live)), "deficiency": int(deficiency),
                            "dep_qubits": involved[:20]})
    peak = max(r["live"] for r in records)
    at_peak = [r for r in records if r["live"] == peak]
    no_dep = [r for r in at_peak if r["deficiency"] == 0]
    print(json.dumps({"peak_live": peak, "steps_at_peak": len(at_peak),
                      "peak_steps_with_a_linear_dependency": len(at_peak) - len(no_dep),
                      "peak_steps_without": len(no_dep),
                      "first_steps_without": [r["step"] for r in no_dep[:15]]}, indent=1))
    if a.out:
        json.dump(records, open(a.out, "w"))


if __name__ == "__main__":
    main()
