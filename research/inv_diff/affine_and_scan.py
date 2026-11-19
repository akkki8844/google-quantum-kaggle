"""Scan every CCX / CCZ (top level and inside PUSH_CONDITION blocks) of an invsqrt circuit for a
pair product that equals an affine (GF(2)) function of the live qubits at that moment.
  CCX a b t with ab = L (L not using t)  ->  CX q t for q in L (+ X t if const): 0 Toffolis.
  CCZ a b c with ab = L                   ->  CZ q c for q in L (q == c -> Z c; const -> Z c): 0 Toffolis.
Conditioned gates count 0.5 on average, unconditioned 1. Sampled evidence; confirm with the all-x check.
Usage: python -I affine_and_scan.py CIRCUIT.kmx [--shots 2048]"""
import argparse, json, re, sys
import numpy as np
import kickmix as km
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from mbu_scan import gf2_solve

MAX_X = 3 * (2**10 - 2) ** 2


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
    rng = np.random.default_rng(5)
    p2 = sorted({v for k in range(23) for v in (2**k - 1, 2**k, 2**k + 1) if 1 <= v <= MAX_X} | {MAX_X, MAX_X - 1})
    edge = np.array(sorted(set(range(1, 4097)) | set(p2) | {4 * v for v in range(1, 1024)}), dtype=np.uint64)
    edge = edge[edge <= MAX_X]
    xs = np.concatenate([edge, rng.integers(1, MAX_X + 1, size=max(0, a.shots - len(edge)), dtype=np.uint64)])
    a.shots = len(xs)
    sim = km.Simulator(batch_size=a.shots, debug_print=False, seed=13)
    sim.use_same_registers_as(km.Circuit("\n".join(header)))
    sim.write_across_shots(c.register_data["x"], (xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1 == 1)
    allq = km.array([km.q(i) for i in range(n)])
    pat = re.compile(r"^(CCX|CCZ) q(\d+) q(\d+) q(\d+)$")
    cands, depth, group = [], 0, []
    ones = np.ones(a.shots, dtype=bool)
    for idx, line in enumerate(body):
        m = pat.match(line.strip())
        if line.startswith("PUSH_CONDITION") and depth == 0:
            st_block = sim.read_across_shots(allq)
        if m:
            st = sim.read_across_shots(allq) if not group else st_block
            g, qa, qb, qc = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            pairs = [(qa, qb, qc)] if g == "CCX" else [(qa, qb, qc), (qa, qc, qb), (qb, qc, qa)]
            for x1, x2, rest in pairs:
                prod = st[x1] & st[x2]
                others = [q for q in range(n) if not (g == "CCX" and q == rest)]
                sol = gf2_solve(np.vstack([st[others], ones]), prod)
                if sol is not None:
                    cands.append({"line": idx, "gate": line.strip(), "cond_depth": depth,
                                  "pair": [x1, x2], "other": rest,
                                  "terms": [others[i] for i in range(len(others)) if sol[i]], "const": bool(sol[-1])})
                    break
        if depth == 0 and not line.startswith("PUSH_CONDITION"):
            sim.do(km.Circuit(line))
            continue
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
