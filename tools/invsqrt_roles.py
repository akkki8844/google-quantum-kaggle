"""Reverse-engineer what each qubit of an invsqrt circuit holds over time.

Simulates on a batch of inputs and, at sampled steps, classifies every qubit's value (as a function of x):
  zero        : 0 on all inputs
  const1      : 1 on all inputs
  x_bit[j]    : equals input bit j (or its negation)
  xnorm_bit[j]: equals bit j of the normalized x (x * 4^s, 22 bits)
  s_bit / flag: equals a bit of s (leading zero pairs) or s >= k (thermometer)
  y_bit[j]    : equals bit j of the ideal mantissa y = 2^(s-10)... approximated (unshifted 1/sqrt(M))
  other       : none of the above (intermediate arithmetic)
Prints a compact timeline (one row per sampled step, one character per qubit) plus a legend, so the
register layout and its reuse can be read off.

Usage: python -I tools/invsqrt_roles.py CIRCUIT.kmb [--every 200] [--shots 2048]
"""
import argparse

import numpy as np
import kickmix as km

MAX_X = 3 * (2**10 - 2) ** 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--shots", type=int, default=2048)
    a = ap.parse_args()
    c = km.Circuit.from_file(a.circuit)
    n = c.num_qubits
    lines = [l for l in str(c).splitlines() if l.strip()]
    header = [l for l in lines if l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
    body = [l for l in lines if not l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
    rng = np.random.default_rng(3)
    xs = rng.integers(1, MAX_X + 1, size=a.shots, dtype=np.uint64)
    xi = xs.astype(np.int64)
    s = (22 - (np.floor(np.log2(xi)).astype(np.int64) + 1)) // 2
    xn = xi << (2 * s)
    feats = {}
    for j in range(22):
        feats[f"x{j}"] = (xi >> j) & 1
        feats[f"n{j}"] = (xn >> j) & 1
    for j in range(4):
        feats[f"s{j}"] = (s >> j) & 1
    for k in range(1, 11):
        feats[f"t{k}"] = (s >= k).astype(np.int64)
    # mantissa of 1/sqrt: y = 2^(s-10)/sqrt(x) * ... use 1/sqrt(xn/2^20) in (0.5,1], 28 fractional bits
    y = np.floor(2.0**28 / np.sqrt(xn / 2.0**20)).astype(np.int64)
    for j in range(29):
        feats[f"y{j}"] = (y >> j) & 1
    F = {k: v.astype(bool) for k, v in feats.items()}
    sim = km.Simulator(batch_size=a.shots, debug_print=False, seed=1)
    sim.use_same_registers_as(km.Circuit("\n".join(header)))
    sim.write_across_shots(c.register_data["x"], (xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1 == 1)
    allq = km.array([km.q(i) for i in range(n)])
    legend = {}
    depth, group = 0, []
    print("step  | qubit roles (0=zero 1=one x=input bit n=normalized-x bit s/t=shift info y=mantissa bit .=other)")
    for idx, line in enumerate(body):
        group.append(line)
        depth += line.startswith("PUSH_CONDITION") - line.startswith("POP_CONDITION")
        if depth:
            continue
        if idx % a.every and idx != len(body) - 1:
            sim.do(km.Circuit("\n".join(group)))
            group = []
            continue
        sim.do(km.Circuit("\n".join(group)))
        group = []
        V = sim.read_across_shots(allq)  # qubits x shots
        row = []
        for q in range(n):
            v = V[q]
            if not v.any():
                row.append("0"); continue
            if v.all():
                row.append("1"); continue
            tag = "."
            for k, f in F.items():
                if np.array_equal(v, f) or np.array_equal(v, ~f):
                    tag = k[0]
                    legend.setdefault((idx, q), k)
                    break
            row.append(tag)
        print(f"{idx:6d} | {''.join(row)}")
    print("columns = qubits 0..", n - 1)


if __name__ == "__main__":
    main()
