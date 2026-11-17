"""Exact qubit-liveness profile of an invsqrt circuit, measured by simulation.

Runs the circuit one instruction at a time in kickmix on a batch of inputs and records, after
each instruction, which qubits are nonzero in any shot. A qubit is "live" at step i if it is
nonzero in some shot after instruction i. The peak live count is what bounds Q; the profile
shows how wide the peak is and which steps hold it.

Usage:  python -I tools/liveness.py CIRCUIT.kmb|kmx [--shots 1024] [--out prof.json]
"""
import argparse
import json
import sys

import numpy as np
import kickmix as km

MAX_X = 3 * (2**10 - 2) ** 2


def instructions(circuit: km.Circuit) -> tuple[list[str], list[str]]:
    lines = [l for l in str(circuit).splitlines() if l.strip() and not l.lstrip().startswith("#")]
    header = [l for l in lines if l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
    body = [l for l in lines if not l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
    return header, body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = km.Circuit.from_file(a.circuit)
    n = c.num_qubits
    header, body = instructions(c)
    rng = np.random.default_rng(1)
    edge = np.array([1, 2, 3, 4, MAX_X, MAX_X - 1, 2**21, 2**21 - 1, 2**20, 2**20 + 1], dtype=np.uint64)
    xs = np.concatenate([edge, rng.integers(1, MAX_X + 1, size=a.shots - len(edge), dtype=np.uint64)])

    sim = km.Simulator(batch_size=a.shots, debug_print=False, seed=7)
    head = km.Circuit("\n".join(header))
    sim.use_same_registers_as(head)
    x = c.register_data["x"]
    bits = (xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1 == 1
    sim.write_across_shots(x, bits)
    allq = km.array([km.q(i) for i in range(n)])

    live = np.zeros((len(body), n), dtype=bool)
    # Execute in small groups; PUSH/POP blocks are kept whole so conditions stay balanced.
    i = 0
    depth = 0
    group: list[str] = []
    group_start = 0
    for idx, line in enumerate(body):
        group.append(line)
        if line.startswith("PUSH_CONDITION"):
            depth += 1
        elif line.startswith("POP_CONDITION"):
            depth -= 1
        if depth == 0:
            sim.do(km.Circuit("\n".join(group)))
            nz = sim.read_across_shots(allq).any(axis=1)
            live[group_start:idx + 1] = nz
            group, group_start = [], idx + 1
    counts = live.sum(axis=1)
    peak = int(counts.max())
    hist = {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    at_peak = np.flatnonzero(counts == peak)
    # Per-qubit: fraction of steps live.
    frac = live.mean(axis=0)
    res = {
        "circuit": a.circuit, "num_qubits": n, "instructions": len(body), "peak_live": peak,
        "steps_at_peak": int(len(at_peak)), "first_peak_step": int(at_peak[0]), "last_peak_step": int(at_peak[-1]),
        "histogram_live_count_to_steps": hist,
        "peak_windows": [[int(w[0]), int(w[-1])] for w in np.split(at_peak, np.flatnonzero(np.diff(at_peak) > 1) + 1)][:40],
        "qubit_live_fraction": {int(q): round(float(f), 4) for q, f in enumerate(frac)},
    }
    print(json.dumps({k: v for k, v in res.items() if k != "qubit_live_fraction"}, indent=1))
    if a.out:
        json.dump(res | {"counts": counts.tolist()}, open(a.out, "w"))


if __name__ == "__main__":
    main()
