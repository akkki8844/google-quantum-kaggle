"""Scan a kickmix .kmx body for Toffoli 'dead writes' and other measure-first motifs.

Motif A (dead write): `CCX a b t` where the next instruction touching t is `HMR t bK`
(t is not read in between). Replacing the CCX by `HMR t bK` + `CZ a b` conditioned on bK
is exact and saves one Toffoli per shot (the CZ is Clifford).
Motif B: `CCX a b t` followed only by phase uses of t (Z/CZ/CCZ with t) before `HMR t` —
reported for manual review (needs the phase uses rewritten).

Instructions inside PUSH_CONDITION blocks are handled conservatively: a write inside a
conditioned block is skipped.

Usage: python tools/dead_write_scan.py BODY.kmx
"""
import re
import sys

READS = {"CX": [0], "CZ": [0, 1], "CCX": [0, 1], "CCZ": [0, 1, 2], "Z": [0], "Z_POW": [0],
         "SWAP": [0, 1], "X": [], "HMR": [0]}


def qubits(tokens):
    return [t for t in tokens if re.fullmatch(r"q\d+", t)]


def main() -> None:
    lines = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith(("REGISTER", "APPEND"))]
    depth = 0
    found_a, found_b = [], []
    for i, l in enumerate(lines):
        tok = l.split()
        if tok[0] == "PUSH_CONDITION":
            depth += 1
        elif tok[0] == "POP_CONDITION":
            depth -= 1
        if tok[0] != "CCX" or depth:
            continue
        a, b, t = tok[1:4]
        phase_uses = []
        d2 = 0
        for j in range(i + 1, len(lines)):
            tj = lines[j].split()
            if tj[0] == "PUSH_CONDITION":
                d2 += 1
                continue
            if tj[0] == "POP_CONDITION":
                d2 -= 1
                continue
            qs = qubits(tj)
            if t not in qs:
                # a or b changing before the HMR would break the CZ rewrite only if we move it; we don't.
                continue
            op = tj[0]
            if op == "HMR" and qs[0] == t:
                if not phase_uses:
                    found_a.append((i, j, l, lines[j]))
                else:
                    found_b.append((i, j, l, phase_uses))
                break
            if op in ("CZ", "CCZ", "Z", "Z_POW") and d2 == 0:
                phase_uses.append((j, lines[j]))
                continue
            break
    print(f"motif A (dead write): {len(found_a)}")
    for i, j, l, h in found_a[:50]:
        print(f"  op {i}: {l}   ->  op {j}: {h}")
    print(f"motif B (phase-only reads before HMR): {len(found_b)}")
    for i, j, l, pu in found_b[:30]:
        print(f"  op {i}: {l}  phase uses {[p[1] for p in pu]}  HMR at {j}")


if __name__ == "__main__":
    main()
