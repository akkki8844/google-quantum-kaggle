"""List top-level PUSH_CONDITION blocks of an invsqrt .kmx with their Toffoli-like count, touched qubits,
nested-condition and HMR use. Usage: python -I blocks.py CIRCUIT.kmx [min_tof]"""
import re, sys, collections
lines = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
mint = int(sys.argv[2]) if len(sys.argv) > 2 else 2
body = [l for l in lines if not l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
depth, start, out = 0, None, []
for i, l in enumerate(body):
    if l.startswith("PUSH_CONDITION"):
        if depth == 0: start = i
        depth += 1
    elif l.startswith("POP_CONDITION"):
        depth -= 1
        if depth == 0:
            blk = body[start:i + 1]
            tof = sum(1 for b in blk if b.split()[0] in ("CCX", "CCZ", "CSWAP"))
            if tof >= mint:
                qs = sorted({int(q) for b in blk for q in re.findall(r"\bq(\d+)", b)})
                nested = sum(1 for b in blk[1:-1] if b.startswith("PUSH_CONDITION"))
                hmr = sum(1 for b in blk if b.startswith("HMR"))
                out.append((start, i, blk[0], tof, len(qs), nested, hmr))
c = collections.Counter(o[3] for o in out)
print("blocks with >=%d toffoli-like:" % mint, len(out), dict(c))
for o in out: print(o)
