"""Classify full-width windows of an invsqrt .kmx: for each window, the wire(s) that became busy at its opener and
how that wire is used from its creation to its release (ops touching it, roles)."""
import sys, collections, numpy as np
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse
path = sys.argv[1]
nq, body = parse(path); busy = np.load(path + ".busy.npy"); L = len(body)
full = busy.sum(1)[:L + 1] == nq
runs = []; i = 0
while i <= L:
    if full[i]:
        j = i
        while j <= L and full[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
ROLE = {"CX": "cw", "CCX": "ccw", "CZ": "pp", "CCZ": "ppp", "Z": "p", "X": "w", "SWAP": "ww", "CSWAP": "cww", "HMR": "e"}
def uses(w, a, b):
    out = []
    for k in range(a, min(b, L)):
        op, ar, c = body[k]
        qs = [x for x in ar if isinstance(x, int)]
        if w in qs:
            r = ROLE.get(op, "?")[qs.index(w)] if op in ROLE else "?"
            out.append((k, op, r, c is not None))
    return out
cls = collections.Counter(); ex = {}
for a, b in runs:
    new = [w for w in range(nq) if busy[a, w] and not busy[a - 1, w]]
    if not new: cls["no-new"] += 1; continue
    w = new[0]
    # lifetime of w: from a-1 back to creation (= a-1) to the first line where w is free again
    e = a
    while e <= L and busy[e, w]: e += 1
    u = uses(w, a - 1, e)
    roles = "".join(r for _, _, r, _ in u)
    # summarize: created by (first op), used as control c / phase p / written w, erased e
    key = (u[0][1] if u else "?", "c" in roles[1:], "p" in roles[1:], "w" in roles[1:] or "cw" in roles[1:], u[-1][1] if u else "?", e - a + 1)
    k2 = (key[0], "ctrl" if key[1] else "-", "phase" if key[2] else "-", "write" if key[3] else "-", key[4])
    cls[k2] += 1; ex.setdefault(k2, (a, b, w, u[:12]))
for k, v in cls.most_common(): print(v, k)
print()
for k, (a, b, w, u) in ex.items(): print(k, "window", (a, b), "wire", w, u)
