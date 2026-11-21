"""Width profile of an invsqrt .kmx: per line, which qubits hold a nonzero value in any sampled shot (our own sim)."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\smart\Desktop\googlequantum\research\inv_cross")
from bitsim import parse, Sim, edge_and_random
path = sys.argv[1]
nq, body = parse(path)
xs = edge_and_random(int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
sim = Sim(nq, xs, seed=3)
L = len(body)
busy = np.zeros((L + 1, nq), dtype=bool)
def hook(i, s):
    busy[i] = (s.q != 0).any(axis=1)
sim.run(body, hook=hook)
busy[L] = (sim.q != 0).any(axis=1)
x0 = Sim(nq, xs).q[:22]
print("nq", nq, "lines", L, "shots", sim.S, "x restored", bool((sim.q[:22] == x0).all()), "scratch clean", bool((sim.q[22:] == 0).all()),
      "mean tof/shot", sim.tof / sim.S)
cnt = busy.sum(axis=1)
np.save(path + ".busy.npy", busy)
print("max busy", cnt.max(), "lines at max", int((cnt == cnt.max()).sum()), "at max-1", int((cnt == cnt.max() - 1).sum()))
# windows at full width
full = cnt == nq
runs, i = [], 0
while i <= L:
    if full[i]:
        j = i
        while j <= L and full[j]: j += 1
        runs.append((i, j)); i = j
    else:
        i += 1
print("full-width windows", len(runs))
for a, b in runs[:80]:
    print(a, b, b - a)
