"""Fast exact local check of an invsqrt circuit over ALL x (1 shot per x by default).

Mirrors the official check (x restored, scratch clean, phase within tol(x) in 2^-64 units) but runs
single-process in large batches, so a full sweep takes minutes on a laptop. It is a pre-check: the
official verifier also runs 64 shots per x with seeded HMR outcomes; use --shots to add more.
Reports: pass/fail, first failure, min margin, mean Toffoli-like / T / rotation counts per shot, and
the cost Q * (Toffoli + (T + rotation T)/4) with the official rotation pricing.

Usage:  python -I tools/invsqrt_fastcheck.py CIRCUIT.kmb [--batch 4096] [--shots 1] [--limit N]
"""
import argparse
import math
import time

import numpy as np
import kickmix as km

MAX_X = 3 * (2**10 - 2) ** 2
MASK64 = (1 << 64) - 1


def reference(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rs = [math.isqrt((1 << 130) // int(x)) for x in xs]
    target = np.array([((r + 1) >> 1) & MASK64 for r in rs], dtype=np.uint64)
    tol = np.array([min(r >> 25, 1 << 63) for r in rs], dtype=np.uint64)
    return target, tol


def units(h) -> int:
    n, d = h.as_integer_ratio()
    return (((n << 64) // d + 1) >> 1) & MASK64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--limit", type=int, default=MAX_X)
    a = ap.parse_args()
    c = km.Circuit.from_file(a.circuit)
    x_reg = c.register_data["x"]
    xset = set(x_reg)
    scratch = km.array([km.q(i) for i in range(c.num_qubits) if km.q(i) not in xset])
    t0 = time.time()
    print("precomputing reference values...", flush=True)
    allx = np.arange(1, a.limit + 1, dtype=np.uint64)
    target, tol = reference(allx)
    print(f"  {time.time() - t0:.0f}s", flush=True)
    sim = km.Simulator(batch_size=a.batch, debug_print=False, counting_operations=True, seed=2026)
    min_margin, worst = 1 << 64, (0.0, 0)
    shots = 0
    for rep in range(a.shots):
        for start in range(0, a.limit, a.batch):
            xs = allx[start:start + a.batch]
            n = len(xs)
            if n < a.batch:
                xs = np.concatenate([xs, np.full(a.batch - n, xs[-1], dtype=np.uint64)])
            sim.clear_for_shot()
            sim.write_across_shots(x_reg, (xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1 == 1)
            sim.do(c)
            back = (sim.read_across_shots(x_reg).astype(np.uint64) << np.arange(22, dtype=np.uint64)[:, None]).sum(0)
            if not np.array_equal(back[:n], xs[:n]):
                print("FAIL x not restored at", xs[np.flatnonzero(back[:n] != xs[:n])[0]]); return
            if len(scratch) and sim.read_across_shots(scratch)[:, :n].any():
                bad = np.flatnonzero(sim.read_across_shots(scratch)[:, :n].any(0))[0]
                print("FAIL dirty scratch at x =", int(xs[bad])); return
            th = np.array([units(sim.read_shot_phase(i)) for i in range(n)], dtype=np.uint64)
            idx = xs[:n] - np.uint64(1)
            err = (th - target[idx]).view(np.int64)
            used = np.abs(err).view(np.uint64)
            allowed = tol[idx]
            badm = np.flatnonzero(used > allowed)
            if len(badm):
                i = badm[0]
                print(f"FAIL phase at x={int(xs[i])}: err {err[i] / 2**64:.3e} > tol {allowed[i] / 2**64:.3e}"); return
            min_margin = min(min_margin, int((allowed - used).min()))
            r = (used / allowed.astype(np.float64)).max()
            if r > worst[0]:
                worst = (float(r), int(xs[int(np.argmax(used / allowed.astype(np.float64)))]))
            shots += n
            if (start // a.batch) % 100 == 0:
                print(f"  rep {rep} x up to {start + n:,} ok, {time.time() - t0:.0f}s", flush=True)
    st = sim.operation_statistics()
    # Counts accumulate over all batches including padding; scale by simulated shots.
    total_sim = a.batch * math.ceil(a.limit / a.batch) * a.shots
    tof = st.num_toffoli_like / total_sim
    tcount = st.num_t_like / total_sim
    nrot = st.num_z_pow(4) / total_sim
    eps = math.pi * min_margin / 2**64 / nrot if nrot else None
    trot = nrot * (0.56 * math.log2(1 / eps) + 5.3) if nrot else 0.0
    C = tof + (tcount + trot) / 4
    print(f"PASS {shots:,} shots in {time.time() - t0:.0f}s | Q {c.num_qubits} | Toffoli {tof:.3f} | T {tcount:.3f} | "
          f"rot {nrot:.2f} eps {eps:.3e} | C {C:.3f} | cost {c.num_qubits * C:,.2f} | worst err/tol {worst[0]:.5f} at x={worst[1]}")


if __name__ == "__main__":
    main()
