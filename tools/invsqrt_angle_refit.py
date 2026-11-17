"""Refit the fine Z_POW angles of an invsqrt circuit to maximize the minimum phase margin.

Rotation cost in the verifier is n_rot * (0.56*log2(1/eps) + 5.3) T with eps = pi*min_margin/n_rot,
where min_margin = min over x of tol(x) - |err(x)| (turns). The fine angles are free parameters, so we
choose them to maximize min_margin. Steps:
  1. Sweep all x with the fine rotations removed           -> base phase(x)
  2. Sweep all x with rotation k set to tag 2^-(30+k) turns -> which rotations fire, b_k(x) (exact)
  3. LP (cutting planes, HiGHS): maximize m s.t. |err0(x) + sum_k d_k b_k(x)| <= tol(x) - m
  4. Write the circuit with the new angles; confirm with tools/invsqrt_fastcheck.py.

Usage: python -I tools/invsqrt_angle_refit.py IN.kmb OUT.kmb [--batch 4096]
"""
import argparse
import math
import re
import time
from fractions import Fraction

import numpy as np
import kickmix as km
from scipy.optimize import linprog

MAX_X = 3 * (2**10 - 2) ** 2
MASK64 = (1 << 64) - 1
UNIT = 2.0**-64  # phase unit in turns


def reference(xs):
    rs = [math.isqrt((1 << 130) // int(x)) for x in xs]
    target = np.array([((r + 1) >> 1) & MASK64 for r in rs], dtype=np.uint64)
    tol = np.array([min(r >> 25, 1 << 63) for r in rs], dtype=np.uint64)
    return target, tol


def units(h) -> int:
    n, d = h.as_integer_ratio()
    return (((n << 64) // d + 1) >> 1) & MASK64


def dec(fr: Fraction) -> str:
    """Exact decimal for a dyadic Fraction (kmx wants plain decimals)."""
    n, d = fr.numerator, fr.denominator
    k = d.bit_length() - 1
    assert d == 1 << k
    s = str(n * 5**k).rjust(k + 1, "0")
    out = (s[:-k] or "0") + "." + s[-k:] if k else s + ".0"
    return out.rstrip("0").rstrip(".") if "." in out and out.rstrip("0").rstrip(".") else out


def sweep(circ: km.Circuit, xs: np.ndarray, batch: int) -> np.ndarray:
    x_reg = circ.register_data["x"]
    sim = km.Simulator(batch_size=batch, debug_print=False, seed=2026)
    out = np.zeros(len(xs), dtype=np.uint64)
    for s in range(0, len(xs), batch):
        b = xs[s:s + batch]
        n = len(b)
        if n < batch:
            b = np.concatenate([b, np.full(batch - n, b[-1], dtype=np.uint64)])
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, (b[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1 == 1)
        sim.do(circ)
        out[s:s + n] = [units(sim.read_shot_phase(i)) for i in range(n)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=4096)
    a = ap.parse_args()
    t0 = time.time()
    circ = km.Circuit.from_file(a.inp)
    lines = str(circ).splitlines()
    fine = []
    for i, l in enumerate(lines):
        m = re.fullmatch(r"Z_POW (q\d+) ([0-9.]+)", l.strip())
        if m:
            e = Fraction(m.group(2))
            if (e * 4).denominator != 1:  # not a multiple of 1/4 half-turn -> fine rotation
                fine.append((i, m.group(1), e))
    K = len(fine)
    print(f"{K} fine rotations; current sum check ok", flush=True)

    def variant(angles: list[Fraction | None]) -> km.Circuit:
        L = list(lines)
        for (i, q, _), ang in zip(fine, angles):
            L[i] = "" if ang is None or ang == 0 else f"Z_POW {q} {dec(ang)}"
        return km.Circuit("\n".join(x for x in L if x))

    xs = np.arange(1, MAX_X + 1, dtype=np.uint64)
    target, tol = reference(xs)
    print(f"reference ready {time.time() - t0:.0f}s", flush=True)
    base = sweep(variant([None] * K), xs, a.batch)
    print(f"base sweep {time.time() - t0:.0f}s", flush=True)
    # tag k: 2^-(30+k) turns = 2^-(29+k) half-turns
    tagged = sweep(variant([Fraction(1, 2 ** (29 + k)) for k in range(K)]), xs, a.batch)
    print(f"tag sweep {time.time() - t0:.0f}s", flush=True)
    diff = (tagged - base)  # uint64 wraps correctly
    B = np.zeros((len(xs), K), dtype=np.uint8)
    for k in range(K):
        B[:, k] = (diff >> np.uint64(34 - k)) & np.uint64(1)
    recon = (B.astype(np.uint64) << (np.uint64(34) - np.arange(K, dtype=np.uint64))).sum(axis=1)
    assert np.array_equal(recon, diff), "tag decode mismatch (rotation bits not clean)"
    cur = np.array([float(f[2]) / 2 for f in fine])  # current angles in turns
    # err0(x) in turns relative to target, with the current angles
    phase_cur_units = base.astype(np.float64)  # build err in units of 2^-64 turns, signed
    err0 = ((base - target).view(np.int64)).astype(np.float64) + (B @ (cur / UNIT))
    # wrap into (-2^63, 2^63]
    err0 = (err0 + 2.0**63) % 2.0**64 - 2.0**63
    tolf = tol.astype(np.float64)
    m0 = float((tolf - np.abs(err0)).min())
    print(f"current min margin {m0 * UNIT:.4e} turns (eps {math.pi * m0 * UNIT / K:.3e})", flush=True)

    S = 2.0**30  # LP scaling
    active = np.argsort(tolf - np.abs(err0))[:4000]
    best = None
    for it in range(30):
        Bi = B[active].astype(np.float64)
        e = err0[active] / S
        t = tolf[active] / S
        # vars: d_0..d_{K-1} (units 2^-64/S), m ; maximize m  -> minimize -m
        A_ub = np.vstack([np.hstack([Bi, np.ones((len(active), 1))]),
                          np.hstack([-Bi, np.ones((len(active), 1))])])
        b_ub = np.concatenate([t - e, t + e])
        c = np.zeros(K + 1)
        c[-1] = -1
        bound = 2.0**44 / S  # angle changes up to ~2^-20 turns
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(-bound, bound)] * K + [(None, None)], method="highs")
        d = res.x[:K] * S
        err = err0 + B @ d
        marg = tolf - np.abs(err)
        mfull = float(marg.min())
        print(f"  iter {it}: LP m {res.x[-1] * S * UNIT:.4e}, full m {mfull * UNIT:.4e}, active {len(active)}", flush=True)
        best = (mfull, d)
        viol = np.flatnonzero(marg < res.x[-1] * S * (1 - 1e-9))
        if len(viol) == 0:
            break
        active = np.unique(np.concatenate([active, viol[np.argsort(marg[viol])[:4000]]]))
    mfull, d = best
    # Back off a little from the LP optimum to keep exact rounding safe.
    new = [Fraction(int(round((cur[k] + d[k] * UNIT) * 2**70)), 2**70) * 2 for k in range(K)]  # half-turns
    out = variant(new)
    out.to_file(a.out, format="kmb" if a.out.endswith(".kmb") else "kmx")
    print(f"min margin {m0 * UNIT:.4e} -> {mfull * UNIT:.4e} turns; eps {math.pi * m0 * UNIT / K:.3e} -> "
          f"{math.pi * mfull * UNIT / K:.3e}; rotation T/rot {0.56 * math.log2(K / (math.pi * m0 * UNIT)) + 5.3:.2f} -> "
          f"{0.56 * math.log2(K / (math.pi * mfull * UNIT)) + 5.3:.2f}; wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
