"""Exact integer model of a piecewise-polynomial invsqrt design, over all x in [1, 3,133,452].

Design family (the public line's structure, parameterised so we can search narrower layouts):
  1. normalize: s = leading zero pairs of x (22-bit), M = x * 4^s in [2^20, 2^22)
  2. segment index: the top `abits` bits of M (the top bit pair is nonzero, so 2 bits encode the
     parity class e and the rest the mantissa); local coordinate u = the next `ubits` bits of M,
     taken as a centered offset in [-1, 1) (all lower bits are dropped: a truncation error the fit absorbs)
  3. y = sum_k c_k[seg] * u^k, degree `deg`, each coefficient a fixed-point integer with `cbits`
     fractional bits, evaluated by Horner with every intermediate truncated to `hbits` fractional bits
  4. phase = y * 2^(s-10-?) turns -> compared with the official tolerance 2^-24 / sqrt(x)
Coefficients are fitted per segment by least-max (minimax via LP on a dense grid of that segment's
actual inputs), then the exact fixed-point evaluation is checked on every x.

Prints, per design: worst error / tolerance over all x, the register widths, and a rough cost model
(lookup Toffolis ~ number of segments, Horner adds ~ ubits per stage).
"""
import argparse
import itertools
import math
import time

import numpy as np
from scipy.optimize import linprog

MAX_X = 3 * (2**10 - 2) ** 2


def normalize(xs: np.ndarray):
    lz = 22 - (np.floor(np.log2(xs)).astype(np.int64) + 1)
    s = lz // 2
    M = xs << (2 * s)
    return s, M


def fit_segment(u: np.ndarray, target: np.ndarray, weight: np.ndarray, deg: int):
    """Minimax fit of target ~ sum c_k u^k with per-point weights (relative error)."""
    A = np.vstack([u**k for k in range(deg + 1)]).T
    # minimize t s.t. |w*(A c - target)| <= t
    W = weight[:, None]
    n = len(u)
    Aub = np.vstack([np.hstack([W * A, -np.ones((n, 1))]), np.hstack([-W * A, -np.ones((n, 1))])])
    bub = np.concatenate([weight * target, -weight * target])
    c = np.zeros(deg + 2)
    c[-1] = 1
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * (deg + 2), method="highs")
    return res.x[:-1], res.x[-1]


def evaluate(design, xs, s, M, target_turns, tol_turns, verbose=False):
    abits, ubits, deg, cbits, hbits = design
    seg = M >> (22 - abits)
    rest_bits = 22 - abits
    uraw = (M >> (rest_bits - ubits)) & ((1 << ubits) - 1)
    # centered u in [-1, 1): u = (2*uraw + 1 - 2^ubits) / 2^ubits
    u_num = 2 * uraw + 1 - (1 << ubits)  # odd integers, scale 2^ubits
    # ideal y in units of 1: y = 2^(10-s) * phase_turns, so that y in (0.5, 2]
    y_ideal = target_turns * np.exp2(10 - s)
    rel_w = 1.0 / y_ideal  # relative weighting
    coefs = {}
    worst_fit = 0.0
    for sg in np.unique(seg):
        idx = np.flatnonzero(seg == sg)
        # dense subsample for the LP
        take = idx if len(idx) <= 3000 else idx[np.linspace(0, len(idx) - 1, 3000).astype(int)]
        cf, t = fit_segment(u_num[take] / 2.0**ubits, y_ideal[take], rel_w[take], deg)
        coefs[int(sg)] = np.round(cf * 2.0**cbits).astype(np.int64)
        worst_fit = max(worst_fit, t)
    # exact fixed-point Horner: acc = c_deg; acc = trunc(acc*u) + c_{k}
    C = np.array([coefs[int(g)] for g in seg])  # (N, deg+1) ints scaled 2^cbits
    acc = C[:, deg] << (hbits - cbits) if hbits >= cbits else C[:, deg] >> (cbits - hbits)
    for k in range(deg - 1, -1, -1):
        prod = acc * u_num  # scale 2^(hbits+ubits)
        acc = np.floor_divide(prod, 1 << ubits)  # truncate back to 2^hbits
        ck = C[:, k] << (hbits - cbits) if hbits >= cbits else C[:, k] >> (cbits - hbits)
        acc = acc + ck
    y = acc / 2.0**hbits
    phase = y * np.exp2(s - 10)
    err = np.abs(phase - target_turns)
    ratio = err / tol_turns
    ymax = float(np.abs(acc).max())
    out = {
        "design(abits,ubits,deg,cbits,hbits)": design,
        "segments": len(coefs),
        "worst_err_over_tol": float(ratio.max()),
        "worst_x": int(xs[np.argmax(ratio)]),
        "fit_rel_err": worst_fit,
        "acc_bits": int(math.ceil(math.log2(ymax + 1))) + 1,
        "coef_bits_max": int(max(int(np.abs(v).max()).bit_length() for v in coefs.values())) + 1,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=MAX_X)
    a = ap.parse_args()
    t0 = time.time()
    xs = np.arange(1, a.limit + 1, dtype=np.int64)
    s, M = normalize(xs)
    target = 1.0 / np.sqrt(xs.astype(np.float64))  # turns
    tol = 2.0**-24 / np.sqrt(xs.astype(np.float64))
    designs = []
    # u always uses every remaining bit (no truncation); sweep table size, degree and widths
    for abits, deg in [(7, 2), (8, 2), (5, 3), (6, 3)]:
        for w in (26, 27, 28):
            designs.append((abits, 22 - abits, deg, w, w))
    for d in designs:
        r = evaluate(d, xs, s, M, target, tol)
        print(r, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
