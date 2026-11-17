"""Workstream A, step 1: measure the frontier chemistry angle loaders.

For each ANGLE_k circuit: qubits, static Toffoli-like gates, and HMR count.
For each circuit's target function (the 40 output bits of delta[k] over the selection register):
  - ANF size (number of monomials) and degree per output bit, with the 828 unused addresses
    filled with 0 (no don't-care use), and the number of distinct monomials across all 40 bits.
That tells us how much structure the loaders already exploit and what a don't-care-aware
synthesizer could start from.

Usage:  python -I tools/chem_loader_stats.py DFTHCSOS_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import kickmix as km


def batch_deltas(words: np.ndarray, J: int) -> np.ndarray:
    n_rows, n_ang = words.shape
    K = -(-n_ang // J)
    pad = np.zeros((n_rows, K * J), dtype=np.int64)
    pad[:, :n_ang] = words
    b = pad.reshape(n_rows, K, J).transpose(1, 0, 2)
    d = b.copy()
    d[1:] ^= b[:-1]
    return d  # (K, rows, J)


def anf(truth: np.ndarray) -> np.ndarray:
    """Moebius transform: truth table (2^n,) of 0/1 -> ANF coefficients (2^n,)."""
    a = truth.astype(np.uint8).copy()
    n = int(np.log2(len(a)))
    for i in range(n):
        step = 1 << i
        a = a.reshape(-1, 2 * step)
        a[:, step:] ^= a[:, :step]
        a = a.reshape(-1)
    return a


def main() -> None:
    d = Path(sys.argv[1])
    sub = np.load(d / "submission.npz", allow_pickle=False)
    words = sub["angle_words"]
    rows = words.shape[0]
    nbits = int(np.ceil(np.log2(rows)))
    circuits = sorted(d.glob("angle_k_*.kmb"), key=lambda p: int(p.stem.split("_")[-1]))
    c0 = km.Circuit.from_file(str(circuits[0]))
    J = sum(1 for k in c0.register_data if str(k).startswith("word_"))
    brot = len(c0.register_data["word_0"])
    deltas = batch_deltas(words, J)
    print(f"rows={rows} selection_bits={nbits} J={J} brot={brot} K={len(circuits)} (deltas {deltas.shape})")
    out = []
    tot_tof = 0
    for k, path in enumerate(circuits):
        c = km.Circuit.from_file(str(path))
        st = c.operation_statistics()
        tof = st.num_toffoli_like
        tot_tof += tof
        # ANF per output bit with unused addresses = 0.
        mons_all = set()
        sizes, degs = [], []
        for j in range(J):
            for b in range(brot):
                tt = np.zeros(1 << nbits, dtype=np.uint8)
                tt[:rows] = (deltas[k, :, j] >> b) & 1
                a = anf(tt)
                idx = np.flatnonzero(a)
                sizes.append(len(idx))
                degs.append(max((int(bin(i).count("1")) for i in idx), default=0))
                mons_all.update(int(i) for i in idx if bin(i).count("1") >= 2)
        nonlinear = len(mons_all)
        out.append(dict(k=k, qubits=c.num_qubits, toffoli_static=int(tof), hmr=int(dict(st.items()).get("HMR", 0)),
                        anf_terms_mean=float(np.mean(sizes)), anf_degree_max=int(max(degs)),
                        distinct_nonlinear_monomials=nonlinear,
                        zero_rows_frac=float((deltas[k] == 0).all(axis=1).mean())))
        print(json.dumps(out[-1]))
    print("total static Toffoli per pass:", tot_tof)


if __name__ == "__main__":
    main()
