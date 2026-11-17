"""Workstream B2 (long runs; B1 header kept below)

Workstream B1 (Kaggle GPU): can cheaper factorization shapes reach the 20 mHa gate?

For each candidate shape (R, B, C), train the official starter objective (JAX + optax,
symmetry shift frozen to the baseline's, verifier floors in the objective), seeded from the
public frontier table (8, 143, 44) by truncation/padding (the starter's load_seed_params).
Records the continuous two-body error (epsilon_fro), Lambda and lambda_eff over time.

This is a feasibility screen only: no quantization, no circuits. A shape is worth the full
pipeline if its continuous error falls well below 20 mHa (target <= ~15 mHa, since 8-bit
words cost several mHa even with the frontier's co-fitted rounding).

Inputs: competition data, sohier/verifiable-circuit-compilation-starter-kit,
maoxin123/qcc-frontier-dqi-chem-mirror (for a seed table; falls back to the kit baseline).
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
ROOT = Path("/tmp/skroot")  # writable copy: <ROOT>/starter_kit + fcidump + fixed shifts

SHAPES = [(7, 143, 50), (6, 129, 50)]
EPOCHS = {}
DEFAULT_EPOCHS = 300000
TIME_BUDGET_S = 10.5 * 3600


def setup() -> Path:
    kit = next(p for p in INPUT.rglob("starter_kit") if (p / "dfthc").is_dir())
    pkg = kit.parent
    if ROOT.exists():
        shutil.rmtree(ROOT)
    shutil.copytree(kit, ROOT / "starter_kit", ignore=shutil.ignore_patterns("*.fcidump"))
    # FeMoCo76o113e reference integrals and fixed shifts: kit root, else competition verifiers.
    for name in ("FeMoCo76o113e.fcidump", "FeMoCo76o113e.fixed_shifts.npz"):
        src = next((p for p in [pkg / name] if p.is_file()), None) or next(INPUT.rglob(name))
        os.symlink(src, ROOT / name)
    (ROOT / "starter_kit" / "data" / "integrals").mkdir(parents=True, exist_ok=True)
    return ROOT / "starter_kit"


def seed_table() -> Path:
    for p in INPUT.rglob("submission.npz"):
        if "sample_submission" in str(p):
            continue
        import numpy as np
        with np.load(p, allow_pickle=False) as z:
            if z["U"].shape == (8, 143, 76):
                return p
    return next((ROOT / "starter_kit" / "baselines").glob("FeMoCo76o113e_*.npz"))


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, s):
        self.f.write(s)
        self.f.flush()
        sys.__stdout__.write(s)

    def flush(self):
        self.f.flush()


def main() -> None:
    kit = setup()
    sys.path.insert(0, str(kit))
    os.chdir(kit)
    import jax
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    from starter.task1_factorization import train_df_thc_qaggle as T

    seed = seed_table()
    print("seed table:", seed, flush=True)
    (OUT / "runs").mkdir(exist_ok=True)
    summary = []
    start = time.time()
    for shape in SHAPES:
        if time.time() - start > TIME_BUDGET_S:
            print("time budget reached", flush=True)
            break
        R, B, C = shape
        own = next(iter(INPUT.rglob(f'shape_{R}_{B}_{C}_ep30000.npz')), None)
        shape_seed = own if own is not None else seed
        epochs = EPOCHS.get(shape, DEFAULT_EPOCHS)
        log = OUT / "runs" / f"shape_{R}_{B}_{C}.log"
        out = OUT / "runs" / f"shape_{R}_{B}_{C}_ep{epochs}.npz"
        t0 = time.time()
        old = sys.stdout
        sys.stdout = Tee(log)
        try:
            params, values = T.train_system(
                "FeMoCo76o113e", epochs=epochs, lr=1e-3, print_interval=10000,
                rank=R, bases=B, copies=C, seed_solution_file=str(shape_seed),
                rho=1e-4, epsilon_dfthc=0.012, output_file=str(out))
            row = {"shape": shape, "epochs": epochs, "seconds": round(time.time() - t0),
                   **{k: float(v) for k, v in values.items() if hasattr(v, "__float__") and getattr(v, "ndim", 0) == 0}}
        except Exception as ex:  # keep going with the next shape
            row = {"shape": shape, "error": repr(ex)[:500]}
        finally:
            sys.stdout = old
        summary.append(row)
        print(json.dumps(row)[:600], flush=True)
        (OUT / "b1_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
