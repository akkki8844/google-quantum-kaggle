"""Workstream C1 (Kaggle CPU): phase profile of the public DQI decoder (ccc2ab84, 890 qubits).

Streams the circuit's .kmx text in chunks (PUSH/POP blocks kept whole), runs each chunk in
kickmix on 256 real verifier shots, and records per chunk:
  - instruction range, static CCX/CCZ/CSWAP count, HMR count, PUSH_CONDITION count
  - live qubits after the chunk (nonzero in any shot) and the max seen inside sub-steps
  - dynamic Toffoli-like count executed in the chunk (simulator counters)
Writes /kaggle/working/c1_profile.json (+ a compact CSV). Used to split T into the forward
extraction, the 4,095-position scan and the reverse pass, and to see where qubits peak.
"""
import csv
import importlib.util
import json
import lzma
import shutil
import sys
import time
from pathlib import Path

import numpy as np

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
TMP = Path("/tmp/c1")
TMP.mkdir(exist_ok=True)

comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dqi").is_dir())
sys.path.insert(0, str(comp / "kickmix"))
import kickmix as km  # noqa: E402

spec = importlib.util.spec_from_file_location("vdqi", comp / "verifiers" / "dqi" / "verify_submission_dqi.py")
vdqi = importlib.util.module_from_spec(spec)
sys.modules["vdqi"] = vdqi
spec.loader.exec_module(vdqi)

CHUNK = 200_000  # instructions per chunk (approx.; PUSH/POP blocks are never split)
BATCH = 256


def load_dqi() -> km.Circuit:
    xz = next(INPUT.rglob("dqi.kmb.xz"))
    dst = TMP / "dqi.kmb"
    if not dst.exists():
        with lzma.open(xz) as s, open(dst, "wb") as d:
            shutil.copyfileobj(s, d)
    return km.Circuit.from_file(str(dst))


def main() -> None:
    t0 = time.time()
    c = load_dqi()
    n = c.num_qubits
    print("loaded", n, "qubits", len(c), "ops", f"{time.time() - t0:.0f}s", flush=True)
    kmx = TMP / "dqi.kmx"
    c.to_file(str(kmx), format="kmx")
    print("kmx written", kmx.stat().st_size // 1_000_000, "MB", flush=True)

    spec_ = vdqi.ProblemSpecDQI.benchmark_instance(num_shots=BATCH)
    tables = vdqi.FieldTables(km.GF2Field(spec_.degree))
    rng = np.random.default_rng(12345)
    syndromes, mbuc, phases = vdqi.sample_shots(spec_, tables, rng, BATCH)
    syn_q, mb_b, _ = vdqi.validate_registers(c, spec_)
    sim = km.Simulator(batch_size=BATCH, debug_print=False, counting_operations=True, seed=99)
    sim.use_same_registers_as(c)
    sim.write_across_shots(km.array(syn_q), vdqi.to_bits(syndromes, spec_.degree))
    sim.write_across_shots(km.array(mb_b), vdqi.to_bits(mbuc, spec_.degree))
    allq = km.array([km.q(i) for i in range(n)])
    weights = np.count_nonzero(syndromes, axis=1)

    rows = []
    depth = 0
    buf: list[str] = []
    start_idx = 0
    idx = 0
    last_tof = 0.0

    def flush(end_idx: int) -> None:
        nonlocal buf, start_idx, last_tof
        if not buf:
            return
        text = "\n".join(buf)
        static = {k: text.count("\n" + k + " ") + (1 if text.startswith(k + " ") else 0)
                  for k in ("CCX", "CCZ", "CSWAP", "HMR", "PUSH_CONDITION")}
        sim.do(km.Circuit(text))
        live = int(sim.read_across_shots(allq).any(axis=1).sum())
        tof = float(sim.operation_statistics().num_toffoli_like) / BATCH
        rows.append({"start": start_idx, "end": end_idx, **static, "live_after": live,
                     "dyn_toffoli": round(tof - last_tof, 2)})
        last_tof = tof
        buf = []
        start_idx = end_idx

    with open(kmx) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(("REGISTER", "APPEND_TO_REGISTER")):
                continue
            buf.append(s)
            idx += 1
            if s.startswith("PUSH_CONDITION"):
                depth += 1
            elif s.startswith("POP_CONDITION"):
                depth -= 1
            if depth == 0 and len(buf) >= CHUNK:
                flush(idx)
                if len(rows) % 25 == 0:
                    print(f"chunk {len(rows)} instr {idx:,} live {rows[-1]['live_after']} "
                          f"dynT {last_tof:,.0f} {time.time() - t0:.0f}s", flush=True)
    flush(idx)

    phase = sim.read_phase_flipped_across_shots()
    restored = (sim.read_across_shots(km.array(syn_q)) == vdqi.to_bits(syndromes, spec_.degree)).all(axis=0)
    res = {"qubits": n, "instructions": idx, "chunks": len(rows), "total_dyn_toffoli_per_shot": last_tof,
           "phase_ok_frac": float((phase == phases).mean()), "restored_frac": float(restored.mean()),
           "mean_error_weight_proxy_nonzero_syndromes": float(weights.mean()),
           "seconds": round(time.time() - t0)}
    print(json.dumps(res), flush=True)
    (OUT / "c1_profile.json").write_text(json.dumps({"summary": res, "rows": rows}))
    with open(OUT / "c1_profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
