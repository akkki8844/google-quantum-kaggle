"""P1 on Kaggle: score the public frontier files with the official verifiers and profile them.

Inputs (attached by tools/kpush.py, read-only on Kaggle):
  - the competition data (verifiers, competition kickmix build)
  - maoxin123/qcc-frontier-dqi-chem-mirror   (DQI ccc2ab84 as .kmb.xz, chemistry f6e12f02 family)
  - maoxin123/qcc-v27-... notebook output      (44-qubit invsqrt, if present in its output)
Writes /kaggle/working/p1_results.json and per-task profiles. No network access.
"""
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
WORK = Path("/tmp/frontier")  # scratch, not saved as kernel output (the DQI circuit is ~0.9 GB)


def find_competition_dir() -> Path:
    for p in sorted(INPUT.rglob("verifiers")):
        if (p / "dqi").is_dir():
            return p.parent
    raise FileNotFoundError("verifiers/ not found")


COMP = find_competition_dir()
sys.path.insert(0, str(COMP / "kickmix"))
import kickmix as km  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listing() -> list[str]:
    return sorted(str(p.relative_to(INPUT)) for p in INPUT.rglob("*") if p.is_file() and "verifiers" not in p.parts
                  and "FeMoCo76o113e.fcidump" not in p.name)


def collect() -> dict[str, Path]:
    """Find the frontier files among the attached inputs; unpack archives into WORK."""
    WORK.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    # Unpack any submission zips from attached notebook outputs.
    for z in INPUT.rglob("*.zip"):
        if "google-verifiable" in str(z):
            continue
        dest = WORK / ("unzipped_" + z.stem)
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
    for xz in INPUT.rglob("dqi.kmb.xz"):
        dst = WORK / "dqi.kmb"
        with lzma.open(xz) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
        found["dqi"] = dst
    roots = [p for p in INPUT.rglob("*") if p.is_dir() and "competitions" not in str(p)] + [WORK]
    for root in roots:
        for name in ("invsqrt.kmb", "invsqrt.kmx"):
            if (root / name).is_file() and "invsqrt" not in found:
                found["invsqrt"] = root / name
        if (root / "angle_k_0.kmb").is_file() and "dfthcsos" not in found:
            found["dfthcsos"] = root
    # The chemistry mirror has no submission.npz next to its circuits in some versions; look for one.
    if "dfthcsos" in found and not (found["dfthcsos"] / "submission.npz").is_file():
        npz = next((p for p in INPUT.rglob("submission.npz") if "sample_submission" not in str(p)), None)
        if npz:
            dst = WORK / "dfthcsos"
            shutil.copytree(found["dfthcsos"], dst, dirs_exist_ok=True)
            shutil.copy(npz, dst / "submission.npz")
            found["dfthcsos"] = dst
    return found


def run(name: str, args: list[str]) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(COMP / "kickmix") + os.pathsep + env.get("PYTHONPATH", "")
    t = time.time()
    p = subprocess.run([sys.executable, *args], cwd=COMP / "verifiers", env=env, capture_output=True, text=True)
    out = p.stdout.strip()
    try:
        summary = json.loads(out[out.index("{"):])
    except ValueError:
        summary = {"status": "UNPARSED", "stdout_tail": out[-3000:], "stderr_tail": p.stderr[-3000:]}
    summary["_seconds"] = round(time.time() - t, 1)
    summary.pop("shot_operation_counts", None)
    print(f"[{name}] {summary.get('status')} score={summary.get('score')} in {summary['_seconds']}s", flush=True)
    return summary


def profile(circuit_path: Path) -> dict:
    c = km.Circuit.from_file(str(circuit_path))
    st = c.operation_statistics()
    return {
        "num_qubits": c.num_qubits, "num_ops": len(c),
        "toffoli_like_static": st.num_toffoli_like, "t_like": st.num_t_like,
        "z_pow_level4plus": st.num_z_pow(4), "ops": {k: int(v) for k, v in st.items()},
        "registers": {str(k): len(v) for k, v in list(c.register_data.items())[:8]},
    }


def main() -> None:
    print("inputs:", json.dumps(listing()[:200], indent=0), flush=True)
    found = collect()
    print("found:", {k: str(v) for k, v in found.items()}, flush=True)
    res: dict = {"found": {k: str(v) for k, v in found.items()}}
    if "invsqrt" in found:
        res["invsqrt_profile"] = profile(found["invsqrt"])
        res["invsqrt_sha256"] = sha(found["invsqrt"])
        res["invsqrt"] = run("invsqrt", ["invsqrt/verify_submission_invsqrt.py", str(found["invsqrt"])])
        # Keep the circuit text for offline study (small).
        km.Circuit.from_file(str(found["invsqrt"])).to_file(str(OUT / "frontier_invsqrt.kmx"), format="kmx")
    if "dfthcsos" in found:
        res["dfthcsos"] = run("dfthcsos", ["dfthcsos/verify.py", str(found["dfthcsos"])])
    if "dqi" in found:
        res["dqi_profile"] = profile(found["dqi"])
        res["dqi_sha256"] = sha(found["dqi"])
        res["dqi"] = run("dqi", ["dqi/verify_submission_dqi.py", str(found["dqi"]), "--max-workers", "4"])
    (OUT / "p1_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: (v.get("score") if isinstance(v, dict) else v) for k, v in res.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
