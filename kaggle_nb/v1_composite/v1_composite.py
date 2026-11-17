"""v1 composite: today's best public file per task, pinned by SHA-256 prefix, fully re-verified.

  invsqrt   0af21cb8  (maoxin123, 44 qubits; Syed Asad Ali generator line)        <- maoxin123 28.14 output
  chemistry npz a73b25d2 (labelled 56ef2562 by its author) + its 15 loaders (maoxin123 4,022-Toffoli loaders)           <- maoxin123 28.14 output
  dqi       61c8348d  (grantprater merge of boristown + Balram Srinivasan; Syed Asad Ali line) <- grantprater output

Runs the three unmodified official verifiers (DQI at the default salt and two extra salts, so we see
its failure behaviour) and writes /kaggle/working/submission.zip only if all pass.
Lineage NOTICE (Syed Asad Ali, Apache-2.0 4(d)) is written next to the zip.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
STAGE = Path("/tmp/v1")
WANT = {"invsqrt.kmb": "0af21cb8", "dqi.kmb": "61c8348d", "dfthcsos/submission.npz": "a73b25d2"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def comp_dir() -> Path:
    return next(p.parent for p in INPUT.rglob("verifiers") if (p / "dqi").is_dir())


def gather() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "dfthcsos").mkdir(parents=True)
    # Unpack every attached submission zip into its own folder, then pick files by hash.
    pool = Path("/tmp/pool")
    pool.mkdir(exist_ok=True)
    for i, z in enumerate(INPUT.rglob("*.zip")):
        if "google-verifiable" in str(z):
            continue
        with zipfile.ZipFile(z) as f:
            f.extractall(pool / f"z{i}")
    cands = [p for p in list(pool.rglob("*")) + list(INPUT.rglob("*")) if p.is_file() and p.suffix in (".kmb", ".npz")]
    found = {}
    for p in cands:
        if p.stat().st_size > 2_000_000_000:
            continue
        d = sha(p)[:8]
        for name, pref in WANT.items():
            if d == pref and name not in found:
                found[name] = p
    print("found:", {k: str(v) for k, v in found.items()}, flush=True)
    missing = set(WANT) - set(found)
    assert not missing, f"missing pinned files: {missing}"
    shutil.copy(found["invsqrt.kmb"], STAGE / "invsqrt.kmb")
    shutil.copy(found["dqi.kmb"], STAGE / "dqi.kmb")
    npz = found["dfthcsos/submission.npz"]
    shutil.copy(npz, STAGE / "dfthcsos" / "submission.npz")
    for p in sorted(npz.parent.glob("angle_k_*.kmb")):  # loaders shipped next to that npz
        shutil.copy(p, STAGE / "dfthcsos" / p.name)
    k = sorted(int(p.stem.split("_")[-1]) for p in (STAGE / "dfthcsos").glob("angle_k_*.kmb"))
    assert k == list(range(len(k))) and k, f"angle_k files missing or gapped: {k}"


def verify(name: str, args: list[str]) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(comp_dir() / "kickmix")
    t = time.time()
    p = subprocess.run([sys.executable, *args], cwd=comp_dir() / "verifiers", env=env, capture_output=True, text=True)
    out = p.stdout
    try:
        r = json.loads(out[out.index("{"):])
    except ValueError:
        r = {"status": "UNPARSED", "tail": out[-2000:], "err": p.stderr[-2000:]}
    r.pop("shot_operation_counts", None)
    r["_s"] = round(time.time() - t)
    print(name, r.get("status"), r.get("score"), r.get("value_errors"), r.get("phase_errors"), f"{r['_s']}s", flush=True)
    return r


def main() -> None:
    gather()
    res = {
        "dfthcsos": verify("chem", ["dfthcsos/verify.py", str(STAGE / "dfthcsos")]),
        "dqi": verify("dqi", ["dqi/verify_submission_dqi.py", str(STAGE / "dqi.kmb"), "--max-workers", "4"]),
        "dqi_s1": verify("dqi_s1", ["dqi/verify_submission_dqi.py", str(STAGE / "dqi.kmb"), "--max-workers", "4", "--salt", "s1"]),
        "dqi_s2": verify("dqi_s2", ["dqi/verify_submission_dqi.py", str(STAGE / "dqi.kmb"), "--max-workers", "4", "--salt", "s2"]),
        "invsqrt": verify("invsqrt", ["invsqrt/verify_submission_invsqrt.py", str(STAGE / "invsqrt.kmb")]),
    }
    REF = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155375}
    ok = all(res[k].get("status") == "PASS" for k in ("invsqrt", "dqi", "dfthcsos"))
    if ok:
        ratios = {k: REF[k] / res[k]["score"] for k in REF}
        res["ratios"] = ratios
        res["geomean_default_salt"] = (ratios["invsqrt"] * ratios["dqi"] * ratios["dfthcsos"]) ** (1 / 3)
        with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in sorted(STAGE.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(STAGE).as_posix())
        (OUT / "NOTICE_SYED_ASAD_ALI.txt").write_text(
            "Lineage: the invsqrt and DQI circuits derive from generators originally built by Syed Asad Ali "
            "(syed3000/vqcc-own-circuit-portfolio, Apache-2.0). Later contributors: maoxin123, grantprater, "
            "boristown, Balram Srinivasan, Hayato Hikawa and others credited in their notebooks.\n")
    (OUT / "v1_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res.get(k) for k in ("ratios", "geomean_default_salt")}, default=str))


if __name__ == "__main__":
    main()
