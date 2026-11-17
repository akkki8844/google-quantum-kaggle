"""B3 (Kaggle CPU): rebuild Syed Asad Ali's 46-qubit invsqrt generator (v12 source, Apache-2.0) from source.

Goal: a working, exactly-proved generator on Kaggle so we can make structural changes and re-prove them.
Expected output SHA-256: 321aff83636214196daa9cbc9dcc9c5e12a155bcff8d9989e98b8cee40935177 (Q46, 1511.57 Toffolis).
The build wants CPython 3.13 + numpy; if the image's Python is another version we fetch 3.13 with uv (internet on).
Writes /kaggle/working/invsqrt_syed46.kmb, the build logs, and the official verifier result for the rebuilt file.
"""
import glob
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
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def run(cmd, **kw):
    log("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    print(p.stdout[-4000:], p.stderr[-4000:], flush=True)
    return p


comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "invsqrt").is_dir())
zips = glob.glob(str(INPUT / "**" / "invsqrt_source.zip"), recursive=True)
builds = [Path(p) for p in glob.glob(str(INPUT / "**" / "build.sh"), recursive=True)
          if "invsqrt.kmb" in (Path(p).parent / "README.md").read_text(errors="ignore")[:600]]
log("python", sys.version.split()[0], "competition", comp, "zips", zips, "extracted builds", builds)

py = sys.executable
if sys.version_info[:2] != (3, 13):
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run(["uv", "python", "install", "3.13"])
    run(["uv", "venv", "--python", "3.13", "/tmp/v313"])
    run(["uv", "pip", "install", "--python", "/tmp/v313/bin/python", "numpy"])
    py = "/tmp/v313/bin/python"
run([py, "-c", "import sys,numpy;print(sys.version, numpy.__version__)"])

work = Path("/tmp/b3")
if work.exists():
    shutil.rmtree(work)
work.mkdir()
if zips:
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(work)
else:
    shutil.copytree(builds[0].parent, work / "invsqrt_source")
srcdir = next(work.rglob("build.sh")).parent
outdir = Path("/tmp/b3_out")
outdir.mkdir(exist_ok=True)
env = dict(os.environ, PY=py, WORK="/tmp/b3_work")
p = run(["bash", str(srcdir / "build.sh"), str(comp / "kickmix"), str(comp / "verifiers"), str(outdir)], env=env)
res = {"build_rc": p.returncode, "build_tail": p.stdout[-3000:], "build_err": p.stderr[-3000:]}
for lg in glob.glob("/tmp/b3_work/invsqrt_build.*/*.log"):
    shutil.copy(lg, OUT / ("b3_" + Path(lg).name))
kmb = outdir / "invsqrt.kmb"
if kmb.exists():
    shutil.copy(kmb, OUT / "invsqrt_syed46.kmb")
    venv = dict(os.environ, PYTHONPATH=str(comp / "kickmix"))
    v = subprocess.run([sys.executable, "invsqrt/verify_submission_invsqrt.py", str(kmb)], cwd=comp / "verifiers",
                       env=venv, capture_output=True, text=True)
    try:
        r = json.loads(v.stdout[v.stdout.index("{"):])
        r.pop("shot_operation_counts", None)
    except ValueError:
        r = {"tail": v.stdout[-2000:], "err": v.stderr[-2000:]}
    res["official"] = r
    log("official", r.get("status"), r.get("score"))
(OUT / "b3_result.json").write_text(json.dumps(res, indent=1, default=str))
# keep the generated work tree (circuit text, proofs) for analysis
for d in glob.glob("/tmp/b3_work/invsqrt_build.*/tree/final"):
    shutil.make_archive(str(OUT / "b3_final"), "zip", d)
log("done rc", p.returncode)
