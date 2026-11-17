"""B4 (Kaggle CPU): run Syed Asad Ali's generic invsqrt finishing passes (v12 source, Apache-2.0) on our 44-qubit
frontier circuit (maoxin123's 0af21cb8, from our own v2 package), then judge the result with the official verifier.

Passes (tools/pipeline.sh minus its model-specific prover):
  audit_affine -> audit_pairs -> hold_pass -> (audit_affine -> postpass) x<=4 -> merge_zpow
Each audit is a full-domain scan of all 3,133,452 inputs. The official verifier is the acceptance test.
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
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "invsqrt").is_dir())
sys.path.insert(0, str(comp / "kickmix"))
import kickmix as km  # noqa: E402


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def run(cmd, out=None, **kw):
    log("$", " ".join(map(str, cmd)))
    p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, **kw)
    txt = p.stdout + "\n" + p.stderr
    if out:
        Path(out).write_text(txt)
    print(txt[-2500:], flush=True)
    return p


# --- set up Syed's tree exactly as build.sh does ---
builds = [Path(p) for p in glob.glob(str(INPUT / "**" / "build.sh"), recursive=True)
          if "invsqrt.kmb" in (Path(p).parent / "README.md").read_text(errors="ignore")[:600]]
zips = glob.glob(str(INPUT / "**" / "invsqrt_source.zip"), recursive=True)
stage = Path("/tmp/b4src")
if stage.exists():
    shutil.rmtree(stage)
if zips:
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(stage)
    srcdir = next(stage.rglob("build.sh")).parent
else:
    srcdir = builds[0].parent
X = Path("/tmp/b4tree")
if X.exists():
    shutil.rmtree(X)
shutil.copytree(srcdir / "src", X)
off = X / "work/c10/narrow/base/official"
(off / "kickmix").mkdir(parents=True, exist_ok=True)
(off / "verifier_invsqrt").mkdir(parents=True, exist_ok=True)
shutil.copy(comp / "kickmix" / "kickmix.so", off / "kickmix" / "kickmix.so")
for f in ("__init__.py", "nr_reference_circuit.py", "verify_submission_invsqrt.py"):
    if (comp / "verifiers/invsqrt" / f).exists():
        shutil.copy(comp / "verifiers/invsqrt" / f, off / "verifier_invsqrt" / f)
base = X / "work" / "c10"
for d, subdirs, files in os.walk(X):
    if Path(d).resolve() == base.resolve():
        subdirs[:] = []
        continue
    for f in files:
        p = Path(d) / f
        b = p.read_bytes()
        if b"@FG_ROOT@" in b and f.endswith((".py", ".sh")):
            p.write_bytes(b.replace(b"@FG_ROOT@", str(X).encode()))
T = X / "tools"
env = dict(os.environ, FG=str(X), PY=sys.executable, MG="", NARROW=str(X / "work/c10/narrow"),
           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def official(kmb: Path) -> dict:
    v = subprocess.run([sys.executable, "invsqrt/verify_submission_invsqrt.py", str(kmb)], cwd=comp / "verifiers",
                       env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")), capture_output=True, text=True)
    try:
        r = json.loads(v.stdout[v.stdout.index("{"):])
        r.pop("shot_operation_counts", None)
    except ValueError:
        r = {"status": "UNPARSED", "tail": v.stdout[-1500:], "err": v.stderr[-1500:]}
    log("official", kmb.name, r.get("status"), r.get("score"), r.get("toffoli_count"), r.get("num_qubits"))
    return r


def frontier_kmb() -> Path:
    for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True):
        with zipfile.ZipFile(z) as f:
            if "invsqrt.kmb" in f.namelist():
                f.extract("invsqrt.kmb", "/tmp/b4in")
                return Path("/tmp/b4in/invsqrt.kmb")
    raise SystemExit("no invsqrt.kmb")


def pipeline(name: str, kmb: Path, Q: int) -> dict:
    o = OUT / name
    o.mkdir(exist_ok=True)
    c = km.Circuit.from_file(str(kmb))
    inp = o / "in.kmx"
    c.to_file(str(inp), format="kmx")
    res = {"Q": c.num_qubits}
    run([sys.executable, T / "audit_affine.py", inp, "--out", o / "s1.audit.json"], o / "s1.audit.log", env=env, cwd=X)
    run([sys.executable, T / "audit_pairs.py", inp, "--live", o / "s1.audit.live.npy", "--out", o / "s1.pairs.json"],
        o / "s1.pairs.log", env=env, cwd=X)
    cur = inp
    if (o / "s1.pairs.json").exists():
        p = run([sys.executable, T / "hold_pass.py", inp, o / "s1.pairs.json", "--out", o / "s2.kmx", "--Q", Q],
                o / "s2.hold.log", env=env, cwd=X)
        if p.returncode == 0 and (o / "s2.kmx").exists():
            cur = o / "s2.kmx"
    for rnd in range(1, 5):
        aj = o / f"r{rnd}.audit.json"
        run([sys.executable, T / "audit_affine.py", cur, "--out", aj], o / f"r{rnd}.audit.log", env=env, cwd=X)
        try:
            n = len(json.loads(aj.read_text())["findings"])
        except Exception as e:
            log("audit read failed", e)
            break
        res[f"affine_round{rnd}"] = n
        if n == 0:
            break
        nxt = o / f"r{rnd}.kmx"
        p = run([sys.executable, T / "postpass.py", cur, aj, "--out", nxt], o / f"r{rnd}.post.log", env=env, cwd=X)
        if p.returncode != 0:
            break
        cur = nxt
    p = run([sys.executable, T / "merge_zpow.py", cur, "--out", o / "zpow.kmx"], o / "zpow.log", env=env, cwd=X)
    if p.returncode == 0 and (o / "zpow.kmx").exists():
        cur = o / "zpow.kmx"
    final = o / f"{name}_pipe.kmb"
    km.Circuit.from_file(str(cur), format="kmx").to_file(str(final), format="kmb")
    res["orig"] = official(kmb)
    res["new"] = official(final)
    return res


def main():
    results = {"frontier_0af21cb8": pipeline("frontier", frontier_kmb(), 44)}
    (OUT / "b4_result.json").write_text(json.dumps(results, indent=1, default=str))
    log("done")


if __name__ == "__main__":
    main()
