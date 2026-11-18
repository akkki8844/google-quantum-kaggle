"""R2: private-salt robustness of our submitted files (invsqrt 864f81cd, DQI 5f901c55) with the official verifiers.
A failure on the private salt zeroes the whole submission, so check several salts before choosing finals."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/r2")
TMP.mkdir(exist_ok=True)
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "invsqrt").is_dir())
WANT = {"invsqrt.kmb": "864f81cd", "dqi.kmb": "5f901c55"}
for p in list(INPUT.rglob("submission.zip")):
    with zipfile.ZipFile(p) as f:
        for n in WANT:
            if n in f.namelist() and not (TMP / n).exists() and hashlib.sha256(f.read(n)).hexdigest().startswith(WANT[n]):
                (TMP / n).write_bytes(f.read(n))
for p in INPUT.rglob("*.kmb"):
    if p.name in WANT and not (TMP / p.name).exists() and hashlib.sha256(p.read_bytes()).hexdigest().startswith(WANT[p.name]):
        (TMP / p.name).write_bytes(p.read_bytes())
assert all((TMP / n).exists() for n in WANT)
res = []
def run(script, path, salt):
    args = [sys.executable, script, str(path)] + (["--max-workers", "4"] if "dqi" in script else []) + ["--salt", salt]
    v = subprocess.run(args, cwd=comp / "verifiers", env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")), capture_output=True, text=True)
    try:
        r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
    except ValueError:
        r = {"status": "UNPARSED", "tail": v.stdout[-800:], "err": v.stderr[-800:]}
    row = {"task": script.split("/")[0], "salt": salt, "status": r.get("status"), "score": r.get("score"),
           "value_errors": r.get("value_errors"), "phase_errors": r.get("phase_errors"), "max_err_ratio": r.get("max_error_ratio", r.get("worst_error_ratio"))}
    print(row, flush=True); res.append(row)
    (OUT / "r2_salts.json").write_text(json.dumps(res, indent=1, default=str))
for s in ("private-a", "private-b", "private-c", "q7", "zz-final"):
    run("invsqrt/verify_submission_invsqrt.py", TMP / "invsqrt.kmb", s)
for s in ("private-a", "private-b", "private-c", "q7"):
    run("dqi/verify_submission_dqi.py", TMP / "dqi.kmb", s)
