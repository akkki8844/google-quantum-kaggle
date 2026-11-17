"""D1 (Kaggle CPU): dead-write scan + exact rewrite of our current DQI, then official verification.

Dead write: an unconditioned `CCX a b t` whose target t is touched by nothing until an unconditioned `HMR t bK`,
and bK is not mentioned in between. Measuring first is the same operation without the Toffoli:
    HMR t bK ; PUSH_CONDITION if bK ; CZ a b ; POP_CONDITION      (and the later HMR is deleted)
since the measured phase (-1)^{bK (v xor ab)} is reproduced exactly (t = v at the early HMR, CZ adds ab).

Input: dqi.kmb inside our own v2 notebook's submission.zip (attached as a kernel source).
Output: d1_report.json (candidates, verifier results for the original and the patched circuit at the default salt
and salts s1/s2) and, if the patched circuit passes with a lower cost, dqi_d1.kmb.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
TMP = Path("/tmp/d1")
TMP.mkdir(exist_ok=True)
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dqi").is_dir())
sys.path.insert(0, str(comp / "kickmix"))
import kickmix as km  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def get_dqi() -> Path:
    zips = [Path(z) for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True)]
    for z in zips:
        with zipfile.ZipFile(z) as f:
            if "dqi.kmb" in f.namelist():
                f.extract("dqi.kmb", TMP)
                log("dqi.kmb from", z)
                return TMP / "dqi.kmb"
    raise SystemExit("no submission.zip with dqi.kmb attached")


def scan(kmx: Path):
    pending = {}        # qubit -> (line index, CCX text) for an unconditioned CCX not yet followed by any touch
    last_bit = {}       # classical bit -> last line index mentioning it
    cands = []
    depth = 0
    n = 0
    with open(kmx) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s or s[0] == "#" or s.startswith(("REGISTER", "APPEND_TO_REGISTER")):
                continue
            n += 1
            tok = s.split()
            op = tok[0]
            qs = [t for t in tok[1:] if t[0] == "q"]
            bs = [t for t in tok[1:] if t[0] == "b"]
            if op == "PUSH_CONDITION":
                depth += 1
            elif op == "POP_CONDITION":
                depth -= 1
            if op == "HMR" and depth == 0 and qs:
                t = qs[0]
                p = pending.get(t)
                if p is not None and bs and last_bit.get(bs[0], -1) < p[0]:
                    cands.append({"ccx_line": p[0], "ccx": p[1], "hmr_line": i, "hmr": s})
            for q in qs:
                pending.pop(q, None)
            if op == "CCX" and depth == 0 and len(qs) == 3:
                pending[qs[2]] = (i, s)
            for b in bs:
                last_bit[b] = i
    return cands, n


def patch(kmx: Path, cands, out: Path):
    rep = {c["ccx_line"]: c for c in cands}
    drop = {c["hmr_line"] for c in cands}
    with open(kmx) as f, open(out, "w") as g:
        for i, line in enumerate(f):
            if i in drop:
                continue
            c = rep.get(i)
            if c is None:
                g.write(line)
                continue
            _, a, b, t = c["ccx"].split()
            bit = c["hmr"].split()[2]
            g.write(f"HMR {t} {bit}\nPUSH_CONDITION if {bit}\nCZ {a} {b}\nPOP_CONDITION\n")


def verify(path: Path, salt=None):
    env = dict(os.environ, PYTHONPATH=str(comp / "kickmix"))
    args = [sys.executable, "dqi/verify_submission_dqi.py", str(path), "--max-workers", "4"]
    if salt:
        args += ["--salt", salt]
    p = subprocess.run(args, cwd=comp / "verifiers", env=env, capture_output=True, text=True)
    out = p.stdout
    try:
        r = json.loads(out[out.index("{"):])
    except ValueError:
        r = {"status": "UNPARSED", "tail": out[-1500:], "err": p.stderr[-1500:]}
    r.pop("shot_operation_counts", None)
    log("verify", path.name, salt, r.get("status"), r.get("score"), r.get("toffoli_count"),
        r.get("value_errors"), r.get("phase_errors"))
    return r


def main():
    src = get_dqi()
    rep = {"input_sha": sha(src)}
    c = km.Circuit.from_file(str(src))
    kmx = TMP / "dqi.kmx"
    c.to_file(str(kmx), format="kmx")
    del c
    log("kmx", kmx.stat().st_size // 10**6, "MB")
    cands, n = scan(kmx)
    rep.update({"instructions": n, "candidates": len(cands), "examples": cands[:40]})
    log("instructions", n, "dead-write candidates", len(cands))
    (OUT / "d1_report.json").write_text(json.dumps(rep, indent=1))
    if not cands:
        return
    pk = TMP / "dqi_d1.kmx"
    patch(kmx, cands, pk)
    pc = km.Circuit.from_file(str(pk), format="kmx")
    pkb = TMP / "dqi_d1.kmb"
    pc.to_file(str(pkb), format="kmb")
    del pc
    rep["patched_sha"] = sha(pkb)
    rep["orig_default"] = verify(src)
    rep["patched_default"] = verify(pkb)
    (OUT / "d1_report.json").write_text(json.dumps(rep, indent=1, default=str))
    ok = rep["patched_default"].get("status") == "PASS" and \
        rep["patched_default"].get("score", 1e99) < rep["orig_default"].get("score", 0)
    if ok:
        for s in ("s1", "s2"):
            rep[f"patched_{s}"] = verify(pkb, s)
        os.replace(pkb, OUT / "dqi_d1.kmb")
    rep["accepted"] = ok
    (OUT / "d1_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log("done", ok)


if __name__ == "__main__":
    main()
