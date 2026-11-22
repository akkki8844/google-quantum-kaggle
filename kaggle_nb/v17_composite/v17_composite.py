"""v17: invsqrt bb676b95 (leotang09 3-T phase fold of d2148a7c; verified here at 3 salts), DQI 4b572141 (verified at 5 salts in v14;
default salt here) and the cheapest passing chemistry bundle from the 33.144 notebooks (goodpjw2008 3,372 chain + table refit). Original v14 text: invsqrt d2148a7c (37Q; qcc-milestone-set / rithvikredrouthu line), DQI 4b572141
(tonylica 900Q, via quantum-ai-32-85306-ready), chemistry = cheapest passing dfthcsos bundle found in the attached
public outputs (goodpjw2008 dead-word loader chain DP; boristown DAG splicing). Lineage credits: Syed Asad Ali
(Apache-2.0 generators), rithvikredrouthu, tonylica, ozertuu, Almond Labs, goodpjw2008, boristown and the authors
credited in their notices. Official verifiers: every chemistry candidate at the default salt, invsqrt at 3 salts,
DQI at 5 salts. Writes submission.zip only if all chosen files pass at every tested salt (DQI failures only raise R)."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/v17")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
WANT_INV, WANT_DQI = "bb676b95", "4b572141"
def h(b): return hashlib.sha256(b).hexdigest()[:8]
inv = dqi = None
chem = {}   # npz hash -> {member: bytes}
notices = {}
for z in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True):
    try:
        f = zipfile.ZipFile(z)
    except zipfile.BadZipFile:
        continue
    names = f.namelist()
    if inv is None and "invsqrt.kmb" in names:
        b = f.read("invsqrt.kmb")
        if h(b) == WANT_INV: inv = b
    if dqi is None and "dqi.kmb" in names:
        b = f.read("dqi.kmb")
        if h(b) == WANT_DQI: dqi = b
    if "dfthcsos/submission.npz" in names:
        k = h(f.read("dfthcsos/submission.npz"))
        if k not in chem:
            chem[k] = {m: f.read(m) for m in names if m.startswith("dfthcsos/") and not m.endswith("/")}
    for m in names:
        if m.upper().startswith("NOTICE") and m not in notices:
            notices[m] = f.read(m)
print("inv", inv is not None, "dqi", dqi is not None, "chem candidates", {k: len(v) for k, v in chem.items()}, flush=True)
assert inv is not None and dqi is not None and chem
TMP.mkdir(parents=True, exist_ok=True)
(TMP / "invsqrt.kmb").write_bytes(inv); (TMP / "dqi.kmb").write_bytes(dqi)
def run(args):
    v = subprocess.run([sys.executable] + args, cwd=comp / "verifiers", env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")),
                       capture_output=True, text=True)
    try:
        r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
    except ValueError:
        r = {"status": "UNPARSED", "tail": v.stdout[-800:], "err": v.stderr[-800:]}
    return r
res = {"chem": {}, "inv": [], "dqi": []}
def save(): (OUT / "v17_results.json").write_text(json.dumps(res, indent=1, default=str))
for k, files in chem.items():
    d = TMP / f"chem_{k}"
    for m, b in files.items():
        p = d / m[len("dfthcsos/"):]; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
    r = run(["dfthcsos/verify.py", str(d)])
    res["chem"][k] = {"status": r.get("status"), "score": r.get("score"), "err": r.get("error"),
                      "Q": r.get("num_qubits"), "T": r.get("toffoli_count"), "lam": r.get("lambda_eff")}
    print("chem", k, res["chem"][k], flush=True); save()
ok_chem = {k: v for k, v in res["chem"].items() if v["status"] == "PASS"}
assert ok_chem, "no chemistry candidate passed"
best = min(ok_chem, key=lambda k: ok_chem[k]["score"]); res["chem_pick"] = best
for s in ("", "private-a", "q7"):
    r = run(["invsqrt/verify_submission_invsqrt.py", str(TMP / "invsqrt.kmb")] + (["--salt", s] if s else []))
    res["inv"].append({"salt": s, "status": r.get("status"), "score": r.get("score"), "worst": r.get("worst_error_over_tolerance"),
                       "Q": r.get("num_qubits"), "err": r.get("error")}); print("inv", res["inv"][-1], flush=True); save()
for s in ("",):
    r = run(["dqi/verify_submission_dqi.py", str(TMP / "dqi.kmb"), "--max-workers", "4"] + (["--salt", s] if s else []))
    res["dqi"].append({"salt": s, "status": r.get("status"), "score": r.get("score"), "value_errors": r.get("value_errors"),
                       "phase_errors": r.get("phase_errors"), "err": r.get("error")}); print("dqi", res["dqi"][-1], flush=True); save()
ref = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155375}
ok = all(r["status"] == "PASS" for r in res["inv"] + res["dqi"])
if ok:
    gm = (ref["invsqrt"] / res["inv"][0]["score"] * ref["dqi"] / res["dqi"][0]["score"] * ref["dfthcsos"] / ok_chem[best]["score"]) ** (1 / 3)
    res["gm_default"] = gm; print("GM default", gm, "chem pick", best, flush=True)
    files = {"invsqrt.kmb": inv, "dqi.kmb": dqi, **chem[best]}
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(files):
            z.writestr(n, files[n])
    lines = ["Lineage and credits: invsqrt bb676b95 (leotang09 3-T phase fold of rithvikredrouthu's 37Q d2148a7c); DQI 4b572141 (tonylica 900Q); "
             f"chemistry {best} (from the attached public notebooks: goodpjw2008 3,372 chain + table refit / maoxin123 loaders / Leo Tang table / ozertuu table / Almond Labs / rithvikredrouthu lines); "
             "all on Syed Asad Ali's Apache-2.0 generators. Upstream notices follow.\n"]
    for m, b in sorted(notices.items()):
        lines.append(f"\n=== {m} ===\n" + b.decode("utf-8", "replace"))
    (OUT / "NOTICE.txt").write_text("".join(lines))
save()
