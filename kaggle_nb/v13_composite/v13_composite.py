"""v13: invsqrt d0a9a694 (maoxin123 "QCC 28.24", on ozertuu acc7b9c8) + DQI db0a845f (goodpjw2008 swap-collapse + leaf-port; fallbacks edcf0e62 cosmorin PRELUDE3, f6b555a6
ozertuu) + chemistry 6ef06081 (Manvender Singh; Leo Tang / maoxin123 loaders / Lera table). All on Syed Asad Ali's
Apache-2.0 generators. Official verifiers: invsqrt at 3 salts, chemistry at 1, each DQI candidate at 5 salts.
Writes submission.zip only if invsqrt and chemistry pass and the chosen DQI passes at the default salt."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/v13")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
WANT = {"invsqrt.kmb": ["d0a9a694"], "dqi.kmb": ["db0a845f"], "dfthcsos/submission.npz": ["6ef06081"]}
def h(b): return hashlib.sha256(b).hexdigest()[:8]
got, angles = {}, {}
for z in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True):
    try:
        f = zipfile.ZipFile(z)
    except zipfile.BadZipFile:
        continue
    names = f.namelist()
    for n, ws in WANT.items():
        if n in names:
            b = f.read(n); k = h(b)
            if k in ws and (n, k) not in got:
                got[(n, k)] = b
                if n.startswith("dfthcsos/") and not angles:
                    angles = {m: f.read(m) for m in names if m.startswith("dfthcsos/angle_k_")}
for p in list(INPUT.rglob("*.kmb")) + list(INPUT.rglob("*.npz")):
    n = "dqi.kmb" if p.name.startswith("dqi") else "invsqrt.kmb" if p.name.startswith("invsqrt") else \
        "dfthcsos/submission.npz" if p.name == "submission.npz" else None
    if n is None:
        continue
    b = p.read_bytes(); k = h(b)
    if k in WANT[n] and (n, k) not in got:
        got[(n, k)] = b
        if n.startswith("dfthcsos/") and not angles:
            angles = {"dfthcsos/" + q.name: q.read_bytes() for q in p.parent.glob("angle_k_*.kmb")}
print("found", sorted(got), "angles", len(angles), flush=True)
assert ("invsqrt.kmb", "d0a9a694") in got and ("dfthcsos/submission.npz", "6ef06081") in got and len(angles) == 15
dqis = [k for k in WANT["dqi.kmb"] if ("dqi.kmb", k) in got]
assert dqis, "no DQI candidate found"
(TMP / "dfthcsos").mkdir(parents=True, exist_ok=True)
(TMP / "invsqrt.kmb").write_bytes(got[("invsqrt.kmb", "d0a9a694")])
(TMP / "dfthcsos" / "submission.npz").write_bytes(got[("dfthcsos/submission.npz", "6ef06081")])
for m, b in angles.items():
    (TMP / m).write_bytes(b)
for k in dqis:
    (TMP / f"dqi_{k}.kmb").write_bytes(got[("dqi.kmb", k)])
def run(args):
    v = subprocess.run([sys.executable] + args, cwd=comp / "verifiers", env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")),
                       capture_output=True, text=True)
    try:
        r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
    except ValueError:
        r = {"status": "UNPARSED", "tail": v.stdout[-800:], "err": v.stderr[-800:]}
    return r
res = {"chem": [], "inv": [], "dqi": {k: [] for k in dqis}}
def save(): (OUT / "v13_results.json").write_text(json.dumps(res, indent=1, default=str))
r = run(["dfthcsos/verify.py", str(TMP / "dfthcsos")])
res["chem"].append({"salt": "", "status": r.get("status"), "score": r.get("score"), "err": r.get("error")}); print("chem", res["chem"][-1], flush=True); save()
for k in dqis:
    for s in ("",):
        r = run(["dqi/verify_submission_dqi.py", str(TMP / f"dqi_{k}.kmb"), "--max-workers", "4"] + (["--salt", s] if s else []))
        res["dqi"][k].append({"salt": s, "status": r.get("status"), "score": r.get("score"), "value_errors": r.get("value_errors"),
                              "phase_errors": r.get("phase_errors"), "err": r.get("error")}); print("dqi", k, res["dqi"][k][-1], flush=True); save()
for s in ("", "private-a", "q7"):
    r = run(["invsqrt/verify_submission_invsqrt.py", str(TMP / "invsqrt.kmb")] + (["--salt", s] if s else []))
    res["inv"].append({"salt": s, "status": r.get("status"), "score": r.get("score"), "worst": r.get("worst_error_over_tolerance"),
                       "err": r.get("error")}); print("inv", res["inv"][-1], flush=True); save()
ref = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155375}
ok = all(r["status"] == "PASS" for r in res["chem"] + res["inv"])
pick = next((k for k in dqis if res["dqi"][k][0]["status"] == "PASS"), None)
if ok and pick:
    for k in dqis:
        d = res["dqi"][k]
        if d[0]["status"] == "PASS":
            res[f"gm_{k}"] = (ref["invsqrt"] / res["inv"][0]["score"] * ref["dqi"] / d[0]["score"] * ref["dfthcsos"] / res["chem"][0]["score"]) ** (1 / 3)
    res["pick"] = pick
    print("pick", pick, {k: v for k, v in res.items() if k.startswith("gm_")}, flush=True)
    files = {"invsqrt.kmb": got[("invsqrt.kmb", "d0a9a694")], "dqi.kmb": got[("dqi.kmb", pick)],
             "dfthcsos/submission.npz": got[("dfthcsos/submission.npz", "6ef06081")], **angles}
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(files):
            z.writestr(n, files[n])
    (OUT / "NOTICE_SYED_ASAD_ALI.txt").write_text(
        "Lineage: invsqrt and DQI derive from generators by Syed Asad Ali (Apache-2.0). invsqrt d0a9a694: maoxin123 "
        "(on ozertuu acc7b9c8 / boristown ecf908fe / maoxin123 / Ali Haydar Ozdag line). DQI db0a845f: goodpjw2008 (swap collapse + leaf port, on ozertuu f6b555a6 / "
        "koushikrudra / manwithacat line). Chemistry 6ef06081: Manvender Singh's refit of Leo Tang's weights on maoxin123's "
        "4,022-Toffoli loaders and Lera's coded-word table.\n")
save()
