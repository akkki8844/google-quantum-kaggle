"""v9 (hedge): DQI e208d350 (manwithacat, Apache-2.0 public circuits) in place of 15a4b740. Base v8: invsqrt ecf908fe (boristown, on maoxin123 / Ali Haydar Ozdag / Syed Asad Ali line) + DQI 15a4b740 (ozertuu) + chemistry 6ef06081 (Manvender Singh;
DQI 15a4b740 by ozertuu; invsqrt lineage maoxin123 d0227f3c; chemistry lineage Leo Tang / maoxin123 loaders / Lera table;
all on Syed Asad Ali's Apache-2.0 generators). Official verifiers: invsqrt and chemistry at several salts (a failure there
would zero a submission), DQI at 5 salts (failed shots only raise R). Writes submission.zip only if invsqrt and chemistry
pass at every salt."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/v9")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
WANT = {"invsqrt.kmb": "ecf908fe", "dqi.kmb": "e208d350", "dfthcsos/submission.npz": "6ef06081"}
got = {}
def h(b): return hashlib.sha256(b).hexdigest()[:8]
for z in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True):
    try:
        f = zipfile.ZipFile(z)
    except zipfile.BadZipFile:
        continue
    names = f.namelist()
    for n, w in WANT.items():
        if n in names and n not in got and h(f.read(n)) == w:
            got[n] = f.read(n)
            if n.startswith("dfthcsos/"):
                for m in names:
                    if m.startswith("dfthcsos/angle_k_"):
                        got[m] = f.read(m)
for p in INPUT.rglob("e208d3507475*.kmb"):
    if "dqi.kmb" not in got:
        b = p.read_bytes()
        if h(b) == WANT["dqi.kmb"]:
            got["dqi.kmb"] = b
print({n: h(b) for n, b in got.items() if "angle" not in n}, sum("angle" in n for n in got), flush=True)
assert all(n in got for n in WANT) and sum("angle" in n for n in got) == 15
TMP.mkdir(parents=True, exist_ok=True)
(TMP / "dfthcsos").mkdir(exist_ok=True)
for n, b in got.items():
    (TMP / n).write_bytes(b)
def run(args):
    v = subprocess.run([sys.executable] + args, cwd=comp / "verifiers", env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")),
                       capture_output=True, text=True)
    try:
        r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
    except ValueError:
        r = {"status": "UNPARSED", "tail": v.stdout[-800:], "err": v.stderr[-800:]}
    return r
res = {"chem": [], "dqi": [], "inv": []}
def save(): (OUT / "v9_results.json").write_text(json.dumps(res, indent=1, default=str))
for s in ("",):
    r = run(["dfthcsos/verify.py", str(TMP / "dfthcsos")] + (["--salt", s] if s else []))
    res["chem"].append({"salt": s, "status": r.get("status"), "score": r.get("score"), "err": r.get("error")}); print("chem", res["chem"][-1], flush=True); save()
for s in ("", "s1", "s2", "private-a", "private-b"):
    r = run(["dqi/verify_submission_dqi.py", str(TMP / "dqi.kmb"), "--max-workers", "4"] + (["--salt", s] if s else []))
    res["dqi"].append({"salt": s, "status": r.get("status"), "score": r.get("score"), "value_errors": r.get("value_errors"),
                       "phase_errors": r.get("phase_errors"), "err": r.get("error")}); print("dqi", res["dqi"][-1], flush=True); save()
for s in ("", "private-a", "q7"):
    r = run(["invsqrt/verify_submission_invsqrt.py", str(TMP / "invsqrt.kmb")] + (["--salt", s] if s else []))
    res["inv"].append({"salt": s, "status": r.get("status"), "score": r.get("score"), "worst": r.get("worst_error_over_tolerance"),
                       "err": r.get("error")}); print("inv", res["inv"][-1], flush=True); save()
ok = all(r["status"] == "PASS" for r in res["chem"] + res["inv"] + res["dqi"])
ref = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155375}
if ok:
    gm = (ref["invsqrt"] / res["inv"][0]["score"] * ref["dqi"] / res["dqi"][0]["score"] * ref["dfthcsos"] / res["chem"][0]["score"]) ** (1 / 3)
    dmean = sum(r["score"] for r in res["dqi"]) / len(res["dqi"])
    gmm = (ref["invsqrt"] / res["inv"][0]["score"] * ref["dqi"] / dmean * ref["dfthcsos"] / res["chem"][0]["score"]) ** (1 / 3)
    res["gm_default"], res["gm_dqi_mean"] = gm, gmm
    print("GM default", gm, "GM with DQI mean over salts", gmm, flush=True)
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(got):
            z.writestr(n, got[n])
    (OUT / "NOTICE_SYED_ASAD_ALI.txt").write_text(
        "Lineage: invsqrt and DQI derive from generators by Syed Asad Ali (Apache-2.0). invsqrt ecf908fe: boristown "
        "(on maoxin123 d0227f3c and Ali Haydar Ozdag 64-row table). DQI e208d350: manwithacat (vqcc-apache-public-circuits; Syed Asad Ali line). Chemistry 6ef06081: "
        "Manvender Singh's refit of Leo Tang's weights on maoxin123's 4,022-Toffoli loaders and Lera's coded-word table.\n")
save()
