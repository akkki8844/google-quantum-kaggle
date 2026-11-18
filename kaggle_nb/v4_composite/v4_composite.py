"""v4: best public file per task (default salt), re-verified with the unchanged official verifiers.
  invsqrt 864f81cd (maoxin123 QCC 28.18 / boristown package; Syed Asad Ali generator lineage)
  dqi     5f901c55 (boristown gap-3 residue repair package; Syed Asad Ali / grantprater / Balram Srinivasan lineage)
  chem    703843d3 (Atakan Aldemir package; maoxin123 4,022 loaders, Lera's table) - byte-identical to our scored v2
Runs invsqrt + DQI (default, s1, s2) verifiers and writes submission.zip if both PASS at the default salt."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path

INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/v4")
TMP.mkdir(exist_ok=True)
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "invsqrt").is_dir())
WANT = {"invsqrt.kmb": "864f81cd", "dqi.kmb": "5f901c55", "dfthcsos/submission.npz": "703843d3"}
files = {}
for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True):
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        for n, pref in WANT.items():
            if n in names and n not in files:
                b = f.read(n)
                if hashlib.sha256(b).hexdigest().startswith(pref):
                    files[n] = b
                    if n.startswith("dfthcsos/"):
                        for m in names:
                            if m.startswith("dfthcsos/angle_k_"):
                                files[m] = f.read(m)
assert all(n in files for n in WANT), [n for n in WANT if n not in files]
for n in ("invsqrt.kmb", "dqi.kmb"):
    (TMP / n).write_bytes(files[n])


def verify(script, path, salt=None):
    args = [sys.executable, script, str(path)] + (["--max-workers", "4"] if "dqi" in script else [])
    if salt:
        args += ["--salt", salt]
    v = subprocess.run(args, cwd=comp / "verifiers", env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")),
                       capture_output=True, text=True)
    r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
    print(script, salt, r.get("status"), r.get("score"), r.get("value_errors"), r.get("phase_errors"), flush=True)
    return r


res = {"invsqrt": verify("invsqrt/verify_submission_invsqrt.py", TMP / "invsqrt.kmb"),
       "dqi": verify("dqi/verify_submission_dqi.py", TMP / "dqi.kmb")}
for s in ("s1", "s2"):
    res["dqi_" + s] = verify("dqi/verify_submission_dqi.py", TMP / "dqi.kmb", s)
ref = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155375}
if res["invsqrt"]["status"] == "PASS" and res["dqi"]["status"] == "PASS":
    gm = (ref["invsqrt"] / res["invsqrt"]["score"] * ref["dqi"] / res["dqi"]["score"]
          * ref["dfthcsos"] / 294642299.7630277) ** (1 / 3)
    res["projected_gm"] = gm
    print("projected GM", gm, flush=True)
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(files):
            z.writestr(n, files[n])
    (OUT / "NOTICE_SYED_ASAD_ALI.txt").write_text(
        "Lineage: invsqrt and DQI derive from generators by Syed Asad Ali (Apache-2.0). invsqrt 864f81cd: maoxin123; "
        "DQI 5f901c55: boristown (gap-3 residue repair) on grantprater/Balram Srinivasan/Hayato Hikawa patches; "
        "chemistry 703843d3: Atakan Aldemir with maoxin123's loaders and Lera's table.\n")
(OUT / "v4_results.json").write_text(json.dumps(res, indent=1, default=str))
