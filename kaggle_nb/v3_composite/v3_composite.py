"""v3: our v2 package (DQI 575b043f, chemistry 703843d3) + maoxin123's newer 44q invsqrt 563d0583
(QCC 28.17 notebook; Syed Asad Ali generator lineage, Apache-2.0). Verifies invsqrt officially, writes submission.zip.
DQI and chemistry files are byte-identical to the already-scored v2 package (58464611 = 28.15096)."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path

INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "invsqrt").is_dir())
files, inv = {}, None
for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True):
    with zipfile.ZipFile(z) as f:
        for n in f.namelist():
            b = f.read(n)
            h = hashlib.sha256(b).hexdigest()
            if n == "invsqrt.kmb" and h.startswith("563d0583"):
                inv = b
            if n == "dqi.kmb" and h.startswith("575b043f"):
                files["dqi.kmb"] = b
            if n == "dfthcsos/submission.npz" and h.startswith("703843d3"):
                files[n] = b
                for m in f.namelist():
                    if m.startswith("dfthcsos/angle_k_"):
                        files[m] = f.read(m)
assert inv is not None and "dqi.kmb" in files and "dfthcsos/submission.npz" in files, list(files)
files["invsqrt.kmb"] = inv
Path("/tmp/inv.kmb").write_bytes(inv)
v = subprocess.run([sys.executable, "invsqrt/verify_submission_invsqrt.py", "/tmp/inv.kmb"], cwd=comp / "verifiers",
                   env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")), capture_output=True, text=True)
r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
print("invsqrt", r.get("status"), r.get("score"), flush=True)
(OUT / "v3_invsqrt.json").write_text(json.dumps(r, indent=1))
if r.get("status") == "PASS":
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(files):
            z.writestr(n, files[n])
    ref = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155375}
    gm = (ref["invsqrt"] / r["score"] * ref["dqi"] / 346409762292.72 * ref["dfthcsos"] / 294642299.7630277) ** (1 / 3)
    print("projected GM", gm)
    (OUT / "NOTICE_SYED_ASAD_ALI.txt").write_text("Lineage: invsqrt and DQI derive from generators by Syed Asad Ali "
        "(Apache-2.0). invsqrt 563d0583 from maoxin123 (QCC 28.17); DQI/chemistry from Atakan Aldemir's public package "
        "(grantprater, boristown, Balram Srinivasan, maoxin123, Lera and others).\n")
