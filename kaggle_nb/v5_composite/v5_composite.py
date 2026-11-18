"""v5: v4 (invsqrt 864f81cd + DQI 5f901c55, both already verified by us in v4) with chemistry 4d63be3d
(Leo Tang's weights on maoxin123's 4,022-Toffoli loaders and Lera's coded-word table; Syed Asad Ali lineage).
Verifies the chemistry package with the official dfthcsos verifier and writes submission.zip if it PASSes and beats 703843d3."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path

INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/v5")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
WANT = {"invsqrt.kmb": "864f81cd", "dqi.kmb": "5f901c55", "dfthcsos/submission.npz": "4d63be3d"}
files = {}
def offer(n, b, sib):
    """n = logical name; b = bytes; sib = callable(name)->bytes|None for dfthcsos siblings."""
    if n in WANT and n not in files and hashlib.sha256(b).hexdigest().startswith(WANT[n]):
        files[n] = b
        if n.startswith("dfthcsos/"):
            for k in range(64):
                m = f"dfthcsos/angle_k_{k}.kmb"
                sb = sib(m)
                if sb is None:
                    break
                files[m] = sb
for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True):
    with zipfile.ZipFile(z) as f:
        names = set(f.namelist())
        for n in WANT:
            if n in names:
                offer(n, f.read(n), lambda m: f.read(m) if m in names else None)
# Kaggle auto-extracts zips: also scan loose files
for p in INPUT.rglob("*"):
    if p.is_file():
        rel = "dfthcsos/submission.npz" if (p.name == "submission.npz" and p.parent.name == "dfthcsos") else p.name
        if rel in WANT:
            base = p.parent
            offer(rel, p.read_bytes(), lambda m: (base / Path(m).name).read_bytes() if (base / Path(m).name).exists() else None)
print("INPUT TREE:", sorted(str(p) for p in INPUT.rglob("*") if p.is_file() and ("verifiers" not in str(p)) and ("kickmix" not in str(p)))[:80], flush=True)
assert all(n in files for n in WANT), [n for n in WANT if n not in files]
pkg = TMP / "dfthcsos"
pkg.mkdir(parents=True, exist_ok=True)
for n, b in files.items():
    if n.startswith("dfthcsos/"):
        (pkg / Path(n).name).write_bytes(b)
v = subprocess.run([sys.executable, "dfthcsos/verify.py", str(pkg)], cwd=comp / "verifiers",
                   env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")), capture_output=True, text=True)
r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
print("chem", r.get("status"), r.get("score"), r.get("lambda_eff"), r.get("toffoli_count"), r.get("num_qubits"), flush=True)
ref = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155375}
if r.get("status") == "PASS" and r["score"] < 294642299.7630277:
    gm = (ref["invsqrt"] / 74490.94706 * ref["dqi"] / 346404203610.0 * ref["dfthcsos"] / r["score"]) ** (1 / 3)
    r["projected_gm"] = gm
    print("projected GM", gm, flush=True)
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(files):
            z.writestr(n, files[n])
    (OUT / "NOTICE_SYED_ASAD_ALI.txt").write_text(
        "Lineage: invsqrt and DQI derive from generators by Syed Asad Ali (Apache-2.0). invsqrt 864f81cd: maoxin123; "
        "DQI 5f901c55: boristown on Grant Prater's 575b043f; chemistry 4d63be3d: Leo Tang's weights on maoxin123's "
        "4,022-Toffoli loaders and Lera's (marculera) coded-word table.\n")
(OUT / "v5_results.json").write_text(json.dumps(r, indent=1, default=str))
