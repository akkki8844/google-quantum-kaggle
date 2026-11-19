"""v6 safe: invsqrt d0227f3c (maoxin123; PASS at 5 salts) + DQI 5f901c55 (boristown on Grant Prater's 575b043f;
0 failures on 8 draws incl. the public LB) + chemistry 4d63be3d (Leo Tang weights, maoxin123 loaders, Lera table).
All three files were already verified with the official verifiers in v5/v6b; this only assembles them by hash."""
import glob, hashlib, zipfile
from pathlib import Path
INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
WANT = {"invsqrt.kmb": "d0227f3c", "dqi.kmb": "5f901c55", "dfthcsos/submission.npz": "4d63be3d"}
got = {}
def h(b): return hashlib.sha256(b).hexdigest()[:8]
for z in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True):
    try:
        f = zipfile.ZipFile(z)
    except zipfile.BadZipFile:
        continue
    names = f.namelist()
    for n, want in WANT.items():
        if n in names and n not in got and h(f.read(n)) == want:
            got[n] = f.read(n)
            if n.startswith("dfthcsos/"):
                for m in names:
                    if m.startswith("dfthcsos/angle_k_"):
                        got[m] = f.read(m)
print({n: h(b) for n, b in got.items() if not n.startswith("dfthcsos/angle")}, sum(n.startswith("dfthcsos/angle") for n in got))
assert all(n in got for n in WANT) and sum(n.startswith("dfthcsos/angle") for n in got) == 15
with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in sorted(got):
        z.writestr(n, got[n])
(OUT / "NOTICE_SYED_ASAD_ALI.txt").write_text(
    "Lineage: invsqrt and DQI derive from generators by Syed Asad Ali (Apache-2.0). invsqrt d0227f3c: maoxin123. "
    "DQI 5f901c55: boristown on Grant Prater's 575b043f. Chemistry 4d63be3d: Leo Tang's weights on maoxin123's "
    "4,022-Toffoli loaders and Lera's (marculera) coded-word table.\n")
