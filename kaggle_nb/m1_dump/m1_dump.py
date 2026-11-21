"""m1: dump the 37-qubit invsqrt d2148a7c (public milestone file, from our v14 output) to .kmx text for local analysis."""
import glob, hashlib, sys, zipfile
from pathlib import Path
INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
sys.path.insert(0, str(comp / "kickmix"))
import kickmix as km
for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True):
    f = zipfile.ZipFile(z)
    if "invsqrt.kmb" in f.namelist():
        b = f.read("invsqrt.kmb")
        print(z, hashlib.sha256(b).hexdigest()[:8])
        if hashlib.sha256(b).hexdigest()[:8] == "d2148a7c":
            (OUT / "inv_d2148a7c.kmb").write_bytes(b)
            km.Circuit.from_file(str(OUT / "inv_d2148a7c.kmb")).to_file(str(OUT / "inv_d2148a7c.kmx"), format="kmx")
            print("written", (OUT / "inv_d2148a7c.kmx").stat().st_size)
            break
