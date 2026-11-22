"""m4: dump the public invsqrt bb676b95 (leotang09 3-T phase fold) to .kmx text for local analysis."""
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
        if hashlib.sha256(b).hexdigest()[:8] == "bb676b95":
            (OUT / "inv_bb676b95.kmb").write_bytes(b)
            km.Circuit.from_file(str(OUT / "inv_bb676b95.kmb")).to_file(str(OUT / "inv_bb676b95.kmx"), format="kmx")
            print("written", (OUT / "inv_bb676b95.kmx").stat().st_size)
            break
