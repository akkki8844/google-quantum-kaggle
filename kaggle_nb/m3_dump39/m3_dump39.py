"""m3: dump ThemBones' public 39-qubit invsqrt (danishlynxthembones/vqcc-thembones-milestone-set) to .kmx for local analysis."""
import hashlib, sys
from pathlib import Path
INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
sys.path.insert(0, str(comp / "kickmix"))
import kickmix as km
for p in INPUT.rglob("invsqrt.kmb"):
    if "verifiers" in str(p) or "sample_submission" in str(p): continue
    b = p.read_bytes(); k = hashlib.sha256(b).hexdigest()[:8]; print(p, k, flush=True)
    km.Circuit.from_file(str(p)).to_file(str(OUT / f"inv_{k}.kmx"), format="kmx")
