"""c5: dump the 15 chemistry loaders of our v18 set (maoxin123 3,357-T loaders) to .kmx text with their static and mean
Toffoli counts, plus the angle-word table and tree edges, for local analysis."""
import glob, hashlib, io, json, sys, zipfile
from pathlib import Path
import numpy as np
INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
sys.path.insert(0, str(comp / "kickmix"))
import kickmix as km
z = next(z for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True) if "v18" in z)
f = zipfile.ZipFile(z); res = {}
for n in sorted(f.namelist()):
    if n.startswith("dfthcsos/angle_k_"):
        p = OUT / Path(n).name; p.write_bytes(f.read(n))
        c = km.Circuit.from_file(str(p)); c.to_file(str(p.with_suffix(".kmx")), format="kmx")
        st = c.operation_statistics(); res[n] = dict(q=c.num_qubits, tof=st.num_toffoli_like, ops=st.num_instructions)
        p.unlink()
arr = dict(np.load(io.BytesIO(f.read("dfthcsos/submission.npz")), allow_pickle=False))
np.savez(OUT / "words.npz", angle_words=arr["angle_words"], tree_edges=arr["tree_edges"])
print(json.dumps(res, indent=0), "sum static", sum(v["tof"] for v in res.values()), flush=True)
