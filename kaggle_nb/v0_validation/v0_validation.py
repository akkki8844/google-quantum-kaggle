"""v0 validation entry: the public 28.09 frontier, repackaged unchanged.

Copies the circuits from maoxin123's public notebook output (Apache-2.0; lineage: Syed Asad Ali's
generators, then maoxin123, boristown, hayatohikawa, Lera/marculera and others as credited
there), checks each file against the published SHA-256 prefixes, and writes submission.zip.
Purpose: prove that our packaging matches the grader. No circuit of ours yet.
"""
import hashlib
import os
import shutil
import zipfile
from pathlib import Path

EXPECT = {"invsqrt.kmb": "f2cdab0b", "dqi.kmb": "ccc2ab84", "dfthcsos/submission.npz": "f6e12f02"}
OUT = Path("/kaggle/working")
STAGE = Path("/tmp/stage")

src_zip = next(p for p in Path("/kaggle/input").rglob("submission.zip") if "google-verifiable" not in str(p))
print("source:", src_zip)
if STAGE.exists():
    shutil.rmtree(STAGE)
with zipfile.ZipFile(src_zip) as z:
    z.extractall(STAGE)
for rel, prefix in EXPECT.items():
    digest = hashlib.sha256((STAGE / rel).read_bytes()).hexdigest()
    print(rel, digest[:8])
    assert digest.startswith(prefix), f"{rel}: expected {prefix}, got {digest[:8]}"
angles = sorted((STAGE / "dfthcsos").glob("angle_k_*.kmb"), key=lambda p: int(p.stem.split("_")[-1]))
assert [int(p.stem.split("_")[-1]) for p in angles] == list(range(len(angles))), "angle_k files must have no gaps"

with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(STAGE).as_posix())
print("wrote", OUT / "submission.zip", os.path.getsize(OUT / "submission.zip"), "bytes")
print(sorted(zipfile.ZipFile(OUT / "submission.zip").namelist()))
