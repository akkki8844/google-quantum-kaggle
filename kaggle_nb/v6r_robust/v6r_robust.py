"""v6 robust: republish v6b's submission_robust.zip (invsqrt d0227f3c + DQI 8a1bda64 + chem 4d63be3d) as submission.zip."""
import hashlib, shutil, zipfile
from pathlib import Path
src = next(Path("/kaggle/input").rglob("submission_robust.zip"))
z = zipfile.ZipFile(src)
got = {n: hashlib.sha256(z.read(n)).hexdigest()[:8] for n in ("invsqrt.kmb", "dqi.kmb", "dfthcsos/submission.npz")}
print(got)
assert got == {"invsqrt.kmb": "d0227f3c", "dqi.kmb": "8a1bda64", "dfthcsos/submission.npz": "4d63be3d"}
shutil.copy(src, "/kaggle/working/submission.zip")
for p in Path("/kaggle/input").rglob("NOTICE_SYED_ASAD_ALI.txt"):
    shutil.copy(p, "/kaggle/working/NOTICE_SYED_ASAD_ALI.txt"); break
