"""List the attached Syed Asad Ali dataset (pinned to version 12) and copy its source archives to output.
Apache-2.0; NOTICE kept with the copies (lineage: Syed Asad Ali, original builder)."""
import shutil
from pathlib import Path
out = Path("/kaggle/working/syed_v12_src")
out.mkdir(parents=True, exist_ok=True)
for p in sorted(Path("/kaggle/input").rglob("*")):
    if "google-verifiable" in str(p):
        continue
    print(p, p.stat().st_size if p.is_file() else "<dir>")
    if p.is_file() and (("source" in p.name) or p.name in ("NOTICE", "LICENSE", "README.md", "SHA256SUMS")):
        shutil.copy(p, out / p.name)
print("copied:", sorted(x.name for x in out.iterdir()))
