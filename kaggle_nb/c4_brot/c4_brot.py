"""c4: would 6-bit angle words work? Requantize the public table eb4b5d79's 7-bit angle words to 6 bits and evaluate
the official metrics (error_1, error_2, E_gap, lambda_eff) and the heuristic Toffoli / qubit costs at brot 7 and 6."""
import glob, hashlib, sys, zipfile, io
from pathlib import Path
import numpy as np
INPUT = Path("/kaggle/input")
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
sys.path[:0] = [str(comp / "kickmix"), str(comp / "verifiers")]
from dfthcsos import verify as V
z = next(z for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True) if "v17" in z)
f = zipfile.ZipFile(z); raw = f.read("dfthcsos/submission.npz")
print("npz", hashlib.sha256(raw).hexdigest()[:8], flush=True)
arr = dict(np.load(io.BytesIO(raw), allow_pickle=False))
inputs = V.load_inputs(arr)
w7 = np.asarray(arr["angle_words"]).astype(np.int64); te = arr["tree_edges"]
print("angle_words", w7.shape, w7.min(), w7.max(), flush=True)
m7 = V.evaluate_words(inputs, w7, te, 7); print("brot 7", {k: m7[k] for k in ("error_1", "error_2", "E_gap", "lambda_eff", "Lambda") if k in m7}, flush=True)
for name, w6 in (("round", (np.round(w7 / 2).astype(np.int64)) % 64), ("floor", (w7 // 2) % 64)):
    m6 = V.evaluate_words(inputs, w6, te, 6); print("brot 6", name, {k: m6[k] for k in ("error_1", "error_2", "E_gap", "lambda_eff", "Lambda") if k in m6}, flush=True)
R, B, C = np.shape(arr["W"]) if "W" in arr else (9, 127, 46)
for b in (7, 6):
    print("heuristic brot", b, V.heuristic_costs(76, R, B, C, b), flush=True)
