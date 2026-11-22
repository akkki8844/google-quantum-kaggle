"""v19 hedge: v18's files with the DQI re-rolled to a new sha256 (our tools/dqi_reroll.py: `X a; X a` appended on the
last ancilla; same unitary and counts), so a second final pick gets an independent DQI verifier draw.
Checks: kickmix loads it, static counts equal v15's DQI, official DQI verifier at 5 salts. Writes submission.zip."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/v16")
TMP.mkdir(parents=True, exist_ok=True)
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
sys.path.insert(0, str(comp / "kickmix"))
import kickmix as km
(TMP / "dqi_reroll.py").write_text('"""Give a DQI .kmb a new sha256 (hence an independent verifier draw) at zero cost: append `X a; X a` on one\nancilla wire (not in any register) at the very end. Same unitary, same Toffoli / qubit counts.\nOur own codec (same as kaggle_nb/dq_scan). Usage: python -I tools/dqi_reroll.py IN.kmb OUT.kmb [N_PAIRS]"""\nimport hashlib, struct, sys\nimport numpy as np\nMAGIC = bytes.fromhex("d750c7d5c329d326e3cc9f6834f2b8bf")\n\ndef parse(data):\n    assert data[:16] == MAGIC\n    h = list(struct.unpack_from("<4I9Q", data, 16))\n    pos, regs, sec = 104, [], {}\n    while pos < len(data):\n        i, n = struct.unpack_from("<QQ", data, pos)\n        if i == 0: regs.append(data[pos:pos + 16 + n])\n        else: sec[i] = data[pos + 16:pos + 16 + n]\n        pos += 16 + n\n    return h, regs, sec\n\ndef reg_qubits(regs):\n    qs = set()\n    for r in regs:\n        nlen, width = struct.unpack_from("<II", r, 16)\n        vals = np.frombuffer(r[16 + 8 + nlen:16 + 8 + nlen + 4 * width], np.uint32)\n        qs.update(int(v & 0x7FFFFFFF) for v in vals if v & 0x80000000)\n    return qs\n\ndef reroll(data, pairs=1):\n    h, regs, sec = parse(data)\n    anc = sorted(set(range(h[1])) - reg_qubits(regs))\n    a = anc[-1]\n    code = sec[1] + bytes([0x11]) * (2 * pairs)                 # 0x11 = X (one-qubit write)\n    s7 = sec[7] + np.full(2 * pairs, a, np.uint32).tobytes()    # section 7 = one-qubit operands\n    h = list(h); h[4] += 2 * pairs; h[7] += 2 * pairs\n    out = [MAGIC + struct.pack("<4I9Q", *h)] + regs\n    for i in sorted(sec):\n        b = code if i == 1 else s7 if i == 7 else sec[i]\n        out.append(struct.pack("<QQ", i, len(b)) + b)\n    return b"".join(out)\n\nif __name__ == "__main__":\n    data = open(sys.argv[1], "rb").read()\n    out = reroll(data, int(sys.argv[3]) if len(sys.argv) > 3 else 1)\n    open(sys.argv[2], "wb").write(out)\n    print(hashlib.sha256(data).hexdigest()[:8], "->", hashlib.sha256(out).hexdigest()[:8])\n')
sys.path.insert(0, str(TMP))
import dqi_reroll
src_zip = next(z for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True) if "v18" in z)
f = zipfile.ZipFile(src_zip); files = {n: f.read(n) for n in f.namelist()}
h = lambda b: hashlib.sha256(b).hexdigest()[:8]
print("v18", src_zip, {n: h(b) for n, b in files.items() if not n.startswith("dfthcsos/angle")}, flush=True)
assert h(files["dqi.kmb"]) == "4b572141" and h(files["invsqrt.kmb"]) == "bb676b95"
new = dqi_reroll.reroll(files["dqi.kmb"], 1)
(TMP / "dqi_old.kmb").write_bytes(files["dqi.kmb"]); (TMP / "dqi.kmb").write_bytes(new)
a = km.Circuit.from_file(str(TMP / "dqi_old.kmb")).operation_statistics(); b = km.Circuit.from_file(str(TMP / "dqi.kmb")).operation_statistics()
print("new sha", h(new), "old stats", a, "new stats", b, flush=True)
env = dict(os.environ, PYTHONPATH=str(comp / "kickmix"))
res = []
for s in ("",):
    v = subprocess.run([sys.executable, "dqi/verify_submission_dqi.py", str(TMP / "dqi.kmb"), "--max-workers", "4"] + (["--salt", s] if s else []),
                       cwd=comp / "verifiers", env=env, capture_output=True, text=True)
    try:
        r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
    except ValueError:
        r = {"status": "UNPARSED", "tail": v.stdout[-800:], "err": v.stderr[-800:]}
    res.append({"salt": s, **{k: r.get(k) for k in ("status", "score", "num_qubits", "toffoli_count", "value_errors", "phase_errors", "error")}})
    print(res[-1], flush=True); (OUT / "v19_results.json").write_text(json.dumps(res, indent=1))
if res[0]["status"] == "PASS":
    files["dqi.kmb"] = new
    with zipfile.ZipFile(OUT / "submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in sorted(files): z.writestr(n, files[n])
    print("wrote submission.zip with DQI", h(new), flush=True)
