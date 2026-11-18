"""v6b: finish v6. Verify maoxin123's new invsqrt d0227f3c (5 salts) and DQI candidates f2673ad2 (Ghazaros
Barseghyan), bad74d3e (Aviral Trivedi), 8a1bda64 (Atakan Aldemir) at the default salt, then the best at 5 salts.
Known from v6 (same verifiers): DQI 40294f00 default 3.46040797e11 but 2 / 1 failed shots at salts s1 / s2;
5f901c55 0 failures at 7 salts (~3.46404e11); chemistry 4d63be3d 294,642,170.57.
Writes submission.zip (best default-salt cost per task = public LB) and submission_robust.zip (lowest mean cost
over salts = expected private LB)."""
import glob, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path

INPUT, OUT, TMP = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/v6b")
TMP.mkdir(parents=True, exist_ok=True)
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
SALTS = ["", "s1", "s2", "private-a", "private-b"]
INV_NEW, INV_OLD = "d0227f3c", "864f81cd"
DQI_NEW = ["f2673ad2", "bad74d3e", "8a1bda64"]
files = {}


def h(b):
    return hashlib.sha256(b).hexdigest()[:8]


def offer(kind, b):
    files.setdefault((kind, h(b)), b)


for z in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True):
    try:
        f = zipfile.ZipFile(z)
    except zipfile.BadZipFile:
        continue
    for n in f.namelist():
        if n in ("invsqrt.kmb", "dqi.kmb"):
            offer(n[:-4], f.read(n))
    if "dfthcsos/submission.npz" in f.namelist() and h(f.read("dfthcsos/submission.npz")) == "4d63be3d":
        chem = {n[len("dfthcsos/"):]: f.read(n) for n in f.namelist() if n.startswith("dfthcsos/")}
for p in INPUT.rglob("*.kmb"):
    if "verifiers" in str(p) or "angle_k" in p.name:
        continue
    if p.name.startswith("invsqrt"):
        offer("invsqrt", p.read_bytes())
    elif p.name.startswith("dqi"):
        offer("dqi", p.read_bytes())
for p in INPUT.rglob("submission.npz"):
    if p.parent.name == "dfthcsos" and h(p.read_bytes()) == "4d63be3d" and "chem" not in globals():
        chem = {q.name: q.read_bytes() for q in p.parent.iterdir() if q.is_file()}
print("found", sorted(files), "chem files", len(chem), flush=True)


def run(args):
    v = subprocess.run([sys.executable] + args, cwd=comp / "verifiers",
                       env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")), capture_output=True, text=True)
    try:
        r = json.loads(v.stdout[v.stdout.index("{"):])
        r.pop("shot_operation_counts", None)
    except ValueError:
        r = {"status": "UNPARSED", "tail": v.stdout[-800:], "err": v.stderr[-800:]}
    return r


res = {"inv": {INV_OLD: [{"salt": s, "status": "PASS", "score": 74490.94707587373} for s in SALTS]},
       "dqi": {"5f901c55": [{"salt": s, "status": "PASS", "score": 346404203610.1118, "value_errors": 0} for s in SALTS],
               "40294f00": [{"salt": "", "status": "PASS", "score": 346040797181.99023, "value_errors": 0},
                            {"salt": "s1", "status": "PASS", "score": 348982391127.45844, "value_errors": 2},
                            {"salt": "s2", "status": "PASS", "score": 347629847451.4387, "value_errors": 1},
                            {"salt": "private-a", "status": "PASS", "score": 346040651606.2876, "value_errors": 0},
                            {"salt": "private-b", "status": "PASS", "score": 346040565014.6382, "value_errors": 0}]}}


def save():
    (OUT / "v6b_results.json").write_text(json.dumps(res, indent=1, default=str))


if ("invsqrt", INV_NEW) in files:
    p = TMP / "inv_new.kmb"
    p.write_bytes(files[("invsqrt", INV_NEW)])
    for s in SALTS:
        r = run(["invsqrt/verify_submission_invsqrt.py", str(p)] + (["--salt", s] if s else []))
        row = {"salt": s, "status": r.get("status"), "score": r.get("score"), "worst": r.get("worst_error_over_tolerance"),
               "q": r.get("num_qubits"), "toffoli": r.get("toffoli_count"), "err": r.get("error")}
        print("inv", INV_NEW, row, flush=True)
        res["inv"].setdefault(INV_NEW, []).append(row)
        save()
        if row["status"] != "PASS":
            break


def dqi_run(c, s):
    p = TMP / f"dqi_{c}.kmb"
    p.write_bytes(files[("dqi", c)])
    r = run(["dqi/verify_submission_dqi.py", str(p), "--max-workers", "4"] + (["--salt", s] if s else []))
    row = {"salt": s, "status": r.get("status"), "score": r.get("score"), "value_errors": r.get("value_errors"),
           "phase_errors": r.get("phase_errors"), "err": r.get("error")}
    print("dqi", c, row, flush=True)
    res["dqi"].setdefault(c, []).append(row)
    save()
    return row


first = {c: dqi_run(c, "") for c in DQI_NEW if ("dqi", c) in files}
for c, r in sorted(first.items(), key=lambda kv: kv[1]["score"] or 1e99):
    if r["status"] == "PASS" and r["score"] < 346404203610.1118:
        for s in SALTS[1:]:
            if dqi_run(c, s)["status"] != "PASS":
                break


def default(rows):
    return next((r["score"] for r in rows if r["salt"] == "" and r["status"] == "PASS"), float("inf"))


def mean(rows):
    ok = [r["score"] for r in rows if r["status"] == "PASS"]
    return sum(ok) / len(ok) if len(rows) == len(SALTS) and len(ok) == len(SALTS) else float("inf")


ref = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155375}
CHEM = 294642170.565169
out = {}
for tag, fn in (("public", default), ("robust", mean)):
    bi = min((c for c in res["inv"] if mean(res["inv"][c]) < float("inf")), key=lambda c: fn(res["inv"][c]))
    bd = min((c for c in res["dqi"] if fn(res["dqi"][c]) < float("inf")), key=lambda c: fn(res["dqi"][c]))
    gm = (ref["invsqrt"] / default(res["inv"][bi]) * ref["dqi"] / default(res["dqi"][bd]) * ref["dfthcsos"] / CHEM) ** (1 / 3)
    gm_mean = (ref["invsqrt"] / mean(res["inv"][bi]) * ref["dqi"] / mean(res["dqi"][bd]) * ref["dfthcsos"] / CHEM) ** (1 / 3)
    out[tag] = {"inv": bi, "dqi": bd, "chem": "4d63be3d", "gm_default_salt": gm, "gm_mean_over_salts": gm_mean}
    name = "submission.zip" if tag == "public" else "submission_robust.zip"
    with zipfile.ZipFile(OUT / name, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("invsqrt.kmb", files[("invsqrt", bi)])
        z.writestr("dqi.kmb", files[("dqi", bd)])
        for n in sorted(chem):
            z.writestr("dfthcsos/" + n, chem[n])
res["picks"] = out
print("PICKS", json.dumps(out), flush=True)
(OUT / "NOTICE_SYED_ASAD_ALI.txt").write_text(
    "Lineage: invsqrt and DQI derive from generators by Syed Asad Ali (Apache-2.0). invsqrt 864f81cd / d0227f3c: "
    "maoxin123. DQI: 5f901c55 boristown on Grant Prater's 575b043f; 40294f00 Balram Srinivasan; 8a1bda64 Atakan "
    "Aldemir; f2673ad2 Ghazaros Barseghyan; bad74d3e Aviral Trivedi. Chemistry 4d63be3d: Leo Tang's weights on "
    "maoxin123's 4,022-Toffoli loaders and Lera's (marculera) coded-word table.\n")
save()
