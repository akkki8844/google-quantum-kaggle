"""P0 on Kaggle: confirm the three official verifiers score the sample submission at 1.0.

Runs inside a Kaggle kernel attached to the competition data. Uses the competition's own
kickmix build (what the leaderboard metric uses). Writes /kaggle/working/p0_results.json
and prints a short table. Nothing here talks to the network.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def find_competition_dir() -> Path:
    """The mount point varies between Kaggle images; find the folder holding verifiers/."""
    for p in sorted(Path("/kaggle/input").rglob("verifiers")):
        if (p / "dqi").is_dir():
            return p.parent
    raise FileNotFoundError("no verifiers/ under /kaggle/input: " + str(sorted(Path("/kaggle/input").rglob("*"))[:50]))


COMP = find_competition_dir()
OUT = Path("/kaggle/working")
VERIFIERS = COMP / "verifiers"
SAMPLE = COMP / "sample_submission"

# Reference costs for ratio = ref / cost (paper + SCORING_README).
REF = {"invsqrt": 260 * 7988, "dqi": None, "dfthcsos": 3355599588.3155365}


def machine_info():
    info = {"cpus": os.cpu_count(), "python": sys.version.split()[0]}
    try:
        with open("/proc/meminfo") as f:
            info["mem_gb"] = round(int(f.readline().split()[1]) / 1e6, 1)
    except OSError:
        pass
    return info


def run(name, args, cwd):
    env = dict(os.environ)
    # The competition ships kickmix.so for this image; put its folder first on the path.
    env["PYTHONPATH"] = str(COMP / "kickmix") + os.pathsep + env.get("PYTHONPATH", "")
    t = time.time()
    p = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True)
    dt = time.time() - t
    out = p.stdout.strip()
    try:
        summary = json.loads(out[out.index("{"):])
    except ValueError:
        summary = {"status": "UNPARSED", "stdout_tail": out[-2000:], "stderr_tail": p.stderr[-2000:]}
    summary["_seconds"] = round(dt, 1)
    summary["_returncode"] = p.returncode
    print(f"[{name}] rc={p.returncode} {dt:.0f}s status={summary.get('status')} score={summary.get('score')}", flush=True)
    return summary


def main():
    print("machine:", machine_info(), flush=True)
    print("files:", sorted(x.name for x in COMP.iterdir()), flush=True)
    results = {"machine": machine_info()}

    # Chemistry first: ~25 s, cheapest signal that the environment works.
    results["dfthcsos"] = run("dfthcsos", ["dfthcsos/verify.py", str(SAMPLE / "dfthcsos")], VERIFIERS)
    results["dqi"] = run("dqi", ["dqi/verify_submission_dqi.py", str(SAMPLE / "dqi.kmb"), "--max-workers", "4"], VERIFIERS)
    results["invsqrt"] = run("invsqrt", ["invsqrt/verify_submission_invsqrt.py", str(SAMPLE / "invsqrt.kmb")], VERIFIERS)

    (OUT / "p0_results.json").write_text(json.dumps(results, indent=2, default=str))
    print("\n=== summary ===")
    for task in ("invsqrt", "dqi", "dfthcsos"):
        r = results[task]
        print(f"{task:9s} status={r.get('status')} score={r.get('score')} qubits={r.get('num_qubits')} "
              f"toffoli={r.get('toffoli_count')} seconds={r.get('_seconds')}")


if __name__ == "__main__":
    main()
