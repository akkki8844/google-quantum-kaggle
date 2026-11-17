"""Watch the competition's public notebooks for a new top-scoring entry.

Kaggle's API sorts notebooks by public score but does not return the score, so this tracks the
top N refs and their version numbers and reports any change against the last saved state
(tools/.frontier_state.json). Run once, or with --loop SECONDS to keep watching; it exits with a
message as soon as something changes, so a background run re-wakes the session.

Usage: python tools/frontier_watch.py [--top 5] [--loop 7200]
"""
import argparse
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "google-verifiable-quantum-circuit-compilation"
STATE = Path(__file__).with_name(".frontier_state.json")


def snapshot(api: KaggleApi, top: int) -> list[dict]:
    ks = api.kernels_list(competition=COMP, sort_by="scoreDescending", page_size=top)
    return [{"ref": k.ref, "title": k.title, "version": k.current_version_number,
             "last_run": str(k.last_run_time)} for k in ks[:top]]


def diff(old: list[dict], new: list[dict]) -> list[str]:
    o = {k["ref"]: k for k in old}
    out = []
    for rank, k in enumerate(new, 1):
        prev = o.get(k["ref"])
        if prev is None:
            out.append(f"NEW at #{rank}: {k['ref']}  \"{k['title']}\"  v{k['version']}")
        elif prev["version"] != k["version"]:
            out.append(f"UPDATED #{rank}: {k['ref']}  v{prev['version']} -> v{k['version']}  \"{k['title']}\"")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--loop", type=int, default=0, help="seconds between checks; 0 = check once")
    a = ap.parse_args()
    api = KaggleApi()
    api.authenticate()
    old = json.loads(STATE.read_text()) if STATE.exists() else []
    while True:
        try:
            new = snapshot(api, a.top)
        except Exception as e:  # network hiccup: retry next round
            print("fetch failed:", e, flush=True)
            new = old
        changes = diff(old, new) if old else []
        STATE.write_text(json.dumps(new, indent=1))
        if not old:
            print("baseline saved:", *[k["ref"] for k in new], sep="\n  ")
        for c in changes:
            print(c, flush=True)
        if changes or not a.loop:
            return
        old = new
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
