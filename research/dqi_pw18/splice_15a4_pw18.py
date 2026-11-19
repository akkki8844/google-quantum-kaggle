"""PLAN ONLY (not run): build 15a4b740 + pw18 on Kaggle by a section splice, then verify.

15a4b740 (ozertuu), e208d350 and 4471b493 (koushikrudra, e208d350 + 18 pair-window sites) share every byte of
every KMB section except:
  * 8 local edits e208d350 makes to 15a4b740 at ops 18,946,289..21,375,825 (conditional-CCZ uncompute), and
  * the 18 pair-window edits 4471b493 makes to e208d350 at ops 35,655,134..40,175,434.
So 15a4b740 + pw18 = 15a4b740 sections up to the first pw site, then 4471b493 sections from that point on.
Cut points were measured locally on 2026-11-19 (research/dqi_pw18/FINDINGS.md) and are re-checked below.

Usage on Kaggle: python splice_15a4_pw18.py <15a4b740 dqi.kmb> <4471b493 dqi.kmb> <out.kmb>
Then run the official verifier at '' and a few salts (see FINDINGS.md). Nothing here submits anything.
"""
import hashlib
import struct
import sys

import numpy as np

MAGIC = bytes.fromhex("d750c7d5c329d326e3cc9f6834f2b8bf")
SHA = {"a": "15a4b7403356092e", "b": "4471b4933c18fd65"}
# section -> (cut in 15a4b740 entries, cut in 4471b493 entries); entries are uint8 for section 1, uint32 otherwise
CUT = {1: (35655138, 35655146), 2: (197138, 197138), 3: (197138, 197138), 4: (197042, 197042),
       5: (16604563, 16604571), 6: (16604563, 16604571), 7: (211123, 211123), 8: (14209840, 14209840),
       9: (13955011, 13955027)}
CHECK = 1_000_000  # entries before each cut / at the end that must agree between the two files (alignment check)
PRE_W = {4: 100_000}  # section 4 (CCX targets): 15a4b740 and e208d350 differ up to entry 60,279, so check less there
# common suffix of 15a4b740 and 4471b493 per section (= common suffix of e208d350 and 4471b493), measured locally
SUF = {1: 16679212, 2: 107177, 3: 107177, 4: 107177, 5: 10129233, 6: 10129233, 7: 33470, 8: 5836930, 9: 6175197}
EXPECT = dict(bits=74563, ops=56854693, q3=366198, q2=30631129, q1=267290, bt=20561917, cond=20797236)


def parse(path):
    data = open(path, "rb").read()
    assert data[:16] == MAGIC
    assert hashlib.sha256(data).hexdigest()[:16] in SHA.values(), path
    h = list(struct.unpack_from("<4I9Q", data, 16))
    pos, regs, sec = 104, [], {}
    while pos < len(data):
        i, n = struct.unpack_from("<QQ", data, pos)
        if i == 0:
            regs.append(data[pos:pos + 16 + n])
        else:
            sec[i] = data[pos + 16:pos + 16 + n]
        pos += 16 + n
    return h, regs, sec


def main(a_path, b_path, out_path):
    ha, ra, sa = parse(a_path)   # 15a4b740
    hb, rb, sb = parse(b_path)   # 4471b493
    assert ra == rb, "register chunks differ"
    new = {}
    for i in range(1, 10):
        dt = np.uint8 if i == 1 else np.uint32
        A, B = np.frombuffer(sa[i], dt), np.frombuffer(sb[i], dt)
        ca, cb = CUT[i]
        w = min(CHECK, ca, PRE_W.get(i, CHECK))
        assert (A[ca - w:ca] == B[cb - w:cb]).all(), f"section {i}: no alignment before cut"
        s = min(CHECK, SUF[i])  # shared tail after the last pw18 site
        assert (A[len(A) - s:] == B[len(B) - s:]).all(), f"section {i}: suffix differs"
        new[i] = np.concatenate([A[:ca], B[cb:]])
    h = list(hb)                 # bits = 74563 (18 new classical bits from 4471b493)
    h[4], h[5], h[6], h[7], h[8], h[9] = (len(new[1]), len(new[2]), len(new[5]), len(new[7]), len(new[8]), len(new[9]))
    got = dict(bits=h[2], ops=h[4], q3=h[5], q2=h[6], q1=h[7], bt=h[8], cond=h[9])
    assert got == EXPECT, got
    chunks = [MAGIC + struct.pack("<4I9Q", *h)] + ra
    for i in range(1, 10):
        b = new[i].tobytes()
        chunks.append(struct.pack("<QQ", i, len(b)) + b)
    for i in (10, 11):
        if i in sb:
            chunks.append(struct.pack("<QQ", i, len(sb[i])) + sb[i])
    blob = b"".join(chunks)
    open(out_path, "wb").write(blob)
    print("wrote", out_path, len(blob), "bytes sha256", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:4])
