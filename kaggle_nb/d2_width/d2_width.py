"""d2: DQI 4b572141 width profile with our sound exact-value tracker (from dq_scan). If at every op at least one
wire is provably |0> and untouched, the circuit can be relabelled onto 899 wires at zero Toffoli cost."""
"""dq_scan: find and apply Toffoli-saving rewrites in DQI 4471b493 (koushikrudra, on e208d350 / 15a4b740, Syed Asad Ali line).

Our own code; only numpy, kickmix and the official verifiers are used. Two exact rewrites:

(a) conditional-product site (generalises e208d350's 8 edits over 15a4b740):
      R/zero p ... CCX a b p (i) ... CZ p c if m (u, the only use of p) ... HMR p -> b' (j) ... CZ a b if b' (f)
    with a, b unchanged on (i, f), and f the only reader of b' before b' is overwritten.
    Rewrite: delete i, u -> CCZ a b c if m, delete f.  p is never computed; the HMR of |0> is harmless.
    Static Toffolis unchanged, mean -P(m=1) + 1 = -0.5.
(b) pair-window chain (generalises boristown / koushikrudra pw sites):
      CCX/CCZ ops sharing a control pair {a, b} with a, b unchanged between them, and a spare qubit w that is
      provably |0> at the first op and untouched until the last.  first: CCX a b w + (CX w t | CZ w c);
      middle: CX w t | CZ w c;  last: same + HMR w -> new bit; CZ a b if bit.  Saves (n - 1) Toffolis.
Zero tracking per qubit: ancillas (qubits not in any quantum register) start |0>; R / HMR reset to |0>;
diagonal ops and control roles keep |0>; any other write makes the qubit unknown.

Run on Kaggle: finds 4471b493 under /kaggle/input, scans, applies if the saving is >= APPLY_MIN mean Toffolis,
round-trip-checks the codec, writes /kaggle/working/dqi.kmb, prints sha256 + operation_statistics, runs the
official DQI verifier at '', s1, s2, private-a, and saves dq_scan_results.json + dq_sites.json.
Local test: python dq_scan.py --local <slice.kmb> [--ancilla-from N] [--unitary] (slices assume ancillas zero;
--unitary makes chain ends uncompute with CCX so seeded simulations of original and patch can be compared exactly).

RESULTS on 4471b493 (2026-11-19, Kaggle versions 1-4 of govindmulchandani/qcc-dq-scan):
* Opcodes: 0x81 = PUSH_CONDITION if b, 0x01 = POP_CONDITION (ops in between are conditional), 0x00/0x80 = NEG
  (global phase), 0x40-0x42 = BIT_STORE0 / BIT_STORE1 / BIT_INVERT. HMR and R reset to |0>.
* v1 (reset-only zero tracking, controls-unchanged grouping): 312 candidate pairs, 0 placeable.
* v2 (symbolic tracker that ignored PUSH/POP blocks): 239 chains, -239 static Toffolis, sha 100d964c...; official
  verifier FAILED at '', s1, s2, private-a with ~65% failed shots. Unsound: ops inside conditional blocks were treated
  as unconditional.
* v3/v4 (sound): 3,066 repeated-product groups (up to 3,354 Toffolis if a spare existed) but 0 placeable: no qubit is
  provably |0> and untouched across any window (46,323 zero gaps on 86 qubits, all too short). Pattern (a): 0 new
  sites (the 8 e208d350 sites are already in 4471b493). Window lengths (ops): median 69,704, 90% 171,515.
* Extra dedicated spare qubits: k spares save exactly 108*k Toffolis (k = 1..8), but one extra qubit costs
  (891/890)^2 - 1 = 0.225% ~ 823 Toffolis. Not worth it.
* Verdict: no cheaper sound DQI from these two patterns on 4471b493. Nothing built in v3/v4, nothing submitted.
"""
import glob, hashlib, json, os, struct, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np

MAGIC = bytes.fromhex("d750c7d5c329d326e3cc9f6834f2b8bf")
WANT = "4b572141"
GAPS = []
APPLY_MIN = 10.0
SALTS = ("", "s1", "s2", "private-a")
T0 = time.time()
def log(*a): print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)
SH = np.int64(26)            # key = id << 26 | pos  (pos < 2^26)
MASK = np.int64((1 << 26) - 1)

# ---------------- codec ----------------
def parse(data):
    assert data[:16] == MAGIC
    h = list(struct.unpack_from("<4I9Q", data, 16))
    pos, regs, sec = 104, [], {}
    while pos < len(data):
        i, n = struct.unpack_from("<QQ", data, pos)
        if i == 0: regs.append(data[pos:pos + 16 + n])
        else: sec[i] = data[pos + 16:pos + 16 + n]
        pos += 16 + n
    return h, regs, sec

def masks(code):
    b = code & 127
    return (np.isin(b, [0x30, 0x31, 0x32]), np.isin(b, [0x20, 0x21, 0x22]), np.isin(b, [0x10, 0x11, 0x12, 0x13, 0x50]),
            np.isin(b, [0x40, 0x41, 0x42, 0x50]), code >= 128)

def decode(sec):
    code = np.frombuffer(sec[1], np.uint8).copy()
    assert len(sec.get(10, b"")) == 0 and not ((code & 127) == 0x13).any(), "Z_POW not supported"
    m3, m2, m1, mb, mc = masks(code)
    N = len(code); q = np.full((N, 3), -1, np.int32)
    u = lambda i: np.frombuffer(sec[i], np.uint32).astype(np.int32)
    q[m3, 0], q[m3, 1], q[m3, 2] = u(2), u(3), u(4)
    q[m2, 0], q[m2, 1] = u(5), u(6)
    q[m1, 0] = u(7)
    bit = np.full(N, -1, np.int32); bit[mb] = u(8)
    cnd = np.full(N, -1, np.int32); cnd[mc] = u(9)
    return code, q, bit, cnd

def encode(h, regs, sec_extra, code, q, bit, cnd):
    m3, m2, m1, mb, mc = masks(code)
    s = {1: code.astype(np.uint8), 2: q[m3, 0], 3: q[m3, 1], 4: q[m3, 2], 5: q[m2, 0], 6: q[m2, 1], 7: q[m1, 0],
         8: bit[mb], 9: cnd[mc]}
    for i in range(2, 10):
        assert (s[i] >= 0).all(), i
        s[i] = s[i].astype(np.uint32)
    h = list(h)
    h[4], h[5], h[6], h[7], h[8], h[9] = len(s[1]), len(s[2]), len(s[5]), len(s[7]), len(s[8]), len(s[9])
    out = [MAGIC + struct.pack("<4I9Q", *h)] + regs
    for i in range(1, 10):
        b = s[i].tobytes(); out.append(struct.pack("<QQ", i, len(b)) + b)
    for i in (10, 11):
        if i in sec_extra: out.append(struct.pack("<QQ", i, len(sec_extra[i])) + sec_extra[i])
    return b"".join(out)

def reg_bits(regs):
    bs = set()
    for r in regs:
        nlen, width = struct.unpack_from("<II", r, 16)
        vals = np.frombuffer(r[16 + 8 + nlen:16 + 8 + nlen + 4 * width], np.uint32)
        bs.update(int(v) for v in vals if not v & 0x80000000)
    return bs

def reg_qubits(regs):
    qs = set()
    for r in regs:
        nlen, width = struct.unpack_from("<II", r, 16)
        vals = np.frombuffer(r[16 + 8 + nlen:16 + 8 + nlen + 4 * width], np.uint32)
        qs.update(int(v & 0x7FFFFFFF) for v in vals if v & 0x80000000)
    return qs

# ---------------- roles ----------------
P, W, RS = 0, 1, 2      # preserve value, write, reset to |0>
ROLE = {0x10: (P,), 0x11: (W,), 0x12: (RS,), 0x50: (RS,), 0x20: (P, P), 0x21: (P, W), 0x22: (W, W),
        0x30: (P, P, P), 0x31: (P, P, W), 0x32: (P, W, W)}

def events(code, q):
    """All qubit touches sorted by (qubit, pos): keys, role."""
    b = code & 127; cond = code >= 128
    ks, rs = [], []
    for k in range(3):
        idx = np.flatnonzero(q[:, k] >= 0)
        role = np.full(len(idx), W, np.int8)
        bb = b[idx]
        for c, roles in ROLE.items():
            if len(roles) > k: role[bb == c] = roles[k]
        unknown = ~np.isin(bb, list(ROLE))
        assert not unknown.any(), set(bb[unknown].tolist())
        role[(role == RS) & cond[idx]] = W          # conditional reset: value unknown
        ks.append((q[idx, k].astype(np.int64) << SH) | idx.astype(np.int64)); rs.append(role)
    key = np.concatenate(ks); role = np.concatenate(rs)
    o = np.argsort(key, kind="stable"); return key[o], role[o]

def last_before(sorted_keys, ids, pos):
    """Position of the last key with the same id strictly before pos, else -1."""
    qk = (ids.astype(np.int64) << SH) | pos.astype(np.int64)
    j = np.searchsorted(sorted_keys, qk) - 1
    ok = j >= 0
    jj = np.where(ok, j, 0); kk = sorted_keys[jj]
    ok &= (kk >> SH) == ids
    return np.where(ok, kk & MASK, -1)

def next_after(sorted_keys, ids, pos):
    qk = (ids.astype(np.int64) << SH) | pos.astype(np.int64)
    j = np.searchsorted(sorted_keys, qk, side="right")
    ok = j < len(sorted_keys)
    jj = np.where(ok, j, 0); kk = sorted_keys[jj]
    ok &= (kk >> SH) == ids
    return np.where(ok, kk & MASK, -1)

# ---------------- scan ----------------
def scan(code, q, bit, cnd, ancillas, Q):
    N = len(code); b = code & 127
    key, role = events(code, q)
    eq = key >> SH; ep = key & MASK
    log("events", len(key))
    # status after each event (forward fill of last non-preserving role, per qubit)
    segstart = np.ones(len(key), bool); segstart[1:] = eq[1:] != eq[:-1]
    anc = np.zeros(Q, bool); anc[list(ancillas)] = True
    idx = np.arange(len(key))
    marker = np.where((role != P) | segstart, idx, -1)
    ff = np.maximum.accumulate(marker)
    # a preserving event at a segment start inherits the initial status (zero for ancillas)
    st = np.where(role[ff] == RS, 1, 0).astype(np.int8)
    st = np.where((role[ff] == P) & anc[eq[ff]], 1, st)          # segment-start preserve on an ancilla: still zero
    zero_before = np.empty(len(key), np.int8)                     # status just before each event
    zero_before[1:] = st[:-1]; zero_before[segstart] = anc[eq[segstart]].astype(np.int8)
    # free zero gaps: (after event k, before event k+1) with status zero; plus leading / trailing / untouched
    gq, gs, ge = [], [], []
    same = ~segstart[1:]
    k = np.flatnonzero(same & (st[:-1] == 1))
    gq.append(eq[k]); gs.append(ep[k]); ge.append(ep[k + 1])
    first = np.flatnonzero(segstart); a0 = anc[eq[first]]
    gq.append(eq[first][a0]); gs.append(np.full(a0.sum(), -1)); ge.append(ep[first][a0])
    last = np.r_[first[1:] - 1, len(key) - 1]; z = st[last] == 1
    gq.append(eq[last][z]); gs.append(ep[last][z]); ge.append(np.full(z.sum(), N))
    touched = set(np.unique(eq).tolist())
    for w in sorted(ancillas - touched):
        gq.append(np.array([w])); gs.append(np.array([-1])); ge.append(np.array([N]))
    gq, gs, ge = (np.concatenate(x).astype(np.int64) for x in (gq, gs, ge))
    keep = ge - gs > 3; gq, gs, ge = gq[keep], gs[keep], ge[keep]
    log("zero gaps", len(gq), "on", len(np.unique(gq)), "qubits")
    wkey = key[role != P]                                          # writes and resets, sorted
    # ---------- (b) chains ----------
    tof = np.flatnonzero(np.isin(code, [0x30, 0x31]))
    rows = []
    for (s0, s1) in ((0, 1), (0, 2), (1, 2)):
        sel = tof if (s0, s1) == (0, 1) else tof[code[tof] == 0x30]
        a = np.minimum(q[sel, s0], q[sel, s1]); c = np.maximum(q[sel, s0], q[sel, s1])
        rows.append(np.stack([a, c, last_before(wkey, a, sel), last_before(wkey, c, sel), sel], 1))
    R = np.concatenate(rows).astype(np.int64)
    o = np.lexsort((R[:, 4], R[:, 3], R[:, 2], R[:, 1], R[:, 0])); R = R[o]
    brk = np.r_[True, (np.diff(R[:, :4], axis=0) != 0).any(1)]
    gid = np.cumsum(brk) - 1; sizes = np.bincount(gid)
    multi = np.flatnonzero(sizes >= 2)
    starts = np.flatnonzero(brk)
    groups = [(int(sizes[g]), R[starts[g]:starts[g] + sizes[g]]) for g in multi]
    groups.sort(key=lambda x: -x[0])
    log("(b) candidate groups", len(groups), "members", int(sum(s for s, _ in groups)))
    used = set(); chains = []; busy = {}   # gap index -> list of (s, e)
    def find_gap(s, e):
        cand = np.flatnonzero((gs < s) & (ge > e))
        for gi in cand.tolist():
            if all(e2 < s or s2 > e for s2, e2 in busy.get(gi, [])): return gi
        return None
    for _, rows_ in groups:
        mem = [r for r in rows_.tolist() if r[4] not in used]
        k0 = 0
        while k0 < len(mem) - 1:
            got = None
            for l in range(len(mem) - 1, k0, -1):
                gi = find_gap(mem[k0][4], mem[l][4])
                if gi is not None: got = (l, gi); break
            if got is None: k0 += 1; continue
            l, gi = got
            busy.setdefault(gi, []).append((mem[k0][4], mem[l][4]))
            ch = dict(a=int(mem[k0][0]), b=int(mem[k0][1]), w=int(gq[gi]), ops=[int(m[4]) for m in mem[k0:l + 1]])
            chains.append(ch); used.update(ch["ops"]); k0 = l + 1
    log("(b) chains", len(chains), "Toffolis saved", sum(len(c["ops"]) - 1 for c in chains))
    spare_win = {}                     # w -> list of (s, e)
    for c in chains: spare_win.setdefault(c["w"], []).append((c["ops"][0], c["ops"][-1]))
    # ---------- (a) conditional-product sites ----------
    bkey_l = []
    for arr, idxs in ((bit, np.flatnonzero(bit >= 0)), (cnd, np.flatnonzero(cnd >= 0))):
        bkey_l.append((arr[idxs].astype(np.int64) << SH) | idxs.astype(np.int64))
    bkey = np.unique(np.concatenate(bkey_l))
    ccx = np.flatnonzero(code == 0x31)
    p = q[ccx, 2].astype(np.int64)
    kpos = np.searchsorted(key, (p << SH) | ccx)                 # event index of the CCX on p
    ok = (key[kpos] == ((p << SH) | ccx)) & (zero_before[kpos] == 1)
    k1, k2 = kpos + 1, kpos + 2
    ok &= (k2 < len(key))
    k1c, k2c = np.minimum(k1, len(key) - 1), np.minimum(k2, len(key) - 1)
    ok &= (eq[k1c] == p) & (eq[k2c] == p)
    upos, jpos = ep[k1c], ep[k2c]
    ok &= (code[upos] == 0xA0) & (code[jpos] == 0x50)
    sites = []
    for n in np.flatnonzero(ok).tolist():
        i, u, j = int(ccx[n]), int(upos[n]), int(jpos[n]); pp = int(p[n])
        aa, bb = int(q[i, 0]), int(q[i, 1])
        cz = [int(q[u, 0]), int(q[u, 1])]; cc = cz[1] if cz[0] == pp else cz[0]
        if cc in (aa, bb): continue
        bp = int(bit[j])
        f = int(next_after(bkey, np.array([bp]), np.array([j]))[0])
        if f < 0 or code[f] != 0xA0 or cnd[f] != bp or {int(q[f, 0]), int(q[f, 1])} != {aa, bb}: continue
        nxt = int(next_after(bkey, np.array([bp]), np.array([f]))[0])
        if nxt >= 0 and not (code[nxt] == 0x50 and bit[nxt] == bp and cnd[nxt] != bp): continue
        lw = last_before(wkey, np.array([aa, bb]), np.array([f, f]))
        if (lw >= i).any() or (lw == i).any(): continue
        if any(s < u < e for w in (aa, bb) for s, e in spare_win.get(w, [])): continue
        if {i, u, f} & used: continue
        sites.append(dict(i=i, u=u, j=j, f=f, a=aa, b=bb, c=cc, p=pp, m=int(cnd[u])))
    log("(a) sites", len(sites), "mean Toffolis saved ~", 0.5 * len(sites))
    return chains, sites

# ---------------- symbolic scan (v2) ----------------
def scan2(code, q, bit, cnd, Q, nbits, anc, inbits, mingap=8):
    """Exact-value tracker. Every qubit / bit value is a GF(2) XOR of opaque atoms, stored as a 128-bit linear hash
    (random atom hashes). A product of two values is an opaque atom keyed by the unordered pair of factor hashes
    (x*0=0, x*1=x, x*x=x). Hash 0 <=> provably 0 (up to a 2^-128 collision); equal hashes <=> equal values.
    Sound: classical bit ops and measurement outcomes get fresh atoms; distributivity is never used."""
    import random
    rnd = random.Random(20261119); gb = rnd.getrandbits
    ONE = gb(128); prods = {}
    def prod(A, B):
        if A == 0 or B == 0: return 0
        if A == ONE: return B
        if B == ONE or A == B: return A
        k = (A, B) if A < B else (B, A)
        r = prods.get(k)
        if r is None: r = prods[k] = gb(128)
        return r
    v = [0 if i in anc else gb(128) for i in range(Q)]
    bv = [0] * nbits
    for i in inbits: bv[i] = gb(128)
    last = [-1] * Q; gq, gs, ge = [], [], []
    tofs = []                      # (pos, kind, qubits, pair hashes)
    cand = {}                      # (a) state per p
    bwatch = {}                    # bit -> p for (a) stages 2/3
    sites = []
    depth = 0; nblock = 0
    WRITES = {0x21: (1,), 0x31: (2,), 0x11: (0,), 0x12: (0,), 0x50: (0,), 0x22: (0, 1), 0x32: (1, 2)}
    N = len(code); CH = 1 << 20
    for s0 in range(0, N, CH):
        cs = code[s0:s0 + CH].tolist(); qs = q[s0:s0 + CH].tolist(); bs = bit[s0:s0 + CH].tolist(); ns = cnd[s0:s0 + CH].tolist()
        for off in range(len(cs)):
            pos = s0 + off; c = cs[off]; qq = qs[off]; base = c & 127; cb = ns[off]; bb_ = bs[off]
            if bwatch and depth:                     # any bit touch inside a PUSH_CONDITION block cancels the watch
                for bt in [x for x in (cb, bb_) if x >= 0]:
                    pw = bwatch.pop(bt, None)
                    if pw is not None: cand.pop(pw, None)
            if bwatch:
                tb = [x for x in (cb, bb_) if x >= 0]
                if len(tb) == 2 and tb[0] == tb[1]: tb = tb[:1]
                for bt in tb:
                    pw = bwatch.get(bt)
                    if pw is None: continue
                    st = cand[pw]
                    if st["stage"] == 2:
                        a_, b_ = st["a"], st["b"]
                        if c == 0xA0 and cb == bt and {qq[0], qq[1]} == {a_, b_} and prod(v[a_], v[b_]) == st["h"]:
                            st["stage"] = 3; st["f"] = pos
                        else:
                            del bwatch[bt]; del cand[pw]
                    else:   # stage 3: the next touch of b' must overwrite it (unconditional HMR into b')
                        if c == 0x50 and bb_ == bt and cb != bt:
                            sites.append(st)
                        del bwatch[bt]; del cand[pw]
            for k in range(3):
                x = qq[k]
                if x < 0: break
                if v[x] == 0 and pos - last[x] > mingap:
                    gq.append(x); gs.append(last[x]); ge.append(pos)
                last[x] = pos
                st = cand.get(x)
                if st is not None and st["stage"] < 2 and pos != st["i"]:
                    ok = False
                    if depth: del cand[x]; continue
                    if st["stage"] == 0 and c == 0xA0:
                        cc = qq[1] if qq[0] == x else qq[0]
                        if cc not in (st["a"], st["b"]) and prod(v[st["a"]], v[st["b"]]) == st["h"]:
                            st.update(stage=1, u=pos, c=cc, m=cb); ok = True
                    elif st["stage"] == 1 and c == 0x50 and bb_ not in bwatch:
                        st.update(stage=2, j=pos, bp=bb_); bwatch[bb_] = x; ok = True
                    if not ok:
                        del cand[x]
            if base == 0x01:                         # 0x81 PUSH_CONDITION if b / 0x01 POP_CONDITION
                depth += 1 if c == 0x81 else -1
                assert depth >= 0, pos
                continue
            cv = bv[cb] if cb >= 0 else None
            if depth:
                # inside a conditional block: apply the effect, then any written qubit whose value would change
                # becomes a fresh atom (sound whatever the block condition is); nothing here is a site member.
                wr = WRITES.get(base, ())
                old = [v[qq[k]] for k in wr]
                if base == 0x21:
                    d = v[qq[0]]
                    if cv is not None: d = prod(cv, d)
                    v[qq[1]] ^= d
                elif base == 0x31:
                    d = prod(v[qq[0]], v[qq[1]])
                    if cv is not None: d = prod(cv, d)
                    v[qq[2]] ^= d
                elif base == 0x11:
                    v[qq[0]] ^= (ONE if cv is None else cv)
                elif base == 0x12 or base == 0x50:
                    v[qq[0]] = 0 if cv is None else v[qq[0]] ^ prod(cv, v[qq[0]])
                elif base == 0x22:
                    x_, y_ = qq[0], qq[1]
                    if cv is None: v[x_], v[y_] = v[y_], v[x_]
                    else:
                        d = prod(cv, v[x_] ^ v[y_]); v[x_] ^= d; v[y_] ^= d
                elif base == 0x32:
                    d = prod(v[qq[0]], v[qq[1]] ^ v[qq[2]])
                    if cv is not None: d = prod(cv, d)
                    v[qq[1]] ^= d; v[qq[2]] ^= d
                elif base not in (0x10, 0x20, 0x30, 0x00, 0x40, 0x41, 0x42):
                    raise ValueError(hex(c))
                for k, o in zip(wr, old):
                    if v[qq[k]] != o: v[qq[k]] = gb(128)
                if bb_ >= 0: bv[bb_] = gb(128)
                nblock += 1
                continue
            if base == 0x21:
                d = v[qq[0]]
                if cv is not None: d = prod(cv, d)
                v[qq[1]] ^= d
            elif base == 0x31:
                a_, b_, t_ = qq
                hp = prod(v[a_], v[b_])
                if c == 0x31:
                    tofs.append((pos, 0x31, (a_, b_, t_), (hp,)))
                    if v[t_] == 0 and t_ not in cand and t_ not in bwatch.values():
                        cand[t_] = dict(stage=0, i=pos, a=a_, b=b_, p=t_, h=hp)
                d = hp if cv is None else prod(cv, hp)
                v[t_] ^= d
            elif base == 0x30:
                if c == 0x30:
                    a_, b_, t_ = qq
                    tofs.append((pos, 0x30, (a_, b_, t_), (prod(v[a_], v[b_]), prod(v[a_], v[t_]), prod(v[b_], v[t_]))))
            elif base == 0x11:
                v[qq[0]] ^= (ONE if cv is None else cv)
            elif base == 0x12 or base == 0x50:
                if cv is None: v[qq[0]] = 0
                else: v[qq[0]] ^= prod(cv, v[qq[0]])
                if base == 0x50: bv[bb_] = gb(128)
            elif base == 0x22:
                x_, y_ = qq[0], qq[1]
                if cv is None: v[x_], v[y_] = v[y_], v[x_]
                else:
                    d = prod(cv, v[x_] ^ v[y_]); v[x_] ^= d; v[y_] ^= d
            elif base == 0x32:
                c_, x_, y_ = qq
                d = prod(v[c_], v[x_] ^ v[y_])
                if cv is not None: d = prod(cv, d)
                v[x_] ^= d; v[y_] ^= d
            elif base in (0x40, 0x41, 0x42):
                bv[bb_] = gb(128)
            elif base in (0x10, 0x20, 0x00, 0x01):
                pass
            else:
                raise ValueError(hex(c))
        log("  tracked", min(s0 + CH, N), "ops; products", len(prods), "open (a)", len(cand), "sites", len(sites),
            "ops in blocks", nblock, "depth", depth)
    sites += [st for st in cand.values() if st["stage"] == 3]
    for x in range(Q):
        if v[x] == 0: gq.append(x); gs.append(last[x]); ge.append(N)
    gq, gs, ge = np.array(gq, np.int64), np.array(gs, np.int64), np.array(ge, np.int64)
    log("zero gaps", len(gq), "on", len(np.unique(gq)), "qubits; (a) raw sites", len(sites))
    GAPS[:] = [gq, gs, ge, N, Q]
    rows = []
    for pos, kind, qq, hs in tofs:
        pairs = [(qq[0], qq[1])] if kind == 0x31 else [(qq[0], qq[1]), (qq[0], qq[2]), (qq[1], qq[2])]
        for (a_, b_), hh in zip(pairs, hs):
            if hh != 0: rows.append((hh, pos, a_, b_))
    rows.sort()
    groups = []; k0 = 0
    while k0 < len(rows):
        k1 = k0
        while k1 + 1 < len(rows) and rows[k1 + 1][0] == rows[k0][0]: k1 += 1
        if k1 > k0: groups.append(rows[k0:k1 + 1])
        k0 = k1 + 1
    groups.sort(key=lambda g: -len(g))
    log("(b) product groups", len(groups), "potential Toffolis", sum(len(g) - 1 for g in groups))
    used = set(); chains = []; busy = {}
    def find_gap(s, e):
        for gi in np.flatnonzero((gs < s) & (ge > e)).tolist():
            if all(e2 < s or s2 > e for s2, e2 in busy.get(gi, [])): return gi
        return None
    diag = []
    for g in groups:
        seen = set(); mem = []
        for r in g:
            if r[1] in used or r[1] in seen: continue
            seen.add(r[1]); mem.append(r)
        k0 = 0
        while k0 < len(mem) - 1:
            got = None
            for l in range(len(mem) - 1, k0, -1):
                gi = find_gap(mem[k0][1], mem[l][1])
                if gi is not None: got = (l, gi); break
            if got is None:
                if len(diag) < 60: diag.append(dict(ops=[mem[k0][1], mem[k0 + 1][1]], pair=[mem[k0][2], mem[k0][3]]))
                k0 += 1; continue
            l, gi = got
            busy.setdefault(gi, []).append((mem[k0][1], mem[l][1]))
            chains.append(dict(w=int(gq[gi]), ops=[r[1] for r in mem[k0:l + 1]], pairs=[[r[2], r[3]] for r in mem[k0:l + 1]]))
            used.update(chains[-1]["ops"]); k0 = l + 1
    log("(b) chains", len(chains), "Toffolis saved", sum(len(c["ops"]) - 1 for c in chains))
    spare_win = {}
    for c in chains: spare_win.setdefault(c["w"], []).append((c["ops"][0], c["ops"][-1]))
    out = []
    for st in sites:
        if st["i"] in used: continue
        if any(s < st["u"] < e for w in (st["a"], st["b"]) for s, e in spare_win.get(w, [])): continue
        out.append({k: st[k] for k in ("i", "u", "j", "f", "a", "b", "c", "p", "m")})
    log("(a) sites", len(out), "mean Toffolis saved ~", 0.5 * len(out))
    edges = []
    for gi_, g in enumerate(groups):
        ps = sorted({r[1] for r in g})
        edges += [(ps[k], ps[k + 1], gi_) for k in range(len(ps) - 1)]
    return chains, out, dict(groups=len(groups), potential=sum(len(g) - 1 for g in groups), unplaced_examples=diag, edges=edges,
                             gaps=int(len(gq)), gap_qubits=int(len(np.unique(gq))))

# ---------------- apply ----------------
def build_edits(chains, sites, code, q, bit, cnd, nbits, unitary=False):
    E = {}
    def put(pos, ops):
        assert pos not in E, pos
        E[pos] = ops
    for c in chains:
        w, ops = c["w"], c["ops"]
        pairs = c.get("pairs") or [[c["a"], c["b"]]] * len(ops)
        a, b = pairs[-1]                      # the final measurement fix-up uses the last member's control pair
        for k, pos in enumerate(ops):
            kind = int(code[pos]); qq = [int(x) for x in q[pos]]
            ka, kb = pairs[k]
            if k == 0: a0, b0 = ka, kb
            if kind == 0x31:
                assert {qq[0], qq[1]} == {ka, kb}; core = [(0x21, (w, qq[2]), -1, -1)]
            else:
                third = [x for x in qq if x not in (ka, kb)]; assert len(third) == 1
                core = [(0x20, (w, third[0]), -1, -1)]
            new = []
            if k == 0: new.append((0x31, (a0, b0, w), -1, -1))
            new += core
            if k == len(ops) - 1:
                if unitary:          # test only: unitary uncompute, keeps the simulator's RNG stream aligned
                    new += [(0x31, (a, b, w), -1, -1)]
                else:
                    cb = nbits; nbits += 1
                    new += [(0x50, (w,), cb, -1), (0xA0, (a, b), -1, cb)]
            put(pos, new)
    for c in []:
        for k, pos in enumerate([]):
            if kind == 0x31:
                assert {qq[0], qq[1]} == {a, b}; core = [(0x21, (w, qq[2]), -1, -1)]
            else:
                third = [x for x in qq if x not in (a, b)]; assert len(third) == 1
                core = [(0x20, (w, third[0]), -1, -1)]
            new = []
            if k == 0: new.append((0x31, (a, b, w), -1, -1))
            new += core
            if k == len(ops) - 1:
                cb = nbits; nbits += 1
                new += [(0x50, (w,), cb, -1), (0xA0, (a, b), -1, cb)]
            put(pos, new)
    for s in sites:
        put(s["i"], []); put(s["f"], [])
        put(s["u"], [(0xB0, (s["a"], s["b"], s["c"]), -1, s["m"])])
    return E, nbits

def apply_edits(E, code, q, bit, cnd):
    pos = sorted(E); pieces = {k: [] for k in "cqbn"}; prev = 0
    for p_ in pos:
        pieces["c"].append(code[prev:p_]); pieces["q"].append(q[prev:p_]); pieces["b"].append(bit[prev:p_]); pieces["n"].append(cnd[prev:p_])
        ops = E[p_]
        if ops:
            pieces["c"].append(np.array([o[0] for o in ops], np.uint8))
            qq = np.full((len(ops), 3), -1, np.int32)
            for r, o in enumerate(ops): qq[r, :len(o[1])] = o[1]
            pieces["q"].append(qq)
            pieces["b"].append(np.array([o[2] for o in ops], np.int32)); pieces["n"].append(np.array([o[3] for o in ops], np.int32))
        prev = p_ + 1
    pieces["c"].append(code[prev:]); pieces["q"].append(q[prev:]); pieces["b"].append(bit[prev:]); pieces["n"].append(cnd[prev:])
    return (np.concatenate(pieces["c"]), np.concatenate(pieces["q"]), np.concatenate(pieces["b"]), np.concatenate(pieces["n"]))

# ---------------- main ----------------
def main():
    if "--local" in sys.argv:
        path = sys.argv[sys.argv.index("--local") + 1]
        anc_from = int(sys.argv[sys.argv.index("--ancilla-from") + 1]) if "--ancilla-from" in sys.argv else 840
        data = open(path, "rb").read(); h, regs, sec = parse(data)
        code, q, bit, cnd = decode(sec)
        re = encode(h, regs, sec, code, q, bit, cnd); log("codec round trip", re == data)
        chains, sites, info = scan2(code, q, bit, cnd, h[1], h[2], set(range(anc_from, h[1])), range(h[2]))
        print("info", {k: v for k, v in info.items() if k != "unplaced_examples"})
        for c in chains: print("chain", c)
        for s in sites: print("site", s)
        E, nb = build_edits(chains, [] if "--unitary" in sys.argv else sites, code, q, bit, cnd, h[2], unitary="--unitary" in sys.argv)
        c2, q2, b2, n2 = apply_edits(E, code, q, bit, cnd); hh = list(h); hh[2] = nb
        out = encode(hh, regs, sec, c2, q2, b2, n2); open(path + ".patched.kmb", "wb").write(out)
        log("patched ops", len(c2), "bits", nb)
        return
    INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
    comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dqi").is_dir())
    sys.path[:0] = [str(comp / "kickmix")]
    import kickmix as km
    data = None
    for z in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True):
        try:
            with zipfile.ZipFile(z) as f:
                if "dqi.kmb" in f.namelist():
                    b_ = f.read("dqi.kmb")
                    if hashlib.sha256(b_).hexdigest().startswith(WANT): data = b_; log("input", z); break
        except zipfile.BadZipFile:
            pass
    if data is None:
        for p_ in INPUT.rglob("*dqi*.kmb"):
            b_ = p_.read_bytes()
            if hashlib.sha256(b_).hexdigest().startswith(WANT): data = b_; log("input", p_); break
    assert data is not None, "4471b493 not found"
    h, regs, sec = parse(data)
    code, q, bit, cnd = decode(sec)
    assert encode(h, regs, sec, code, q, bit, cnd) == data, "codec round trip failed"
    log("codec round trip OK; ops", len(code))
    anc = set(range(h[1])) - reg_qubits(regs)
    log("ancillas", len(anc), min(anc), max(anc))
    inb = reg_bits(regs); log("input bits", len(inb))
    chains, sites, info = scan2(code, q, bit, cnd, h[1], h[2], anc, inb)
    edges = info.pop("edges")
    # what dedicated extra spare qubits (always |0>, untouched) would allow: max non-overlapping edges on k spares
    # (edges of one group may share an endpoint on the same spare: that is one chain). Optimal greedy by end time.
    used_op = set(); E2 = []
    for a_, b_, g_ in sorted(edges, key=lambda e: e[1]):
        E2.append((a_, b_, g_))
    extra = {}
    for kq in (1, 2, 3, 4, 6, 8):
        ends = [(-1, -1)] * kq; got = 0; taken = set()
        for a_, b_, g_ in E2:
            best = -1
            for m in range(kq):
                e_, g0 = ends[m]
                if (e_ < a_ or (e_ == a_ and g0 == g_)) and (best < 0 or e_ > ends[best][0]): best = m
            if best >= 0 and (a_, b_) not in taken:
                ends[best] = (b_, g_); got += 1; taken.add((a_, b_))
        extra[kq] = got
    lens = sorted(b_ - a_ for a_, b_, _ in edges)
    info["extra_spare_toffolis"] = extra
    info["edge_len_quantiles"] = [lens[int(f * (len(lens) - 1))] for f in (0, .1, .25, .5, .75, .9, 1)] if lens else []
    (OUT / "dq_edges.json").write_text(json.dumps(edges))
    log("extra-spare potential (Toffolis saved with k always-free spares):", extra, "edge length quantiles", info["edge_len_quantiles"])
    saved_b = sum(len(c["ops"]) - 1 for c in chains); saved_a = 0.5 * len(sites)
    res = dict(input=WANT, chains=len(chains), chain_toffolis_saved=saved_b, a_sites=len(sites), a_mean_saved_est=saved_a,
               spares_used=sorted({c["w"] for c in chains}), info=info)
    (OUT / "dq_sites.json").write_text(json.dumps(dict(chains=chains, sites=sites), indent=0))
    def save(): (OUT / "dq_scan_results.json").write_text(json.dumps(res, indent=1, default=str))
    save(); log("summary", res)
    if saved_b + saved_a < APPLY_MIN:
        log(f"below APPLY_MIN={APPLY_MIN}: not building"); return
    import re
    def build(chs, sts, path):
        E, nb = build_edits(chs, sts, code, q, bit, cnd, h[2])
        c2, q2, b2, n2 = apply_edits(E, code, q, bit, cnd)
        hh = list(h); hh[2] = nb
        out = encode(hh, regs, sec, c2, q2, b2, n2); del c2, q2, b2, n2
        Path(path).write_bytes(out); return hashlib.sha256(out).hexdigest()
    def verify(path, salt):
        v = subprocess.run([sys.executable, "dqi/verify_submission_dqi.py", str(path), "--max-workers", "4"] + (["--salt", salt] if salt else []),
                           cwd=comp / "verifiers", env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")), capture_output=True, text=True)
        try:
            r = json.loads(v.stdout[v.stdout.index("{"):]); r.pop("shot_operation_counts", None)
        except ValueError:
            r = {"status": "UNPARSED", "tail": v.stdout[-1500:], "err": v.stderr[-1500:]}
        out = {"salt": salt, **{k: r.get(k) for k in ("status", "score", "toffoli_count", "value_errors", "phase_errors",
                                                        "repetition_factor", "error", "tail", "err")}}
        if out["value_errors"] is None and out["error"]:
            m = re.search(r"value_errors=(\d+)/\d+, phase_errors=(\d+)/", out["error"])
            if m: out["value_errors"], out["phase_errors"] = int(m.group(1)), int(m.group(2))
        return out
    def nerr(r):
        if r["value_errors"] is None: return 10 ** 9
        return r["value_errors"] + r["phase_errors"]
    tmp = Path("/tmp/cand.kmb")
    res["trials"] = []
    sha = build(chains, sites, tmp); r = verify(tmp, ""); res["trials"].append(dict(chains=len(chains), sites=len(sites), sha=sha, **r))
    log("full build", sha[:16], r); save()
    keep_c, keep_s = chains, sites
    if nerr(r) > 3:
        # bisect: keep only chain subsets that verify (<= 1 failed shot at the default salt)
        keep_c, keep_s = [], []
        queue = [chains]
        while queue and time.time() - T0 < 6 * 3600:
            sub = queue.pop(0)
            sha = build(sub, [], tmp); r = verify(tmp, "")
            res["trials"].append(dict(chains=len(sub), first_op=sub[0]["ops"][0], last_op=sub[-1]["ops"][-1], sha=sha, **{k: r[k] for k in ("status", "value_errors", "phase_errors", "score")}))
            log("bisect", len(sub), "chains ->", r["status"], r["value_errors"], r["phase_errors"]); save()
            if nerr(r) <= 1: keep_c = keep_c + sub
            elif len(sub) > 1: queue += [sub[:len(sub) // 2], sub[len(sub) // 2:]]
            else: res.setdefault("bad_chains", []).append(sub[0])
        log("bisect kept", len(keep_c), "chains,", sum(len(c["ops"]) - 1 for c in keep_c), "Toffolis")
        if not keep_c: log("nothing sound to build"); save(); return
    kp = OUT / "dqi.kmb"
    res["final_chains"] = len(keep_c); res["final_sites"] = len(keep_s)
    res["final_toffolis_saved"] = sum(len(c["ops"]) - 1 for c in keep_c)
    res["sha256"] = build(keep_c, keep_s, kp)
    (OUT / "dq_sites_final.json").write_text(json.dumps(dict(chains=keep_c, sites=keep_s), indent=0))
    c = km.Circuit.from_file(str(kp), format="kmb"); st = c.operation_statistics()
    res["num_qubits"] = c.num_qubits
    res["stats"] = {k: getattr(st, k) for k in ("num_instructions", "num_toffoli_like", "num_t_like", "num_non_clifford")}
    del c; save(); log("new dqi", res["sha256"], res["stats"])
    res["verify"] = []
    for s_ in SALTS:
        res["verify"].append(verify(kp, s_)); log("verify", res["verify"][-1]); save()



def width_main():
    INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
    data = None
    for z in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True):
        try:
            with zipfile.ZipFile(z) as f:
                if "dqi.kmb" in f.namelist():
                    b_ = f.read("dqi.kmb")
                    if hashlib.sha256(b_).hexdigest().startswith(WANT): data = b_; log("input", z); break
        except zipfile.BadZipFile:
            pass
    assert data is not None
    h, regs, sec = parse(data)
    code, q, bit, cnd = decode(sec)
    assert encode(h, regs, sec, code, q, bit, cnd) == data, "codec round trip failed"
    anc = set(range(h[1])) - reg_qubits(regs); inb = reg_bits(regs)
    log("ops", len(code), "qubits", h[1], "ancillas", len(anc), "input bits", len(inb))
    scan2(code, q, bit, cnd, h[1], h[2], anc, inb)
    gq, gs, ge, N, Q = GAPS
    # free wires at op t: zero gaps with gs < t < ge (the wire is provably |0> and untouched at t)
    d = np.zeros(N + 2, np.int64)
    lo = np.clip(gs + 1, 0, N + 1); hi = np.clip(ge, 0, N + 1)
    ok = hi > lo
    np.add.at(d, lo[ok], 1); np.add.at(d, hi[ok], -1)
    free = np.cumsum(d)[:N]
    res = dict(N=int(N), Q=int(Q), gaps=int(len(gq)), gap_qubits=int(len(np.unique(gq))), min_free=int(free.min()),
               n_ops_zero_free=int((free == 0).sum()), free_quantiles=[int(np.quantile(free, f)) for f in (0, .01, .1, .5, .9, 1)])
    z = np.flatnonzero(free == 0)
    if len(z):
        brk = np.flatnonzero(np.diff(z) > 1)
        starts = np.r_[z[0], z[brk + 1]]; ends = np.r_[z[brk], z[-1]]
        res["zero_free_windows"] = int(len(starts))
        res["zero_free_window_examples"] = [[int(a), int(b)] for a, b in list(zip(starts, ends))[:50]]
    log(res)
    (OUT / "d2_width.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    width_main()
