"""d4 (built on d3): bit-packed numba simulator for the DQI circuit 4b572141 (our own code; the official verifier module is used
only to draw test inputs). One pass gives:
  * a check of the simulator: S registers restored, every other wire 0, mean Toffoli-like count vs the official 316,131.22;
  * true idle wires: per wire, gaps between touches where the wire is 0 on every sampled shot. free(t) = number of wires
    idle at op t. If min_t free(t) >= 1, the circuit can be relabelled onto Q - 1 wires at no Toffoli cost (+0.023 GM);
  * a scan of every Toffoli-like gate (on the shots where it acts) for exact 0-Toffoli rewrites:
      ZERO   a&b == 0                 (gate does nothing: delete)
      OR1    ~a & ~b == 0  (a|b == 1) (a&b = a^b^1: CNOTs + X / CZs + Z)
      EQ     a == b                   (a&b = a)
      NEQ    a == ~b                  (a&b = 0)
      MBU    CCX with t == a&b before (measurement-based erase, 0 Toffolis)
Shots: SCREEN_SHOTS for the scan; candidates are re-checked with CONFIRM_SHOTS fresh shots.
"""
import glob, hashlib, json, os, struct, sys, time, zipfile
from pathlib import Path
import numpy as np
import numba as nb

T0 = time.time()
def log(*a): print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)
INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
WANT = "4b572141"
SCREEN_SHOTS = int(os.environ.get("SCREEN_SHOTS", 4096))
CONFIRM_SHOTS = int(os.environ.get("CONFIRM_SHOTS", 32768))
MAGIC = bytes.fromhex("d750c7d5c329d326e3cc9f6834f2b8bf")

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
    m3, m2, m1, mb, mc = masks(code)
    N = len(code); q = np.full((N, 3), -1, np.int32)
    u = lambda i: np.frombuffer(sec[i], np.uint32).astype(np.int32)
    q[m3, 0], q[m3, 1], q[m3, 2] = u(2), u(3), u(4)
    q[m2, 0], q[m2, 1] = u(5), u(6)
    q[m1, 0] = u(7)
    bit = np.full(N, -1, np.int32); bit[mb] = u(8)
    cnd = np.full(N, -1, np.int32); cnd[mc] = u(9)
    return code, q, bit, cnd

def registers(regs):
    out = {}
    for r in regs:
        nlen, width = struct.unpack_from("<II", r, 16)
        name = r[24:24 + nlen].decode()
        vals = np.frombuffer(r[24 + nlen:24 + nlen + 4 * width], np.uint32)
        out[name] = [(bool(v & 0x80000000), int(v & 0x7FFFFFFF)) for v in vals]
    return out

@nb.njit(cache=False)
def popc(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

@nb.njit(cache=False)
def run(code, q, bit, cnd, Q, S, B, anc, seed, track, cand_pos):
    """S: (Q, W) qubit state, B: (nbits, W) bits. track: record idle gaps + Toffoli scan.
    cand_pos: if non-empty, only evaluate the scan at these op positions (confirm mode)."""
    N = code.shape[0]; W = S.shape[1]
    FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
    stack = np.zeros((64, W), np.uint64); stack[0, :] = FULL; sp = 0
    m = np.empty(W, np.uint64)
    rs = np.uint64(seed) | np.uint64(1)
    tof = 0.0
    last = np.full(Q, -1, np.int64)
    zero_after = anc.copy()                       # wire is 0 on all shots after its last touch
    diff = np.zeros(N + 2 if track else 1, np.int32)
    flags = np.zeros(N if track else 1, np.uint8)   # bit0 ZERO, bit1 OR1, bit2 EQ, bit3 NEQ, bit4 MBU, bit5 never acts
    ci = 0
    for p in range(N):
        c = code[p]; base = c & 127
        if c == 0x81:                               # PUSH_CONDITION if b
            b0 = cnd[p]
            for w in range(W): stack[sp + 1, w] = stack[sp, w] & B[b0, w]
            sp += 1
            continue
        if c == 0x01:                               # POP_CONDITION
            sp -= 1
            continue
        if c >= 128:
            b0 = cnd[p]
            for w in range(W): m[w] = stack[sp, w] & B[b0, w]
        else:
            for w in range(W): m[w] = stack[sp, w]
        a0 = q[p, 0]; a1 = q[p, 1]; a2 = q[p, 2]
        # ---- scan (before the gate) ----
        if base == 0x30 or base == 0x31 or base == 0x32:
            cnt = np.uint64(0)
            for w in range(W): cnt += popc(m[w])
            tof += float(cnt)
            do = track and (cand_pos.shape[0] == 0 or (ci < cand_pos.shape[0] and cand_pos[ci] == p))
            if cand_pos.shape[0] > 0 and ci < cand_pos.shape[0] and cand_pos[ci] == p:
                ci += 1
            if do:
                if base == 0x32:
                    x0, x1 = a0, a0                 # CSWAP: only the 'never acts' flag
                else:
                    x0, x1 = a0, a1
                z = True; o = True; e = True; ne = True; mb = base == 0x31; act = False
                for w in range(W):
                    mm = m[w]
                    if mm != 0: act = True
                    va = S[x0, w]; vb = S[x1, w]
                    if (va & vb & mm) != 0: z = False
                    if ((~va) & (~vb) & mm) != 0: o = False
                    if ((va ^ vb) & mm) != 0: e = False
                    if ((~(va ^ vb)) & mm) != 0: ne = False
                    if mb and ((S[a2, w] ^ (va & vb)) & mm) != 0: mb = False
                f = 0
                if base != 0x32:
                    if z: f |= 1
                    if o: f |= 2
                    if e: f |= 4
                    if ne: f |= 8
                    if mb: f |= 16
                if not act: f |= 32
                flags[p] = f
        # ---- execute ----
        if base == 0x31:
            for w in range(W): S[a2, w] ^= S[a0, w] & S[a1, w] & m[w]
        elif base == 0x32:
            for w in range(W):
                d = (S[a1, w] ^ S[a2, w]) & S[a0, w] & m[w]; S[a1, w] ^= d; S[a2, w] ^= d
        elif base == 0x21:
            for w in range(W): S[a1, w] ^= S[a0, w] & m[w]
        elif base == 0x22:
            for w in range(W):
                d = (S[a0, w] ^ S[a1, w]) & m[w]; S[a0, w] ^= d; S[a1, w] ^= d
        elif base == 0x11:
            for w in range(W): S[a0, w] ^= m[w]
        elif base == 0x12:
            for w in range(W): S[a0, w] &= ~m[w]
        elif base == 0x50:                          # HMR q -> bit: random outcome, reset
            b1 = bit[p]
            for w in range(W):
                rs ^= rs << np.uint64(13); rs ^= rs >> np.uint64(7); rs ^= rs << np.uint64(17)
                B[b1, w] = (B[b1, w] & ~m[w]) | (rs & m[w])
                S[a0, w] &= ~m[w]
        elif base == 0x40:
            b1 = bit[p]
            for w in range(W): B[b1, w] &= ~m[w]
        elif base == 0x41:
            b1 = bit[p]
            for w in range(W): B[b1, w] |= m[w]
        elif base == 0x42:
            b1 = bit[p]
            for w in range(W): B[b1, w] ^= m[w]
        # diagonal / phase ops (0x10, 0x20, 0x30, 0x00) change no value
        # ---- idle-gap bookkeeping ----
        if track and cand_pos.shape[0] == 0:
            for k in range(3):
                x = q[p, k]
                if x < 0: continue
                if zero_after[x]:
                    lo = last[x] + 1
                    if p > lo:
                        diff[lo] += 1; diff[p] -= 1
                last[x] = p
                z2 = True
                for w in range(W):
                    if S[x, w] != 0:
                        z2 = False; break
                zero_after[x] = z2
    if track and cand_pos.shape[0] == 0:
        for x in range(Q):
            if zero_after[x]:
                lo = last[x] + 1
                if N > lo:
                    diff[lo] += 1; diff[N] -= 1
    return tof, diff, flags


@nb.njit(cache=False)
def gf2_rank_dep(S, W):
    """Rank of the Q wire rows plus the all-ones row over W words, and the number of zero rows. Gaussian elimination."""
    Q = S.shape[0]
    M = np.empty((Q + 1, W), np.uint64)
    M[:Q] = S[:, :W]
    for w in range(W): M[Q, w] = np.uint64(0xFFFFFFFFFFFFFFFF)
    used = np.zeros(Q + 1, np.bool_)
    r = 0
    for w in range(W):
        for bt in range(64):
            bitm = np.uint64(1) << np.uint64(bt)
            piv = -1
            for i in range(Q + 1):
                if not used[i] and (M[i, w] & bitm) != 0:
                    piv = i; break
            if piv < 0: continue
            used[piv] = True; r += 1
            for i in range(Q + 1):
                if i != piv and (M[i, w] & bitm) != 0:
                    for k in range(w, W): M[i, k] ^= M[piv, k]
    return r

@nb.njit(cache=False)
def run_ck(code, q, bit, cnd, Q, S, B, seed, ck, out, Wr):
    """Execute and, before each op in ck (sorted), store the affine rank of the wires (first Wr words)."""
    N = code.shape[0]; W = S.shape[1]
    FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
    stack = np.zeros((64, W), np.uint64); stack[0, :] = FULL; sp = 0
    m = np.empty(W, np.uint64)
    rs = np.uint64(seed) | np.uint64(1)
    ci = 0
    for p in range(N):
        if ci < ck.shape[0] and ck[ci] == p:
            out[ci] = gf2_rank_dep(S, Wr); ci += 1
        c = code[p]; base = c & 127
        if c == 0x81:
            b0 = cnd[p]
            for w in range(W): stack[sp + 1, w] = stack[sp, w] & B[b0, w]
            sp += 1
            continue
        if c == 0x01:
            sp -= 1
            continue
        if c >= 128:
            b0 = cnd[p]
            for w in range(W): m[w] = stack[sp, w] & B[b0, w]
        else:
            for w in range(W): m[w] = stack[sp, w]
        a0 = q[p, 0]; a1 = q[p, 1]; a2 = q[p, 2]
        if base == 0x31:
            for w in range(W): S[a2, w] ^= S[a0, w] & S[a1, w] & m[w]
        elif base == 0x32:
            for w in range(W):
                d = (S[a1, w] ^ S[a2, w]) & S[a0, w] & m[w]; S[a1, w] ^= d; S[a2, w] ^= d
        elif base == 0x21:
            for w in range(W): S[a1, w] ^= S[a0, w] & m[w]
        elif base == 0x22:
            for w in range(W):
                d = (S[a0, w] ^ S[a1, w]) & m[w]; S[a0, w] ^= d; S[a1, w] ^= d
        elif base == 0x11:
            for w in range(W): S[a0, w] ^= m[w]
        elif base == 0x12:
            for w in range(W): S[a0, w] &= ~m[w]
        elif base == 0x50:
            b1 = bit[p]
            for w in range(W):
                rs ^= rs << np.uint64(13); rs ^= rs >> np.uint64(7); rs ^= rs << np.uint64(17)
                B[b1, w] = (B[b1, w] & ~m[w]) | (rs & m[w])
                S[a0, w] &= ~m[w]
        elif base == 0x40:
            b1 = bit[p]
            for w in range(W): B[b1, w] &= ~m[w]
        elif base == 0x41:
            b1 = bit[p]
            for w in range(W): B[b1, w] |= m[w]
        elif base == 0x42:
            b1 = bit[p]
            for w in range(W): B[b1, w] ^= m[w]

def main():
    comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dqi").is_dir())
    sys.path[:0] = [str(comp / "kickmix"), str(comp / "verifiers" / "dqi")]
    import kickmix as km
    import verify_submission_dqi as V
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
    h, regs, sec = parse(data); code, q, bit, cnd = decode(sec); Q, NB = h[1], h[2]
    R = registers(regs)
    Sq = [[v for isq, v in R[f"S{i}"]] for i in range(70)]; Eb = [[v for isq, v in R[f"E{j}"]] for j in range(4095)]
    regq = {v for s_ in Sq for v in s_}
    anc = np.array([x not in regq for x in range(Q)], np.bool_)
    spec = V.ProblemSpecDQI(m=4095, n=70, degree=12, r=2016, num_shots=4096)
    tables = V.FieldTables(km.GF2Field(12))
    def state(shots, seed):
        rng = np.random.default_rng(seed)
        syn, mbuc, _ = V.sample_shots(spec, tables, rng, shots)
        W = shots // 64
        S = np.zeros((Q, W), np.uint64); B = np.zeros((NB, W), np.uint64)
        def pack(bits): return np.packbits(bits.reshape(W, 64)[:, ::-1], axis=1).view(">u8").astype(np.uint64).ravel()
        for i in range(70):
            for j, x in enumerate(Sq[i]): S[x] = pack(((syn[:, i] >> j) & 1) == 1)
        for i in range(4095):
            for j, x in enumerate(Eb[i]): B[x] = pack(((mbuc[:, i] >> j) & 1) == 1)
        return S, B
    S, B = state(4096, 1)
    _, diff, _ = run(code, q, bit, cnd, Q, S, B, anc, 12345, True, np.zeros(0, np.int64))
    free = np.cumsum(diff)[:len(code)]
    z = np.flatnonzero(free == 0)
    log("zero-free ops", len(z))
    NCK = int(os.environ.get("NCK", 3000))
    ck = np.unique(z[np.linspace(0, len(z) - 1, NCK).astype(np.int64)])
    out = np.zeros(len(ck), np.int64)
    S, B = state(4096, 3)
    run_ck(code, q, bit, cnd, Q, S, B, 99, ck, out, 32)      # rank over 2,048 shots
    defi = Q + 1 - out
    log("checkpoints", len(ck), "deficiency (Q+1-rank) quantiles", np.quantile(defi, [0, .01, .1, .5, .9, 1]).tolist(),
        "checkpoints with no dependency", int((defi == 0).sum()))
    (OUT / "d4_rank.json").write_text(json.dumps(dict(ck=ck.tolist(), defi=defi.tolist())))
    log("done")

main()
