"""Give a DQI .kmb a new sha256 (hence an independent verifier draw) at zero cost: append `X a; X a` on one
ancilla wire (not in any register) at the very end. Same unitary, same Toffoli / qubit counts.
Our own codec (same as kaggle_nb/dq_scan). Usage: python -I tools/dqi_reroll.py IN.kmb OUT.kmb [N_PAIRS]"""
import hashlib, struct, sys
import numpy as np
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

def reg_qubits(regs):
    qs = set()
    for r in regs:
        nlen, width = struct.unpack_from("<II", r, 16)
        vals = np.frombuffer(r[16 + 8 + nlen:16 + 8 + nlen + 4 * width], np.uint32)
        qs.update(int(v & 0x7FFFFFFF) for v in vals if v & 0x80000000)
    return qs

def reroll(data, pairs=1):
    h, regs, sec = parse(data)
    anc = sorted(set(range(h[1])) - reg_qubits(regs))
    a = anc[-1]
    code = sec[1] + bytes([0x11]) * (2 * pairs)                 # 0x11 = X (one-qubit write)
    s7 = sec[7] + np.full(2 * pairs, a, np.uint32).tobytes()    # section 7 = one-qubit operands
    h = list(h); h[4] += 2 * pairs; h[7] += 2 * pairs
    out = [MAGIC + struct.pack("<4I9Q", *h)] + regs
    for i in sorted(sec):
        b = code if i == 1 else s7 if i == 7 else sec[i]
        out.append(struct.pack("<QQ", i, len(b)) + b)
    return b"".join(out)

if __name__ == "__main__":
    data = open(sys.argv[1], "rb").read()
    out = reroll(data, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    open(sys.argv[2], "wb").write(out)
    print(hashlib.sha256(data).hexdigest()[:8], "->", hashlib.sha256(out).hexdigest()[:8])
