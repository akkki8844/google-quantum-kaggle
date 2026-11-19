"""Delete body lines (indices from the scans) from a circuit: python -I del_lines.py IN.kmx OUT.kmb L1 [L2 ...]"""
import sys
import kickmix as km
inp, out, dl = sys.argv[1], sys.argv[2], {int(v) for v in sys.argv[3:]}
c = km.Circuit.from_file(inp)
lines = [l for l in str(c).splitlines() if l.strip() and not l.lstrip().startswith("#")]
header = [l for l in lines if l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
body = [l for l in lines if not l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
for i in dl:
    print("delete", i, body[i])
nc = km.Circuit("\n".join(header + [l for i, l in enumerate(body) if i not in dl]))
nc.to_file(out, format="kmb")
print("static toffoli", nc.operation_statistics().num_toffoli_like, "was", c.operation_statistics().num_toffoli_like, "Q", nc.num_qubits)
