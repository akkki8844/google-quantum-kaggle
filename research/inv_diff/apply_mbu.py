"""Replace top-level 'CCX a b t' (t ends as XOR of `terms`) by measurement-based erase + CNOT rebuild:
    HMR t bX ; PUSH_CONDITION if bX ; CZ a b ; Z q (q in terms) ; POP_CONDITION ; CX q t (q in terms)
Usage: python -I apply_mbu.py IN.kmx OUT.kmb LINE:TERMS [LINE:TERMS ...]   (TERMS = comma list of qubit ids)
LINE is the body index printed by mbu_scan.py."""
import re, sys
import kickmix as km

inp, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
c = km.Circuit.from_file(inp)
lines = [l for l in str(c).splitlines() if l.strip() and not l.lstrip().startswith("#")]
header = [l for l in lines if l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
body = [l for l in lines if not l.startswith(("REGISTER", "APPEND_TO_REGISTER"))]
bits = sorted({int(b) for b in re.findall(r"\bb(\d+)\b", "\n".join(body))})
scratch_bit = f"b{bits[-1] + 1}" if bits else "b0"
repl = {}
for sp in specs:
    ln, terms = sp.split(":")
    ln = int(ln)
    m = re.fullmatch(r"CCX q(\d+) q(\d+) q(\d+)", body[ln].strip())
    assert m, body[ln]
    a, b, t = m.groups()
    ts = [x for x in terms.split(",") if x]
    new = [f"HMR q{t} {scratch_bit}", f"PUSH_CONDITION if {scratch_bit}", f"CZ q{a} q{b}"]
    new += [f"Z q{q}" for q in ts]
    new += ["POP_CONDITION"] + [f"CX q{q} q{t}" for q in ts]
    repl[ln] = new
newbody = []
for i, l in enumerate(body):
    newbody.extend(repl.get(i, [l]))
nc = km.Circuit("\n".join(header + newbody))
nc.to_file(out, format="kmb" if out.endswith(".kmb") else "kmx")
print("wrote", out, "static toffoli", nc.operation_statistics().num_toffoli_like, "was", c.operation_statistics().num_toffoli_like,
      "qubits", nc.num_qubits, "scratch bit", scratch_bit)
