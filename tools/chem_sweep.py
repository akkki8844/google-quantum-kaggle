"""Grid sweep of the closed-form dfthcsos cost (T*Q) over (R, B, C, brot, J).

Uses tools/chem_cost_explorer.py. lambda_eff and the 20 mHa accuracy gate are NOT modelled:
a shape is only usable if a factorization with that rank reaches 20 mHa. This shows how
much the closed-form part alone can move, and where the pressure on the factorization is.

Run:  python -I tools/chem_sweep.py
"""
import importlib.util
import itertools
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chem_cost_explorer", HERE / "chem_cost_explorer.py")
ex = importlib.util.module_from_spec(spec)
sys.modules["chem_cost_explorer"] = ex
spec.loader.exec_module(ex)

best = {}
rows = []
for R, B, C in itertools.product([2, 3, 4, 6, 8, 12, 16, 24], [60, 80, 100, 143, 200, 300], [10, 20, 30, 44, 60]):
    for brot in range(6, 17):
        for J in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 25):
            try:
                r = ex.total(R, B, C, brot, J)
            except Exception:
                continue
            if not r["within_cap"]:
                continue
            rows.append(r)
rows.sort(key=lambda r: r["TQ"])
print(f"{len(rows)} feasible combos")
print(f"{'R':>3} {'B':>4} {'C':>3} {'brot':>4} {'J':>3} {'T':>7} {'Q':>5} {'TQ':>10} {'angT':>6} {'outer':>5} {'inner':>6} {'giv':>5} {'pers':>4} {'temp':>4}")
for r in rows[:15]:
    print(f"{r['R']:>3} {r['B']:>4} {r['C']:>3} {r['brot']:>4} {r['J']:>3} {r['T']:>7} {r['Q']:>5} {r['TQ']:>10.3e} "
          f"{r['angles_T']:>6} {r['outer']:>5} {r['inner']:>6} {r['givens']:>5} {r['persistent']:>4} {r['temp']:>4}")
base = ex.total(65, 60, 60, 12, 8)
print(f"\nbaseline TQ = {base['TQ']:.3e}; best swept TQ = {rows[0]['TQ']:.3e}  -> {base['TQ'] / rows[0]['TQ']:.2f}x on T*Q alone")
