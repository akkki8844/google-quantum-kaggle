"""Explore the closed-form (heuristic) part of the dfthcsos cost for different (R, B, C, brot, J).

Loads the competition's own cost model (costs_prepare.py) by file path, so nothing from the
data folder is imported as a package. The ANGLES part is approximated by the ordinary-QROM
bound the README states: per circuit at most (A - 2) Toffolis with A = N + R*B addresses,
doubled for the forward and reverse pass, times K = ceil(75 / J) circuits. Real circuits can
be cheaper (the reference prunes subtrees whose rows are all equal), so treat it as an upper
bound on ANGLES. lambda_eff is NOT modelled here: it needs a real factorization.

Run:  python -I tools/chem_cost_explorer.py
"""
import importlib.util
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
COSTS = HERE.parent / "kaggle" / "data" / "verifiers" / "dfthcsos" / "scoreinfra" / "costs_prepare.py"
spec = importlib.util.spec_from_file_location("costs_prepare", COSTS)
cp = importlib.util.module_from_spec(spec)
sys.modules["costs_prepare"] = cp  # dataclasses looks the module up by name
spec.loader.exec_module(cp)

N = 76
CHI = 10
CAP = 128


def heuristic(R, B, C, brot):
    m = cp.SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=CHI, beta=brot)
    oq, iq = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
    return dict(
        outer=m.outer_state_prep(), inner=m.inner_state_prep(), givens=m.givens_rotations(),
        persistent=oq["persistent"] + iq["persistent"],
        temp=max(oq["temporary"], iq["temporary"]),
    )


def angle_circuit_qubits(R, B, brot, J):
    a = N + R * B
    sel = math.ceil(math.log2(a))
    return sel + J * brot + max(sel - 1, 0)


def total(R, B, C, brot, J, lam=None):
    h = heuristic(R, B, C, brot)
    a = N + R * B
    k = math.ceil((N - 1) / J)
    angles_t = 2 * k * (a - 2)
    angle_q = angle_circuit_qubits(R, B, brot, J)
    angle_temp = angle_q - math.ceil(math.log2(a)) - J * brot
    t_total = angles_t + h["outer"] + h["inner"] + h["givens"]
    q_total = h["persistent"] + J * brot + max(h["temp"], angle_temp)
    ok = angle_q <= CAP
    out = dict(R=R, B=B, C=C, brot=brot, J=J, K=k, angles_T=angles_t, **h,
               T=t_total, Q=q_total, TQ=t_total * q_total, angle_circuit_qubits=angle_q, within_cap=ok)
    if lam is not None:
        out["score"] = lam * t_total * q_total
    return out


if __name__ == "__main__":
    base = total(65, 60, 60, 12, 8, lam=49.301369)
    print("baseline (65,60,60) brot=12 J=8, README says T=93237 Q=730 score=3.3556e9")
    for k, v in base.items():
        print(f"  {k}: {v}")

    print("\nCandidate shapes at brot=12, J=8 (ANGLES upper bound; lambda_eff left at the baseline's 49.3):")
    print(f"{'R':>3} {'B':>4} {'C':>3} {'T':>8} {'Q':>5} {'TQ':>10}  vs baseline TQ")
    for R, B, C in [(65, 60, 60), (8, 143, 44), (8, 120, 44), (6, 143, 44), (8, 143, 30), (4, 200, 30)]:
        try:
            r = total(R, B, C, 12, 8)
        except Exception as ex:  # cost model raises ValueError for some shapes
            print(f"{R:>3} {B:>4} {C:>3}  invalid: {ex}")
            continue
        print(f"{R:>3} {B:>4} {C:>3} {r['T']:>8} {r['Q']:>5} {r['TQ']:>10.3e}  {base['TQ'] / r['TQ']:.2f}x")
