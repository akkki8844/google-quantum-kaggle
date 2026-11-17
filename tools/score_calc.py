"""Score arithmetic for the Verifiable Quantum Circuit Compilation competition.

All formulas are taken from arXiv 2610.23832 and the handoff notes. Nothing here
reads competition-gated files. Verify against SCORING_README once it is available.

Run:  python -I tools/score_calc.py
"""
import math

DELTA = 2.0 ** -128
N_SHOTS = 102_400


def log_binom_cdf(k, n, p):
    """log P(X <= k) for X ~ Binomial(n, p), via log-sum-exp."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return -math.inf if k < n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = []
    for j in range(k + 1):
        lc = math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        terms.append(lc + j * lp + (n - j) * lq)
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def eps_plus(k, n=N_SHOTS, delta=DELTA):
    """One-sided Clopper-Pearson upper bound: largest p with P(X<=k) >= delta."""
    target = math.log(delta)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if log_binom_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def repetition_factor(eps):
    """R(eps) = ln2 / -ln(1/2 + 2 sqrt(eps)); infinite for eps >= 1/16."""
    if eps >= 1 / 16:
        return math.inf
    return math.log(2) / -math.log(0.5 + 2 * math.sqrt(eps))


def dqi_cost(q, t, failures=0):
    return repetition_factor(eps_plus(failures)) * q * q * t


def geomean(*xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


if __name__ == "__main__":
    print("== DQI failure budget (N=102,400, delta=2^-128) ==")
    base_r = repetition_factor(eps_plus(0))
    for k in (0, 1, 2, 5, 10, 20, 50, 100):
        e = eps_plus(k)
        r = repetition_factor(e)
        print(f"k={k:4d}  eps+={e:.3e}  R={r:.4f}  cost penalty vs k=0: {r / base_r:.4f}x")

    print("\n== Reference costs (from handoff) ==")
    ref_dqi = dqi_cost(1893, 5_725_461)
    print(f"dqi reference  = {ref_dqi:.4e}")
    print(f"invsqrt ref Q*C = {260 * 7988:.4e}")

    print("\n== Where the public notebooks sit ==")
    inv_ratio, dqi_ratio, chem_ratio = 27.6, 70.3, 11.4
    print(f"geomean(27.6, 70.3, 11.4) = {geomean(inv_ratio, dqi_ratio, chem_ratio):.3f}")
    t_dqi = (ref_dqi / dqi_ratio) / (base_r * 890 ** 2)
    print(f"implied dqi T at Q=890, k=0: {t_dqi:,.0f}")
    print(f"implied invsqrt C at Q=44: {(260 * 7988 / inv_ratio) / 44:,.0f}")

    print("\n== What one task has to improve to reach the #1 score (30.285) ==")
    cur = geomean(inv_ratio, dqi_ratio, chem_ratio)
    need = (30.285 / cur) ** 3
    print(f"single-task multiplier needed: {need:.3f}x  (or ~{need ** (1 / 3):.3f}x on each)")

    print("\n== Sensitivity: score gain from improving ONE task by m ==")
    for m in (1.05, 1.10, 1.25, 1.5, 2.0):
        print(f"m={m:.2f}  score {cur:.3f} -> {cur * m ** (1 / 3):.3f}")

    print("\n== DQI floor: syndrome register is 70*12 = 840 qubits ==")
    for q in (890, 880, 870, 860, 850, 845):
        print(f"Q={q}: Q^2 gain vs 890 = {(890 / q) ** 2:.4f}x")
