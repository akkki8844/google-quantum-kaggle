"""R1 (Kaggle GPU): constrained refit of the chemistry weights at fixed angle words (loaders unchanged).

Variables: lowercase weights w (R,B,C) and identity offsets Wo (R,C). Everything else is fixed: the decoded vectors
(from the official word recovery of our current circuits), S, beta_1, the FCIDUMP. Objective: lower lambda_eff while
keeping the official gates: error_1 <= 20 mHa, error_2 <= 20 mHa, E_gap > 3.914898, Lambda > 49.
Acceptance is decided ONLY by the official numpy metrics (dfthc_math) and then the full official verifier.

Inputs: our current package (submission.zip from our v2 notebook output).
Outputs: r1_report.json, and dfthcsos_refit/ (same circuits + refitted submission.npz) if it is strictly better.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
T0 = time.time()
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
sys.path.insert(0, str(comp / "kickmix"))
sys.path.insert(0, str(comp / "verifiers"))
from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as M  # noqa: E402

E1 = V.ERROR_1_THRESHOLD
E2 = V.ERROR_2_THRESHOLD
GAPF = V.E_GAP_FLOOR
LAMF = V.LAMBDA_FLOOR
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = torch.float64


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def stage_package() -> Path:
    d = Path("/tmp/r1/dfthcsos")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for z in glob.glob(str(INPUT / "**" / "submission.zip"), recursive=True):
        with zipfile.ZipFile(z) as f:
            names = [n for n in f.namelist() if n.startswith("dfthcsos/")]
            if names:
                for n in names:
                    (d / Path(n).name).write_bytes(f.read(n))
                return d
    raise SystemExit("no package")


def official_metrics(inputs, words, edges, brot, w, Wo):
    return M.evaluate_dfthc_angle_words(words, w, Wo, inputs.S, inputs.beta_1, inputs.ref, brot=brot,
                                        tree_edges=edges, energy_upper=inputs.energy_upper)


def main():
    pkg = stage_package()
    inputs = V.load_inputs(pkg / "submission.npz")
    edges = V.load_tree_edges(None, pkg)
    log("device", dev, "loading circuits / recovering words")
    base = V.evaluate_circuits(inputs, pkg, edges)
    words, brot = base["angle_words"], base["brot"]
    m0 = base["metrics"]
    log("baseline", {k: m0[k] for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff")}, "score", base["score"])
    rep = {"baseline_metrics": m0, "baseline_score": base["score"], "resources": base["resources"]}

    u_all = M.reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
    u_all = u_all / np.linalg.norm(u_all, axis=1, keepdims=True)
    w0 = inputs.w.copy()
    Wo0 = inputs.W.copy()
    N = inputs.num_orbitals
    R, B, C = w0.shape
    one, ub = M.split_u_vectors(u_all, w0)
    ref = inputs.ref
    oei, tei = np.asarray(ref["oei"]), np.asarray(ref["tei"])
    h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
    S, beta_1, eta = inputs.S, inputs.beta_1, ref["electrons"]
    # fixed eigen-assignment (as the verifier makes it) at the start point
    X0 = M.reconstruct_x(ub, w0)
    hp0 = M.build_h1_prime(h1_thc, S, beta_1, eta, Wo0, X0)
    ev0, evec0 = np.linalg.eigh(hp0)
    from scipy.optimize import linear_sum_assignment
    qi, ii = linear_sum_assignment(-np.abs(one @ evec0))
    perm = np.empty(N, dtype=np.int64)
    perm[qi] = ii  # one-body row j gets eigenvalue index perm[j]

    T = lambda a: torch.tensor(np.asarray(a), dtype=dt, device=dev)
    uu = T(np.einsum("rbp,rbq->rbpq", ub, ub))                     # (R,B,N,N)
    O = T(np.einsum("jp,jq->jpq", one, one))                        # (N,N,N)
    base_h = T(h1_thc - 0.5 * eta * S + beta_1 * np.eye(N))
    # error_2 target tensor (tei + S terms) as a (N^2, N^2) matrix
    Tt = tei.copy()
    di = np.arange(N)
    Tt[:, :, di, di] += 0.5 * S[:, :, None]
    Tt[di, di, :, :] += 0.5 * S[None, :, :]
    Mt = T(Tt.reshape(N * N, N * N))
    t2 = float(np.sum(Tt * Tt))
    del Tt
    permT = torch.tensor(perm, device=dev)
    ecore, Eup = float(ref["ecore"]), float(inputs.energy_upper)

    def metrics(w, Wo):
        X = torch.einsum("rbc,rbpq->rcpq", w, uu)
        Xf = X.reshape(R * C, N * N)
        G = Xf @ Xf.T
        a2 = (G * G).sum()
        cross = ((Xf @ Mt) * Xf).sum()
        e2 = torch.sqrt(torch.clamp(a2 - 2 * cross + t2, min=0.0))
        hp = base_h - torch.einsum("rc,rcpq->pq", Wo, X)
        hp = 0.5 * (hp + hp.T)
        lam = torch.linalg.eigvalsh(hp)
        rec = torch.einsum("j,jpq->pq", lam[permT], O)
        e1 = torch.linalg.norm(rec - hp)
        lam_h1 = lam.abs().sum()
        wB = Wo + w.sum(1)
        lam_h2 = 0.25 * ((w.abs().sum(1) + wB.abs()) ** 2).sum()
        Lam = lam_h1 + lam_h2
        esos = ecore - eta * beta_1 - (lam_h1 - lam.sum()) - 0.5 * (Wo ** 2).sum()
        gap = Eup - esos
        leff = torch.sqrt(gap * (2 * Lam - gap))
        return dict(e1=e1, e2=e2, Lam=Lam, gap=gap, leff=leff)

    w = T(w0).requires_grad_(True)
    Wo = T(Wo0).requires_grad_(True)
    with torch.no_grad():
        mm = metrics(w, Wo)
    log("torch start", {k: float(v) for k, v in mm.items()})
    rep["torch_start"] = {k: float(v) for k, v in mm.items()}
    # margins kept below the official thresholds
    me1, me2, mgap = E1 - 2e-9, E2 - 2e-9, GAPF + 2e-8

    best = (float(m0["lambda_eff"]), w0.copy(), Wo0.copy(), m0)
    nw = w0.size

    def pack(a, b):
        return torch.cat([a.reshape(-1), b.reshape(-1)])

    def unpack(x):
        return x[:nw].reshape(R, B, C), x[nw:].reshape(R, C)

    def evalg(x):
        """f and constraint values g_i <= 0 (e1, e2, gap floor) with gradients."""
        x = x.detach().requires_grad_(True)
        a, b = unpack(x)
        mm = metrics(a, b)
        vals = [mm["leff"], mm["e1"] - me1, mm["e2"] - me2, mgap - mm["gap"]]
        grads = [torch.autograd.grad(v, x, retain_graph=True)[0] for v in vals]
        return [float(v) for v in vals], grads

    def restore(x, iters=6):
        """Gauss-Newton push back onto the feasible side of violated/active constraints."""
        for _ in range(iters):
            vals, grads = evalg(x)
            viol = [i for i in (1, 2, 3) if vals[i] > -1e-10]
            if not any(vals[i] > 0 for i in (1, 2, 3)):
                return x, vals
            G = torch.stack([grads[i] for i in viol], 1)
            r = torch.tensor([vals[i] + 2e-10 for i in viol], dtype=dt, device=dev)
            lamb = torch.linalg.solve(G.T @ G + 1e-30 * torch.eye(len(viol), dtype=dt, device=dev), r)
            x = (x - G @ lamb).detach()
        return x, evalg(x)[0]

    x = pack(T(w0), T(Wo0)).detach()
    for it in range(400):
        vals, grads = evalg(x)
        gf = grads[0]
        act = [i for i in (1, 2, 3) if vals[i] > -1e-7]
        d = -gf
        if act:
            G = torch.stack([grads[i] for i in act], 1)
            # projected gradient: d = -(gf + G lam), lam >= 0 via small NNLS (active set <= 3)
            from scipy.optimize import nnls
            Gc, gc = G.cpu().numpy(), gf.cpu().numpy()
            lam, _ = nnls(Gc, -gc)
            d = -(gf + G @ torch.tensor(lam, dtype=dt, device=dev))
        dn = float(torch.linalg.norm(d))
        if dn < 1e-14:
            log("stationary at it", it)
            break
        improved = False
        for alpha in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]:
            step = alpha / dn
            xn, vn = restore((x + step * d).detach())
            if all(vn[i] <= 0 for i in (1, 2, 3)) and vn[0] < vals[0] - 1e-12:
                x, improved = xn, True
                break
        if not improved:
            log("no feasible descent at it", it, "f", vals[0])
            break
        if it % 20 == 0 or it < 5:
            a, b = unpack(x)
            wn, Won = a.cpu().numpy(), b.cpu().numpy()
            om = official_metrics(inputs, words, edges, brot, wn, Won)
            ok = V.passes_accuracy(om)
            log(f"it {it}", "torch f", round(vn[0], 9), "official", "ok" if ok else "FAIL", om["lambda_eff"],
                om["error_1"], om["error_2"], om["E_gap"], "alpha", alpha)
            if ok and om["lambda_eff"] < best[0]:
                best = (om["lambda_eff"], wn.copy(), Won.copy(), om)
    a, b = unpack(x)
    wn, Won = a.cpu().numpy(), b.cpu().numpy()
    om = official_metrics(inputs, words, edges, brot, wn, Won)
    if V.passes_accuracy(om) and om["lambda_eff"] < best[0]:
        best = (om["lambda_eff"], wn.copy(), Won.copy(), om)
    rep["best_lambda_eff"] = best[0]
    rep["best_metrics"] = best[3]
    rep["gain_rel"] = 1 - best[0] / float(m0["lambda_eff"])
    log("best lambda_eff", best[0], "vs", m0["lambda_eff"], "rel gain", rep["gain_rel"])
    if best[0] < float(m0["lambda_eff"]):
        outd = OUT / "dfthcsos_refit" / "dfthcsos"
        outd.mkdir(parents=True, exist_ok=True)
        for p in pkg.iterdir():
            if p.name != "submission.npz":
                shutil.copy(p, outd / p.name)
        with np.load(pkg / "submission.npz", allow_pickle=False) as z:
            arrs = {k: np.array(z[k]) for k in z.files}
        arrs["W"] = best[1].astype(arrs["W"].dtype)
        arrs["H2_I"] = (best[2] + best[1].sum(axis=1)).astype(arrs["H2_I"].dtype)
        np.savez(outd / "submission.npz", **arrs)
        v = subprocess.run([sys.executable, "dfthcsos/verify.py", str(outd)], cwd=comp / "verifiers",
                           env=dict(os.environ, PYTHONPATH=str(comp / "kickmix")), capture_output=True, text=True)
        try:
            r = json.loads(v.stdout[v.stdout.index("{"):])
        except ValueError:
            r = {"tail": v.stdout[-2000:], "err": v.stderr[-2000:]}
        rep["official_refit"] = r
        log("official refit", r.get("status"), r.get("score"), "baseline", base["score"])
    (OUT / "r1_report.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
