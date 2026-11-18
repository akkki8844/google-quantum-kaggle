"""C3 (Kaggle GPU): fewer copies C at fixed angle words (loaders, words, vectors unchanged).

Start: our v5 chemistry (8,143,44). Per group, rotate the C weight columns to the principal axes of
their X Gram (an exact gauge for error_2 and h1'), drop the weakest columns, then refit the weights
w (R,B,C') and offsets Wo (R,C') by penalty L-BFGS with growing penalty weight.
Acceptance only by the official numpy metrics and then the full official verifier. Cost model:
each copy removed is -0.53% T*Q, so lambda_eff may rise up to the break-even shown in the log.
"""
import glob, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np, torch

INPUT = Path("/kaggle/input"); OUT = Path("/kaggle/working"); T0 = time.time()
comp = next(p.parent for p in INPUT.rglob("verifiers") if (p / "dfthcsos").is_dir())
sys.path.insert(0, str(comp / "kickmix")); sys.path.insert(0, str(comp / "verifiers"))
from dfthcsos import verify as V
from dfthcsos.scoreinfra import dfthc_math as M
from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dt = torch.float64
E1, E2, GAPF = V.ERROR_1_THRESHOLD, V.ERROR_2_THRESHOLD, V.E_GAP_FLOOR
CS = [43, 42, 41, 40]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def stage():
    d = Path("/tmp/c3/dfthcsos")
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
    for p in INPUT.rglob("angle_k_0.kmb"):
        if "sample_submission" in str(p) or "verifiers" in str(p):
            continue
        if (p.parent / "submission.npz").exists():
            for q in p.parent.iterdir():
                if q.is_file():
                    shutil.copy(q, d / q.name)
            return d
    raise SystemExit("no package")


def tq(C, ang_t):
    s = SpinFreeData_V1(8, 143, C, 76, 10, 8)
    Th = s.outer_state_prep() + s.inner_state_prep() + s.givens_rotations()
    o, i = s.outer_state_prep_qubit_cost(), s.inner_state_prep_qubit_cost()
    return Th + ang_t, o["persistent"] + i["persistent"] + 40 + max(o["temporary"], i["temporary"], 77)


def main():
    pkg = stage()
    inputs = V.load_inputs(pkg / "submission.npz")
    edges = V.load_tree_edges(None, pkg)
    log("device", dev, "recovering words")
    base = V.evaluate_circuits(inputs, pkg, edges)
    words, brot = base["angle_words"], base["brot"]
    m0, s0 = base["metrics"], base["score"]
    ang_t = base["resources"]["angles_toffoli"]
    log("baseline", {k: m0[k] for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff")}, "score", s0)
    u_all = M.reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
    u_all = u_all / np.linalg.norm(u_all, axis=1, keepdims=True)
    w0, Wo0 = inputs.w.copy(), inputs.W.copy()
    N = inputs.num_orbitals
    R, B, C0 = w0.shape
    one, ub = M.split_u_vectors(u_all, w0)
    ref = inputs.ref
    oei, tei = np.asarray(ref["oei"]), np.asarray(ref["tei"])
    h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
    S, beta_1, eta = inputs.S, inputs.beta_1, ref["electrons"]
    from scipy.optimize import linear_sum_assignment
    X0 = M.reconstruct_x(ub, w0)
    hp0 = M.build_h1_prime(h1_thc, S, beta_1, eta, Wo0, X0)
    _, evec0 = np.linalg.eigh(hp0)
    qi, ii = linear_sum_assignment(-np.abs(one @ evec0))
    perm = np.empty(N, dtype=np.int64)
    perm[qi] = ii
    Tn = lambda a: torch.tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
    # projectors P (R,B,N^2); Gram PP; per-group target blocks K_r = P_r Mt P_r^T
    P = np.einsum("rbp,rbq->rbpq", ub, ub).reshape(R, B, N * N)
    Tt = tei.copy()
    di = np.arange(N)
    Tt[:, :, di, di] += 0.5 * S[:, :, None]
    Tt[di, di, :, :] += 0.5 * S[None, :, :]
    t2 = float(np.sum(Tt * Tt))
    Mt = Tn(Tt.reshape(N * N, N * N))
    del Tt
    Pt = Tn(P)
    K = torch.stack([Pt[r] @ Mt @ Pt[r].T for r in range(R)])
    del Mt
    PP = torch.einsum("rbx,sdx->rbsd", Pt, Pt)
    uu = Pt.reshape(R, B, N, N)
    O = Tn(np.einsum("jp,jq->jpq", one, one))
    base_h = Tn(h1_thc - 0.5 * eta * S + beta_1 * np.eye(N))
    permT = torch.tensor(perm, device=dev)
    ecore, Eup = float(ref["ecore"]), float(inputs.energy_upper)
    log("precompute done")

    def metrics(w, Wo):
        Gm = torch.einsum("rbc,rbsd,sde->rcse", w, PP, w)
        a2 = (Gm * Gm).sum()
        cross = torch.einsum("rbc,rbd,rdc->", w, K, w)
        e2 = torch.sqrt(torch.clamp(a2 - 2 * cross + t2, min=0.0))
        X = torch.einsum("rbc,rbpq->rcpq", w, uu)
        hp = base_h - torch.einsum("rc,rcpq->pq", Wo, X)
        hp = 0.5 * (hp + hp.T)
        lam = torch.linalg.eigvalsh(hp)
        e1 = torch.linalg.norm(torch.einsum("j,jpq->pq", lam[permT], O) - hp)
        lam_h1 = lam.abs().sum()
        wB = Wo + w.sum(1)
        Lam = lam_h1 + 0.25 * ((w.abs().sum(1) + wB.abs()) ** 2).sum()
        gap = Eup - (ecore - eta * beta_1 - (lam_h1 - lam.sum()) - 0.5 * (Wo ** 2).sum())
        return dict(e1=e1, e2=e2, Lam=Lam, gap=gap, leff=torch.sqrt(gap * (2 * Lam - gap)))

    with torch.no_grad():
        mm = metrics(Tn(w0), Tn(Wo0))
    log("torch start", {k: float(v) for k, v in mm.items()})
    me1, me2, mgap = E1 - 5e-9, E2 - 5e-9, GAPF + 5e-8
    rep = {"baseline_score": s0, "baseline_metrics": m0, "torch_start": {k: float(v) for k, v in mm.items()}, "runs": []}

    # gauge: rotate each group's columns to the principal axes of the X Gram (descending)
    w_rot, Wo_rot = w0.copy(), Wo0.copy()
    with torch.no_grad():
        Gm0 = torch.einsum("rbc,rbsd,sde->rcse", Tn(w0), PP, Tn(w0)).cpu().numpy()
    for r in range(R):
        ev, Q = np.linalg.eigh(Gm0[r, :, r, :])
        Q = Q[:, ::-1]
        w_rot[r] = w0[r] @ Q
        Wo_rot[r] = Wo0[r] @ Q
        log("group", r, "smallest X-Gram eigs", np.round(ev[:4], 6), "largest", np.round(ev[-2:], 4))
    with torch.no_grad():
        mm = metrics(Tn(w_rot), Tn(Wo_rot))
    log("after gauge (should equal start except Lambda)", {k: float(v) for k, v in mm.items()})

    best_all = None
    for Cn in CS:
        T_, Q_ = tq(Cn, ang_t)
        breakeven = s0 / (T_ * Q_)
        log(f"=== C={Cn}: T {T_} Q {Q_} break-even lambda_eff {breakeven:.5f}")
        w = Tn(w_rot[:, :, :Cn]).requires_grad_(True)
        Wo = Tn(Wo_rot[:, :Cn]).requires_grad_(True)
        with torch.no_grad():
            mm = metrics(w, Wo)
        log("after drop", {k: round(float(v), 7) for k, v in mm.items()})
        best = None
        for mu in [1e3, 1e5, 1e7, 1e9, 1e11]:
            opt = torch.optim.LBFGS([w, Wo], lr=1, max_iter=400, history_size=50, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-14, tolerance_change=1e-16)

            def closure():
                opt.zero_grad()
                m = metrics(w, Wo)
                pen = (torch.relu(m["e1"] - me1) ** 2 + torch.relu(m["e2"] - me2) ** 2
                       + torch.relu(mgap - m["gap"]) ** 2)
                f = m["leff"] + mu * pen
                f.backward()
                return f
            for _ in range(3):
                opt.step(closure)
            with torch.no_grad():
                mm = metrics(w, Wo)
            log(f"mu {mu:.0e}", {k: round(float(v), 9) for k, v in mm.items()})
            wn, Won = w.detach().cpu().numpy(), Wo.detach().cpu().numpy()
            om = M.evaluate_dfthc_angle_words(words, wn, Won, S, beta_1, ref, brot=brot, tree_edges=edges,
                                              energy_upper=inputs.energy_upper)
            ok = V.passes_accuracy(om)
            log("  official", "ok" if ok else "FAIL", om["lambda_eff"], om["error_1"], om["error_2"], om["E_gap"])
            if ok and (best is None or om["lambda_eff"] < best[0]):
                best = (om["lambda_eff"], wn.copy(), Won.copy(), om)
        row = {"C": Cn, "T": T_, "Q": Q_, "breakeven": breakeven,
               "best_lambda_eff": None if best is None else best[0],
               "proj_score": None if best is None else best[0] * T_ * Q_,
               "last_torch": {k: float(v) for k, v in mm.items()}}
        log("RESULT", row)
        rep["runs"].append(row)
        if best is not None and best[0] * T_ * Q_ < s0 and (best_all is None or best[0] * T_ * Q_ < best_all[0]):
            best_all = (best[0] * T_ * Q_, Cn, best)
        (OUT / "c3_report.json").write_text(json.dumps(rep, indent=1, default=float))
        if best is None:
            log("C", Cn, "infeasible; not going lower")
            break
    if best_all is not None:
        _, Cn, best = best_all
        outd = OUT / "dfthcsos_c" / "dfthcsos"
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
        rep["official"] = r
        log("OFFICIAL", r.get("status"), r.get("score"), "vs baseline", s0)
    (OUT / "c3_report.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
