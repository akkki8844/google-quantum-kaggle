"""Capacity screen: can small-group DF-THC shapes (B<=23, many groups) fit the two-body
target as well as the shipped (8,143,44) shape? Pure continuous error_2 fit (Gram objective,
torch on GPU), identical init recipe and budget for every shape. No Lambda/E_gap terms, no
quantization: a shape that cannot beat ~13 mHa here is dead."""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch

INPUT = Path("/kaggle/input"); OUT = Path("/kaggle/working")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, flush=True)
SHAPES = [(8, 143, 44), (16, 72, 22), (26, 44, 14), (44, 26, 8), (52, 22, 7), (64, 18, 6), (72, 16, 5), (40, 24, 9)]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
BUDGET = 11.0 * 3600

ver = next(p for p in INPUT.rglob("verify.py") if p.parent.name == "dfthcsos").parent.parent
sys.path.insert(0, str(ver))
from dfthcsos.scoreinfra.reference import load_reference, load_fixed_shifts
from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower
N = 76
ref = load_reference(next(INPUT.rglob("FeMoCo76o113e.fcidump")))
fs = load_fixed_shifts()
S = symmetric_from_lower(fs["Hsym_lower"], N).astype(np.float64)
G = np.asarray(ref["tei"], dtype=np.float64).copy(); d = np.arange(N)
G[:, :, d, d] += 0.5 * S[:, :, None]; G[d, d, :, :] += 0.5 * S[None, :, :]
G = G.reshape(N * N, N * N); G = 0.5 * (G + G.T)
ev, V = np.linalg.eigh(G); o = np.argsort(-ev); ev, V = ev[o], V[:, o]
tot2 = float(np.sum(ev ** 2))
print("EY K=352 mHa", 1e3 * math.sqrt(tot2 - np.sum(ev[:352] ** 2)), flush=True)
Gt = torch.tensor(G, device=dev, dtype=torch.float32)
G2 = float(np.sum(G * G))

def init(R, B, C, rng):
    K = R * C
    Ls = [(math.sqrt(max(ev[k], 0)) * V[:, k].reshape(N, N)) for k in range(K)]
    Ls = [0.5 * (L + L.T) for L in Ls]
    # greedy grouping: assign components round-robin by energy so groups are balanced
    groups = [list(range(r, K, R)) for r in range(R)]
    U = np.zeros((R, B, N)); W = np.zeros((R, B, C))
    for r, g in enumerate(groups):
        M = np.concatenate([Ls[k] for k in g], 1)        # N x N*C
        u, s, _ = np.linalg.svd(M, full_matrices=False)
        nb = min(B, N)
        vecs = [u[:, i] for i in range(nb)]
        while len(vecs) < B:                              # B > N: add mixed vectors
            a = rng.standard_normal(nb) * s[:nb]; x = u[:, :nb] @ a; vecs.append(x / np.linalg.norm(x))
        Ug = np.stack(vecs)                                # B x N
        P = np.stack([np.outer(x, x).ravel() for x in Ug], 1)  # N^2 x B
        for c, k in enumerate(g):
            W[r, :, c] = np.linalg.lstsq(P, Ls[k].ravel(), rcond=None)[0]
        U[r] = Ug
    return U, W

def loss_fn(U, Wt):
    u = U / U.norm(dim=2, keepdim=True)
    X = torch.einsum("rbp,rbq,rbc->rcpq", u, u, Wt).reshape(-1, N * N)  # K x N^2
    gram = X @ X.T
    t1 = (gram * gram).sum(); t2 = ((X @ Gt) * X).sum()
    return torch.sqrt(torch.clamp(t1 - 2 * t2 + G2, min=0))

summary = []; t0 = time.time()
for (R, B, C) in SHAPES:
    if time.time() - t0 > BUDGET: break
    rng = np.random.default_rng(0)
    U0, W0 = init(R, B, C, rng)
    U = torch.tensor(U0, device=dev, dtype=torch.float32, requires_grad=True)
    Wt = torch.tensor(W0, device=dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([U, Wt], lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
    hist = []; ts = time.time()
    for it in range(STEPS + 1):
        opt.zero_grad(); L = loss_fn(U, Wt); L.backward(); opt.step(); sched.step()
        if it % 2000 == 0 or it == STEPS:
            hist.append((it, 1e3 * float(L))); print(R, B, C, it, "e2 mHa %.3f" % (1e3 * float(L)), "%.0fs" % (time.time() - ts), flush=True)
    row = {"shape": [R, B, C], "init_mHa": hist[0][1], "final_mHa": hist[-1][1], "hist": hist, "sec": round(time.time() - ts)}
    summary.append(row); (OUT / "c2_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(OUT / f"fit_{R}_{B}_{C}.npz", U=U.detach().cpu().numpy(), W=Wt.detach().cpu().numpy())
print(json.dumps([(r["shape"], round(r["final_mHa"], 3)) for r in summary]), flush=True)
