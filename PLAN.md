# Kaggle: Verifiable Quantum Circuit Compilation — master plan

Last updated 2026-11-16 (evening). This file is the single source of truth for strategy.
Tags: **[V]** verified from the official verifier/README code, **[K]** read on Kaggle (public pages, 2026-11-16), **[P]** paper, **[I]** inference to test.

---

## 1. Situation in one screen

| | |
|---|---|
| Today | 2026-11-16. **65 days** to the final deadline (2027-01-20 23:59 UTC). |
| Field | 3,181 teams. #1 30.531 (aadi_stonks), #2 30.285, #3 30.131, #10 29.010, #49 28.151 [K]. |
| Public frontier | 28.0949 (boristown 28.09535; maoxin123 v27 28.09486) [K]. |
| Our local state | Data, starter kit and kickmix (venv) installed. Chemistry cost model reproduces the baseline exactly. **Reference scores not yet confirmed locally** (Windows runs stalled; you are fixing that). |
| To beat #1 today | Need ×1.292 on the product of the three task ratios vs the public frontier (= 30.6). Equivalent to ×1.29 on one task, or ×1.09 on each. |

Public frontier per task (maoxin123 v27 notebook, verified there with the official verifier) [K]:

| Task | Q | Cost | Ratio | What it implies |
|---|---|---|---|---|
| invsqrt | 44 | 75,006.38 | 27.688 | 1,464.13 mean Toffolis, 33 fine rotations, worst error/tolerance **0.99956** |
| DQI | 890 | 347,375,470,572 | 70.355 | T ≈ 368,128 at k=0 failures (R=1.1913) |
| Chemistry | 427 | 294,762,733 | 11.384 | λ_eff·T ≈ 690k |

---

## 2. Rubric checklist (everything that can reject or price a submission) [V]

### Submission package
- Zip containing `invsqrt.kmb|kmx`, `dqi.kmb|kmx`, `dfthcsos/` with `angle_k_0..K-1.kmb|kmx` (no gaps) and `submission.npz`.
- 5 submissions/day, ~45 min scoring each, pick 2 finals (else auto-picked). Public LB = ~50% of tests; private = other 50%.
- Seeds come from the SHA-256 of the canonical `.kmb` bytes (+ secret salt). **Any byte change gives fresh tests.**

### invsqrt (`verifiers/invsqrt/verify_submission_invsqrt.py`)
- Exactly one register, named `x`, 22 distinct qubits. Every other qubit is scratch and must end at |0>.
- Every x in 1..3,133,452 × 64 shots (200,540,928 shots). Zero failures allowed.
- Pass if |phase − 1/√x| ≤ tol(x) = 2^-24/√x turns (mod 1), in 2^-64 fixed point.
- Cost = Q · (mean Toffoli-like per shot + (T + rotation-T)/4).
  - **Means are dynamic**: a gate conditioned on a measured bit counts only on shots where the bit is 1.
  - Clifford Z_POW is free. T-angles cost 1 T.
  - Other Z_POW: n_rot × (0.56·log2(1/eps) + 5.3) T, with eps = π·min_margin/2^64/n_rot, min_margin = the smallest tol − |err| over all shots.

### DQI (`verifiers/dqi/verify_submission_dqi.py`)
- Registers exactly: `S0..S69` (12 qubits each, quantum) and `E0..E4094` (12 **classical bits** each). No other named registers. **No Z_POW anywhere.**
- 102,400 shots.
  - Error weight k is drawn ~ w_k² (mean ≈ 31).
  - Positions are uniform and distinct. Values are uniform nonzero. M is uniform.
- A shot fails if the syndromes aren't restored, any ancilla is dirty, or the phase is wrong.
- Cost = R(eps+) · Q² · T.
  - T = mean Toffoli-like per shot (dynamic).
  - eps+ = Clopper–Pearson at 2^-128.
  - k=0 → R=1.1913. k=10 → +3.2%. k=20 → +5.3%. k=50 → +10.3%. eps+ ≥ 1/16 → rejected.
  - (`tools/score_calc.py` reproduces these numbers.)
- Q counts qubits only. **840 syndrome qubits are a hard floor.**

### Chemistry (`verifiers/dfthcsos/verify.py`, `SCORING_README.md`)
- ANGLE_k circuits:
  - Registers `selection` (exactly ceil(log2(76+R·B)) qubits) and `word_0..word_{J-1}` (brot qubits each). Everything else is workspace and must be clean after **each** circuit.
  - No Z_POW. ≤ **128 qubits per circuit**, including the word registers.
- Simulation: every address runs forward and then in reverse, 64 shots each. Phase, selection, workspace, padding and reverse-clearing are all checked, and repeated samples must agree.
- `submission.npz`: `U, W, H2_I, H1_I, Hsym_lower, tree_edges`.
  - **Only the shape of U is used**; the vectors are decoded from the circuits.
  - H1_I and Hsym_lower are ignored (the fixed baseline shift is used).
  - Saved with `allow_pickle=False`.
- Gates on the **quantized** words: error_1 ≤ 20 mHa and error_2 ≤ 20 mHa; Λ > 49.0; E_gap > 3.914898; 0 < E_gap < 2Λ. Failing any gate gives an infinite score.
- Score = λ_eff · T_total · Q_total.
  - λ_eff = √(E_gap(2Λ − E_gap)).
  - T_total = ANGLES (measured, both passes) + outer + inner (closed form) + 2·75·(2·brot − 1).
  - Q_total = outer/inner persistent + J·brot + max(outer_temp, inner_temp, angles_temp).
  - chi is fixed to 10.
  - `tools/chem_cost_explorer.py` reproduces the baseline exactly (T 93,237, Q 730, score 3.3556e9).
- Open host question (thread 755427, unanswered): ANGLE_k is specified as an XOR oracle, but the verifier only tests reachable word states. **Do not rely on reachable-state-only tricks for the finals** unless the host clarifies; they could be ruled invalid.

### Prize eligibility (host answer, thread 752010) [K]
- **"A circuit generator that produces the exact circuits must be provided. The requirement applies for all benchmarks and all prizes."**
- So every circuit in our 2 final submissions needs runnable generator code in our repo, including the tasks we don't optimize ourselves.
- Pasting a public `.kmb` blob is *not* enough. A public Apache-2.0 *generator* (with its source) is fine, with credit.
- Winners must license their code Apache-2.0 and document the method (Rules B5, B7).
- The Syed Asad Ali lineage (46/44-qubit invsqrt generator, 889/890-qubit DQI decoder) carries a NOTICE (Apache-2.0 §4(d)): name him first in the credits of any derivative, at any depth. This costs nothing; just keep a NOTICE file.

---

## 3. Strategy (the efficient route)

1. **Start from the public frontier, never from the Google baseline.** Every top-50 team builds on the same public lineage (Syed Asad Ali → boristown/maoxin123/others). Re-deriving 28.09 would waste weeks. We take the public **generators** (not just circuits) and reproduce 28.09 locally first.
2. **Our edge has to come from the leftover ×1.29.** The public frontier is ~2 points behind #1, so the private leaders have improvements the public doesn't. Assume they found real per-task wins, not luck.
3. **Allocate effort by (expected gain ÷ effort):**
   - **Chemistry first.** Most free knobs and the least studied task (11.38× vs 70× for DQI). Wins can come from several independent sources:
     - better factorizations, with λ_eff able to fall up to 2.5× in principle;
     - shape, brot and J;
     - cheaper angle loaders, since ANGLES dominates T.
     - Training runs on a Kaggle GPU.
   - **invsqrt second.** Quick, deterministic and fully checkable offline: an exhaustive numpy model over all x. Small Q means each qubit is worth ~2.3% of the task.
   - **DQI third.**
     - The qubit floor leaves ≤ ×1.11 from Q.
     - T gains need deep decoder work: Berlekamp–Massey, the Chien scan, the failure budget.
     - Highest skill ceiling, slowest loop (6 GB RAM per verifier worker, minutes per run).
4. **Single-task prizes ($10k each, 3 tasks):** these go to the best score on one task among *selected* submissions. A focused chemistry push could win one even if our overall rank is outside the top 6. Keep this in mind when choosing finals.
5. **Ignore the Nov 20 milestone** (we're 2.4 points behind #1 with 4 days left) and probably the Longest Leader prize (needs public notebooks holding #1).

---

## 4. Timeline (time-boxed)

| Dates | Phase | Exit criterion |
|---|---|---|
| **Nov 16–18** | **P0 Infrastructure** | All 3 official verifiers pass locally (or on Kaggle) on the reference *and* on the public frontier files; harness and logging in place. |
| **Nov 18–Dec 2** | **P1 Fast wins** | invsqrt and chemistry beat the public frontier locally; first submission above 28.1. |
| **Dec 2–Dec 23** | **P2 Deep work** | DQI decoder work, chemistry training sweeps, agentic search loops; target ≥ 30. |
| **Dec 23–Jan 13** | **P3 Push** | Best-of combos of our own circuits. Solo: no merger (user decision 2026-11-17). Jan 13 is the entry deadline. |
| **Jan 13–Jan 20** | **P4 Freeze** | Generators reproduce the final files byte-for-byte; finals chosen (one safe, one aggressive); docs and NOTICE written. |

### Compute decision (2026-11-16): all heavy work runs on Kaggle
The local CPU can't handle full verifier runs, so local work is limited to editing code and quick numpy checks.

Workflow:
1. Write the kernel code locally in `kaggle_nb/<step>/<step>.py` (one `.py` per folder).
2. Push it as a **private** Kaggle script kernel with `tools/kpush.py push kaggle_nb/<step>`. Add `--gpu` for chemistry training and `--dataset owner/slug` for public inputs.
3. Check progress with `tools/kpush.py status ...`.
4. Pull the results JSON and logs with `tools/kpush.py output ...`.

Setup (you, once):
- On kaggle.com go to Settings → API → **Create New Token**, which downloads `kaggle.json`.
- Save it at `C:\Users\smart\.kaggle\kaggle.json`. The script reads only your username from it.
- The `kaggle` CLI is already installed in `.venv`.

First kernel: `kaggle_nb/p0_reference`. It runs all three official verifiers on the sample submission, using the competition's own kickmix build. Expected output: three PASS results at score 1.0, plus timings. Those timings let us budget every later run.

### P0 detail (do first)
1. **Fix compute** (you're on it). Options:
   - **WSL2 Ubuntu.** `fork` works there, so the verifiers parallelize (invsqrt is serial on Windows: 200M shots). Use `--max-workers 2` for DQI because of RAM (15.7 GB vs ~6 GB/worker).
   - **Kaggle notebooks.** 4 cores, 30 GB RAM, GPU for chemistry training. That is also where the leaderboard runs.
   - Re-run the three references and confirm score 1.0. This is a gate.
2. **Get the public generators** (you download them; downloads need your account):
   - a. Dataset `syed3000/vqcc-own-circuit-portfolio`. Version 12 has the builders' full source; version 13 has the 28.04 circuits and the NOTICE.
   - b. Dataset `qcc-frontier-dqi-chem-mirror`, used by maoxin123 v27. It holds the DQI `ccc2ab84` and chemistry `f6e12f02` frontier files.
   - c. Notebook `maoxin123/qcc-v27-a-44-qubit-invsqrt-on-the-28-09-frontier` (Download → .ipynb). It has the 44-qubit invsqrt embedded, credits, and their verification cells.
   - d. Notebook `boristown/qcc-gap-3-residue-repair-and-c43-native-compiler` (top public, 28.09535) and its input datasets.
   - e. Nice to have: the "28.09 LB | Which Circuits Survive Private Salt?" notebook, which covers private-salt robustness.
3. **Check generator coverage.** For each frontier file, find the code that rebuilds it byte-for-byte (SHA-256 match). Any file without a generator gets flagged: either we write the generator or we use the best file that has one.
4. **Harness** (in this folder, git-tracked):
   - `gen/<task>/`: our generators.
   - `bench/run_<task>.py`: wraps the official verifier and prints Q, T, score and margin, plus the per-task ratio and the geo-mean.
   - Fast pre-checks:
     - invsqrt: numpy model over all x (seconds), then the full verifier.
     - DQI: a 4,096-shot smoke test with an explicit `--seed`, then the full run with 102,400 shots.
     - Chemistry: the verifier itself (~25 s on Linux).
   - `log/experiments.csv` columns: date, task, id, change, Q, T, score, ratio, k-failures, seeds tried, notes.
   - Profiler: `builder.mark()` tags plus `flame_chart_svg()`, to see Toffolis and live qubits per phase.

---

## 5. Task plans with ranked ideas

Each idea lists the expected gain on that task, then on the overall score (gain ÷ 3 in log space).

### 5.1 Chemistry (current 11.38×, cost 2.948e8)

First, reproduce `f6e12f02` and decompose it into: λ_eff, Λ, E_gap, error_1/2, ANGLES T, outer, inner, Givens, Q parts, R/B/C, brot, J, K. **Everything else waits on that table.**

1. **Factorization search on Kaggle GPU** (starter `train_df_thc_qaggle.py`, JAX + optax).
   - Sweep shapes (R,B,C) seeded from the frontier factorization, and choose shapes from the closed-form cost model `tools/chem_sweep.py` (T·Q only; it shows where the formula pressure is).
   - The objective is ρ·λ_eff plus penalties at 20 mHa. Tune ρ and ε (the starter uses ε = 10 mHa for safety; the gate is 20 mHa on *quantized* words). Spend the budget: train at about 17–18 mHa, then check after quantizing.
   - λ_eff ≈ √(2·Λ·E_gap). The baseline Λ is 308 against a floor of 49, so lowering Λ is the biggest lever. E_gap has only ~1% of room.
2. **brot and J.**
   - Lower brot shrinks words, Givens (2·75·(2brot−1)) and qubits, but raises the quantization error. Use the starter's `brot_sweep`.
   - J sets K = ceil(75/J). ANGLES ≈ 2·K·A Toffolis with A = 76 + R·B, and each circuit needs J·brot + selection + ancilla ≤ 128.
3. **Cheaper angle loaders.**
   - **Exploit dynamic counting**: gates conditioned on measured bits count only when the bit is 1, so MBU fixups with CCZ cost 0.5.
   - Pruned / shared-subtree QROM: reorder rows so equal deltas form repeated subtrees (Gu et al. got 41k→10k T on an alias table this way).
   - Choose the tree and sibling order to make words repeat.
   - QROAM (select-swap) where the 128-qubit cap allows.
   - A smaller A (fewer two-body rows) via shared/merged vectors.
4. **Quantization-aware refit.** Fit the factorization *after* rounding to brot bits (the frontier did a "weight refit"). Optimize words directly against error_1/error_2.
5. Re-check the floors with margin. Λ and E_gap are strict inequalities on float32 inputs, so leave ≥ 1e-4 headroom.

Expected: ×1.2–2.0 on the task (+6–25% on the overall score). Highest uncertainty, highest ceiling.

### 5.2 invsqrt (current 27.69×; Q=44, C=1,704.69 = 1,464.13 Toffoli + 240.6 rotation-equivalent)

Numbers to keep in mind:
- 1 qubit ≈ 2.3% of the task.
- 1 Toffoli ≈ 0.06%.
- 1 rotation ≈ 7.3 Toffoli-eq (≈ 0.43%).

1. **Margin/rotation trade (corrected 2026-11-16: small).**
   - Back-solving the frontier's rotation cost (29.16 T per rotation) gives eps ≈ 2^-42.6 and a minimum margin ≈ 2^-39.2 turns, about 4.7% of the tightest tolerance.
   - Quadrupling that margin saves only ~1.1 T × 33 ≈ 9 Toffoli-eq (−0.5%).
   - Low priority.
2. **Fewer rotations.**
   - Drop or merge output bits whose angle is below the residual budget (tol(max x) ≈ 2^-34.8 turns).
   - Put T-angle bits where possible: 1 T instead of ~29.
   - Any rotation that can become Clifford is free.
3. **One more qubit** (44→43 = −2.3%).
   - Liveness analysis per gate: someone measured a 63-qubit peak covering 31% of ops in an older version, so the peak is narrow.
   - Recompute instead of storing.
   - Fold the table load into the value register (the frontier already did two such cuts).
4. **Toffoli trims.**
   - Measurement-based uncompute everywhere: del_and, del_lookup, CCZ-if-b counting 0.5.
   - Larger, coarser lookup tables with MBU cleanup (Gu: 17 T vs 126 T to reverse).
   - Fewer pieces: a 64-row table now; try a 128-row table with quadratic pieces.
5. **Gate:** the numpy model must equal the circuit on all x (we'll write it), and the full official verifier must pass at ≥ 2 salts before any submission.

Expected: ×1.03–1.08 on the task (+1–2.6% on the overall score). Low risk, fast loop.

### 5.3 DQI (current 70.36×; Q=890, T≈368k, k=0)

1. Profile the frontier decoder by phase (BM/EEA forward, Chien/Forney scan, reverse).
2. **Failure budget as a resource.** Adaptive coefficient windows, truncated BM rounds and early-stopped scans. Accept k ≲ 10–20 per 102,400 shots (≤ +5% R) when that saves more T. **Test on ≥ 5 seeds**; the private half re-draws.
3. **Scan cost.** 4,095 positions × (zero test + conditional product).
   - Exploit dynamic counting: conditioned work counts only when the measured bit is 1.
   - Chien-step constant multiplies are CNOT-only.
   - Use `gf2_phase_by_product` (zero Toffoli).
   - The trace trick: one parity bit per position, never materializing e_j (Gu et al.).
4. **Qubits** (each = 0.22% of cost ×2 because Q is squared): overlay registers (σ/Ω inside the syndrome words), and keep 1-entry history stacks with MBU.
5. Gate: zero or a budgeted number of failures on ≥ 5 seeds; RAM-bounded runs (2 workers).

Expected: ×1.05–1.2 on the task (+1.6–6%). Slowest loop.

---

## 6. How we work (process)

- **One task per session or agent**, separate folders (`gen/invsqrt`, `gen/dqi`, `gen/chem`), never sharing files. Each runs its own loop and appends to `log/experiments.csv`.
- **Agentic search loop** (the method from Gu et al., arXiv 2610.22367):
  - generator program → official verifier → feedback;
  - keep a MAP-Elites archive over (T, Q) per task, so we don't lose the low-qubit variants;
  - write down every "DISCOVERY" insight.
  - Abandon a direction after N non-improving tries and try a different architecture.
- **Every improvement is a git commit** with its verifier JSON. Nothing is submitted that hasn't passed the full official verifier locally.
- **Submissions (5/day).** Slot 1: current best combo. Slots 2–4: candidates that differ in one task, which tells us their public-LB effect. Slot 5: held in reserve. Track public LB vs local for each.
- **Finals.**
  - Final A (safe): k=0 DQI, margins on every gate.
  - Final B (aggressive): best expected private score.
  - Both need generators for all 3 tasks.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Private salt flips a pass to a fail (DQI k, MBU branches) | Verify each candidate on ≥ 5 seeds; avoid circuits whose failures cluster on rare branches. |
| Prize disqualification for missing generators | Generator-coverage audit in P0; only finals with full source. |
| Reachable-state-only chemistry tricks ruled invalid | Keep true XOR-oracle semantics in the finals unless the host rules otherwise. |
| Lineage / NOTICE terms | Keep a NOTICE file; credit Syed Asad Ali first for his lineage; Apache-2.0 everywhere. |
| Windows serial verification too slow | WSL2 or Kaggle for full runs; numpy pre-checks locally. |
| Overfitting to the public LB | Trust local full-verifier results; the public LB only uses 50% of tests. |
| Rules: no private sharing outside the team | Share only via public forum or notebooks; keep the repo private until we publish deliberately. |

---

## 8. Reference numbers and tools in this folder

- `tools/score_calc.py`: DQI failure→R table, geo-mean sensitivity, DQI qubit floor.
- `tools/chem_cost_explorer.py`: exact closed-form chemistry cost (reproduces the baseline).
- `tools/chem_sweep.py`: grid over (R, B, C, brot, J) for T·Q (ignores λ_eff and accuracy).
- `kaggle/data/`: competition zip (verifiers, sample submission, paper, Linux kickmix.so).
- `kaggle/starter/starter_kit/`: trainer (JAX), encode/trees, build_circuits, baselines.
- Reference invsqrt: 260 Q, 7,735 Toffolis, 37 rotations, numpy-model error 2^-25.07 (matches the paper).
- Background papers:
  - arXiv 2610.23832 (the competition paper);
  - arXiv 2610.22367 (Gu et al.: BM decoder, trace trick, 128-line 1/√x, pruned QROM, MBU everywhere);
  - kickmix docs in Qualtran `docs/kickmix/`.

---

## 9. Path to 30 (written 2026-11-16, after verifying the frontier on Kaggle)

### 9.1 Frontier verified by us (official verifiers on Kaggle, default salt)
| Task | File | Result |
|---|---|---|
| DQI | `ccc2ab84` | PASS, 890 Q, T = 368,139.5 (static 369,010), 0 + 0 failures, R = 1.19126, cost 3.4738e11, 135 s |
| Chemistry | `f6e12f02` | PASS, 427 Q, T 14,712, λ_eff 46.9216, Λ 283.14, E_gap 3.914898002 (floor + 2e-9), error_1 = error_2 = 19.9999985 mHa, 28 s |
| invsqrt | `f2cdab0b` | P1 script missed the file (fixed). Local stats: 44 Q, 20,735 ops, 1,559 static CCX+CCZ (1,464 dynamic per the author), 4 T, 33 rotations at Clifford levels 56–64 |

**Chemistry cost anatomy** (all of it is now known):
- T = ANGLES 8,050 (55%) + inner prep 4,095 (28%) + Givens 2,250 (15%) + outer prep 317.
- Q = 84 persistent + 40 word qubits (J=5 × brot=8) + 303 temp. The 303 is the inner QROAM temp, forced by the cost model's k-selection (k=4, b2=19).
- The shape (8,143,44) sits exactly on cost-model sweet spots:
  - B=143 is the largest B with ceil((B+1)/16)=9;
  - brot=8 is on the low-temp side of the 8/9-bit workspace cliff.
- Each ANGLE circuit uses exactly the 128-qubit cap: 11 selection + 40 words + 77 temp.
- At this point, **1 qubit ≈ 34 Toffolis** in value.

**invsqrt liveness** (`tools/liveness.py`, exact simulation on 1,024 inputs):
- The peak of 44 live qubits holds for 4,274 of 20,735 steps, in 40 separate windows spread over almost the whole circuit. 43 live qubits holds for another 4,302 steps.
- So **no cheap qubit cut**: removing one needs a scheduling redesign, not a local patch.

**Lessons from the public ledger** (`kaggle/public_ds/negres`, 277 recorded dead ends):
- Chemistry fixed-word refits are within ~0.09% of optimal. 7-bit and 6-bit words failed with every recipe tried.
- The rank floor at 20 mHa is R·C ≥ 290 (the frontier has 352).
- invsqrt local resynthesis is exhausted. DQI flag-dropping and release-recompute gave ≤ 0.13%.
- Check this ledger before starting any idea.

### 9.2 What 30 requires
Product of task ratios ×1.2175 over the frontier (28.0949 → 30.0). Plausible ways:

| Route | Needed | Assessment |
|---|---|---|
| Chemistry alone | ×1.2175 (cost 2.42e8) | **Most plausible.** ANGLES is 55% of T; −35% ANGLES (~2,800 T) plus small Q/λ wins gets there. |
| Chemistry ×1.12 + DQI ×1.05 + invsqrt ×1.035 | product ≈ ×1.217 | Spread; needs all three loops to land. |
| DQI alone | ×1.2175 | Needs ~18% fewer Toffolis at the same Q, or a Q/T re-balance. Hard but highest ceiling (see 9.3 C). |

### 9.3 Workstreams, in order of expected gain per effort

**A. Chemistry angle loaders (primary).** ANGLES = 15 circuits × ~268 ANDs per pass. Each circuit is a Boolean function {0,1}^11 → {0,1}^40, and its cost is roughly the number of AND gates (uncompute by measurement is free). Unexploited or under-exploited freedom:
1. **Don't-care addresses.** Only addresses a < 1,220 are ever simulated, so 828 of the 2,048 selection values are free. Synthesize with don't-cares: ESOP/ANF minimization over the care set only.
2. **Row-to-address gauges.** The verifier pairs one-body rows to eigenvectors by assignment ("row order and signs do not matter"). The b index within each ρ, and ρ itself, can be permuted together with W/H2_I. So we can **choose the address map** to minimize AND count.
3. **Vector sign gauge** (u → −u leaves X unchanged), **tree / edge order** (edges on disjoint orbitals commute), and **batch grouping** of the 75 angles into K batches. All of these change the deltas the loaders compute.
4. **J/K trade.** Total ANGLES scales roughly as K·√(J·brot·2^11) ∝ 1/√J. J=10 gives about −25% ANGLES but +40 Q and only 37 temp qubits per circuit; it needs measurement.
5. **Strict XOR-oracle semantics only** (no reachable-state tricks) until the host rules on thread 755427.

Method: an exact multiplicative-complexity / ESOP synthesizer with don't-cares, wrapped in a search over the gauges. Score with the closed-form part plus the official verifier. Every loader is proven exact on all 1,220 addresses × both directions.

**B. Chemistry factorization (secondary, Kaggle GPU).**
- New U, W at the same or a nearby shape, trained with the loader cost in the objective (words that compress well), then quantized to 8 bits.
- λ_eff ∝ √Λ with E_gap pinned at its floor: each −10% Λ gives −5% λ_eff.
- Use the starter trainer plus Gauss–Newton. Prescreen with `evaluate_words` locally or on Kaggle.

**C. DQI architecture synthesis.**
- Public decoder: 890 Q, ~368k T, with in-place syndrome transforms (52 work qubits).
- Gu et al. reach 440k T at 1,837 Q with:
  - Berlekamp–Massey with a shared update path;
  - a precomputed reciprocal Ψ = B² + zA² that halves the operand degree;
  - the trace trick;
  - a 23-Toffoli scan per position.
- Nobody has published the combination: Gu's Toffoli tricks under the public's qubit frugality. At 890 Q, each 10k fewer T is −2.7% on the task.
- First step: a per-phase Toffoli and liveness profile of `ccc2ab84`, run on Kaggle (58M ops; too big for local RAM).

**D. invsqrt (opportunistic).**
- Only a scheduling redesign can remove a qubit (−2.3% on the task).
- Rotation count (33 at ~7.3 Toffoli-eq each) is the other lever: for example, two coarse-and-fine phase passes sharing rotations.

### 9.4 Submission policy (your goal: the first submission ≥ 30)
- We won't submit a "real" entry until the local official verifiers give ≥ 30.0 on the full three-task zip, with the DQI part checked at ≥ 5 salts.
- **Recommended:** one cheap pipeline-validation submission of the verified frontier zip (expected ≈ 28.09). It proves that our Kaggle packaging matches the grader before the 30+ attempt, and costs 1 of 5 daily slots. It's your call; skipping it risks a wasted slot on a format slip instead.
- After the first ≥ 30 entry, submit only strict improvements; keep a running "best verified" zip.

### 9.5 Compute and parallelism
- All heavy runs happen in private Kaggle kernels via `tools/kpush.py`: 4 CPU / 32 GB, GPU for training.
- Kaggle verifier timings: DQI 135 s, chemistry 28 s. invsqrt is pending P0.
- The three workstreams A/B, C, D touch disjoint files (`gen/chem`, `gen/dqi`, `gen/invsqrt`), so they can run as **separate sessions in parallel** once P0 confirms the harness. Each pushes its own kernels and appends to `log/experiments.csv`.

### 9.6 P0 confirmed on Kaggle (2026-11-16)
All three official verifiers PASS on the sample submission. These are the exact reference costs to use as ratio denominators:

| Task | Reference cost | Kaggle verify time |
|---|---|---|
| invsqrt | 2,076,774.2337 (260 Q, 7,735 Toffolis, 37 rotations, worst error/tol 0.477) | 853 s |
| DQI | 24,439,463,583,672.93 (1,893 Q, 5,725,108.4 mean Toffolis) | 237 s |
| Chemistry | 3,355,599,588.3155 (730 Q, 93,237 Toffolis) | 85 s |

Validation submission ref **58433479**: the public frontier zip, unchanged, submitted from kernel `<kaggle-user>/qcc-v0-validation` (this is a notebook-submission competition; a direct file upload returns HTTP 400). Expected score ≈ 28.0949.

### 9.7 Revision after measuring the loaders (same evening) — chemistry priorities flip
- **Loaders are near their plateau.** `tools/chem_loader_stats.py` on the frontier:
  - each of the 15 circuits costs 263–294 static Toffolis;
  - their target functions look random under zero-fill (ANF ≈ 940–1,020 terms of 2,048, degree 11).
  - The cheapness comes from codes and structure the community has already built: low-bit codes on each word and on the XOR of five words, padding rows used as don't-cares (manwithacat), banked QROM, matroid-parity pairing, and π/negation gauges (marculera notebook credits).
  - **Workstream A is demoted.** We keep it as an opportunistic pass only.
- **New primary chemistry lever: the factorization shape (workstream B).** The cost model plus a loader estimate of ~6.6 Toffolis per address (both passes) prices these shapes against (8,143,44):

| Shape (R,B,C) | Addresses A | Est. T | Q | Est. gain | Note |
|---|---|---|---|---|---|
| (2,200,180) | 476 | ~11.4k | 426 | ×1.29 | R·C = 360 |
| (3,180,110) | 616 | ~11.6k | 426 | ×1.28 | R·C = 330 |
| (6,129,50) | 850 | ~11.7k | 427 | ×1.25 | R·C = 300, close to the rank floor 290 |
| (4,160,90) | 716 | ~12.2k | 426 | ×1.21 | R·C = 360 |
| (2,250,160) | 576 | ~12.5k | 426 | ×1.18 | |
| R=1 shapes, (9,124,39), (4,255,128), (10,129,60) | | | | ×0.7–0.9 | worse under the cost model |

- **The catch:** each shape must reach 20 mHa after 8-bit quantization. Notes:
  - Public R=1 runs stalled at 20.5–22.4 mHa.
  - (9,124,39) reached 18.4 mHa.
  - The rank floor (Eckart–Young) is R·C ≥ 290 for 20 mHa and ≥ 330 for 10 mHa; it is necessary, not sufficient.
- **Plan B1 (Kaggle GPU):** train each candidate shape with the starter trainer plus Gauss–Newton.
  - Seed from the frontier table, truncated or extended in B and C.
  - Target ≤ 17 mHa continuous, then quantize to 8 bits with joint weight refit (the frontier's recipe), then build loaders and price exactly.
  - If one shape reaches PASS at a similar λ_eff, chemistry gains ~×1.2–1.3, which is about +6–9% on the overall score.
  - This is the single biggest identified lever, so it gets first GPU time.
- DQI (workstream C) stays second; it needs a per-phase profile first.
- **Validation result:** submission 58433479 scored **28.09486** on the public LB, exactly the frontier's score. The packaging and notebook-submission path work.

### 9.8 B1 result (2026-11-16, GPU T4, starter objective, 30k epochs, ~5.5 min/shape)
| Shape | ε_fro (continuous) | λ_eff | Λ |
|---|---|---|---|
| (8,143,44) control, 2k epochs | 15.9 mHa | 48.64 | 287 |
| (6,129,50) | 46.3 | 65.1 | 520 |
| (4,160,90) | 59.1 | 71.0 | 605 |
| (3,180,110) | 82.5 | 73.6 | 693 |
| (2,200,180) | 214.5 | 79.8 | 807 |

- **The cheap shapes are far off.** Their errors are still falling but are 2–10× the limit, and λ_eff rises 40–70%. That would cancel their 1.2–1.3× cost-model gain even if the error reached 20 mHa.
- **The shape lever is much weaker than the cost model alone suggests.**
- **B2 (running):**
  - (7,143,50) is the near-frontier step: ×1.07 under the cost model, with 92% of the frontier's parameters, seeded from the frontier.
  - (6,129,50) continues from its own B1 result.
  - Both run 300k epochs.
  - Decision rule: pursue only if ε_fro ≤ ~16 mHa with λ_eff ≤ ~47.

### 9.9 Day 2 (2026-11-17)
- **B2 closed the shape route.** (7,143,50) reached ε_fro 20.9 mHa after 300k epochs, but λ_eff was 67.7 (Λ 530) against the frontier's 46.9. (6,129,50) reached 39.3 mHa. The starter objective cannot keep Λ low; the frontier's Λ = 283 comes from long specialized refit chains. Shape changes are deprioritized.
- **Public frontier is now ≈ 28.15.**
  - invsqrt `0af21cb8`: 74,797.37 (maoxin123).
  - DQI `61c8348d`: ≈ 3.4641e11 on a clean draw (grantprater; merges boristown patches and Balram Srinivasan's GF24 closings).
  - Chemistry: 4,022-Toffoli loaders plus a refit, 294,643,137.73 (maoxin123).
  - Gains are about 0.05/day from patch merging, so 30 needs a structural change.
- **C1 running:** a phase profile of the DQI decoder (Toffolis and live qubits over the 58M instructions).
- **C1 result: DQI phase profile of `ccc2ab84`** (256 real shots, all phases correct, syndromes restored; mean T 368,138 per shot):
  - Instructions 0–17.4M: zero Toffolis. Clifford and classical prep, with 2.06M classically conditioned ops on the M record. Free under the metric.
  - Forward extraction (17.4M–30.2M): ≈ 153k T. Live qubits peak at 888–890.
  - Scan (30.2M–36.8M, 192k conditioned blocks): ≈ 52k T, i.e. **≈ 12.8 Toffolis per position**. That already beats Gu et al.'s 23 per position.
  - Reverse extraction plus cleanup (36.8M–53.9M): ≈ 163k T.
  - **The extraction is ≈ 86% of T.**
  - Back-of-envelope comparison with Berlekamp–Massey: ~2,400 field multiplies per direction at ~55 Toffolis each ≈ 130k per direction, similar to the public's figure. BM keeps σ and B alongside the syndromes, though, so it needs ~1,700+ qubits; Q² makes that 3.6× worse.
  - The public's in-place continued-fraction design trades ~2× Toffolis for the 890-qubit width, and that trade is near the Q²·T optimum.
  - **A DQI breakthrough needs an in-place key-equation solver with fewer field multiplies.** That is open research, not a patch.
- **Local invsqrt loop is now fast and exact.**
  - `tools/invsqrt_fastcheck.py` sweeps all 3,133,452 x in about 2.5 minutes on this PC. It reproduces the official cost (75,006.8 vs 75,006.38 for `f2cdab0b`).
  - `tools/invsqrt_angle_refit.py` refits the 33 fine Z_POW angles. It runs two sweeps (base, plus power-of-two tags that decode exactly which rotations fire per x), then a cutting-plane LP that maximizes the minimum margin. That margin sets the rotation precision eps and therefore the T price per rotation.
  - Running now on `0af21cb8` (frontier invsqrt, 74,797.37).
- **Angle refit result: no gain.** The LP optimum equals the current angles (min margin 1.7233e-12 turns, eps 1.641e-13, 29.08 T per rotation). The binding margin is at x=231 and comes from arithmetic error that no fine rotation affects. The published angle refit (ozertuu) is already optimal. Closed.

### 9.10 Status at end of day 2
Every cheap lever we measured is at its optimum:
- chemistry shape (B1/B2);
- chemistry loaders (public codes);
- invsqrt qubits (liveness), invsqrt angles (LP);
- DQI scan (12.8 Toffolis per position); the DQI extraction is near the Q²·T optimum for in-place designs.

Remaining routes to 30 are research bets:
- R1: an in-place key-equation solver for DQI with fewer GF(2^12) multiplies (extraction = 316k of 368k T).
- R2: an invsqrt redesign with ≤ 40 qubits (each qubit ≈ 2.3%).
- R3: a chemistry factorization with lower Λ at the frontier shape (Λ 283 → ~230 gives ≈ 10% lower λ_eff).

Each is weeks of work with uncertain payoff.
- **invsqrt redundancy search (`tools/invsqrt_dependency.py`, 4,096 inputs): closed.** At 3,944 of 4,021 peak steps the 44 live qubits are GF(2)-linearly independent (constants included), so no qubit can be rebuilt with CNOTs alone. With 40 peak windows, any non-linear rebuild costs more than the ~39 Toffoli-equivalents a qubit is worth. **Fewer qubits needs a new architecture, not a patch.**
- Scale check: the 30.5 leaders are ×1.28 over the public frontier. A 34-qubit invsqrt at today's C would explain all of it, which hints that the leaders may hold such a design. Building one is the R2 research bet.
- **invsqrt role trace (`tools/invsqrt_roles.py`) of `0af21cb8`:**
  - Qubits 0–21 (the x register) are normalized in place and then reused for the arithmetic.
  - Qubits 22–24 hold the low bits of the shift s.
  - **Qubit 25 is the top shift bit s3, which is 1 only for x < 64 (63 of 3,133,452 inputs), yet it is live from step 234 to step 20,748** (82 instructions: copy, 12 CCX in the unnormalize stage, CX/CZ fix-ups, uncompute).
  - **43-qubit lead:** keep s3 for the 63 small inputs in positions that are provably zero for them (e.g. the ≥16 trailing zeros of x_norm while they are still unused), or special-case x < 64. Value: −2.3% invsqrt cost (≈ +0.76% overall). It needs netlist surgery verified on all x with `invsqrt_fastcheck.py`.
- **43-qubit lead: closed.**
  - Host search over all steps: at some steps of s3's lifetime, no qubit is zero on all 63 small inputs. The best candidate, q24 (the s2 bit, which is 0 whenever s ≥ 8), covers 98.8%.
  - At some of q25's 68 control uses, no qubit is idle on all inputs.
  - Root cause: s ∈ {0..10} needs log2(11) ≈ 3.46 bits. The normalized-x trailing zeros can't hold the spare state because those bits are read as δ digits, and a 2^-21 relative error breaks the 2^-24 tolerance.
  - Removing a qubit needs a different encoding of the shift, i.e. a new architecture.

### 9.11 R2 started: from-scratch invsqrt design model (`gen/invsqrt/design_model.py`)
Exact integer model over all 3,133,452 x: normalize, table by top bits, Horner on the remaining bits, fixed-point truncation.

| Layout (table bits, coordinate bits, degree, coefficient bits) | Segments | Worst err/tol |
|---|---|---|
| (6, 16, 3, 28) | 48 | **0.971 (passes)** |
| (6, 16, 3, 27) | 48 | 1.069 |
| (8, 14, 2, 27) | 192 | 1.19 |
| (7, 15, 2, 28) | 96 | 4.9 |
| (5, 17, 3, 28) | 24 | 7.0 |

Reading:
- The precision floor is about 28-bit coefficients and accumulator for a plain Horner layout.
- That layout needs x (22) + s (4) + accumulator (~28) + multiplier workspace, which is well above 44.
- The 44-qubit public line beats this by transforming x in place into the result. Its role trace shows y bits inside the x register. It then restores x by a "division clean-up" (recomputing x from y).
- **A sub-44 design has to keep that in-place transform and make it narrower.** The next step is to model in-place x → y transforms and their inverse clean-up cost.

### 9.12 Day 2 evening: track the best public package
- The best public LB score is 28.15096: Atakan Aldemir, "QCC: Shared Products and Controlled Refits" (all-49 GF24 DQI closings plus 4 product-retention pairs; maoxin123's 4,022-Toffoli chemistry loaders with Atakan's refit weights; maoxin123's 44-qubit invsqrt).
- `kaggle_nb/v2_best_public` re-verifies that package with the official verifiers (DQI at the default salt plus salts s1 and s2) before we submit it.
- `kaggle_nb/v1_composite` (our own pick of files) is superseded. Lesson: an author's label is not always the file's SHA-256 (maoxin123's "56ef2562" chemistry is npz `a73b25d2`).
- Merger outreach drafts in `outreach/` are shelved: the user competes solo (2026-11-17).
- **Best-public package re-verified (`kaggle_nb/v2_best_public`):**
  - invsqrt 74,797.37; DQI 3.46410e11 (0 failures at the default salt and s1, 1 at s2); chemistry 294,642,299.76.
  - **GM 28.15096.** Submitted as ref 58464611.
  - The v1 composite (DQI `61c8348d`) lost 1 shot at the default salt (GM 28.108) and is not used.
- **invsqrt Toffoli split by stage (`0af21cb8`, static):** normalize 165, lookup/Horner 188, pre-phase 285, phase 53, then about 640 for post-phase plus clean-up (the division that rebuilds x from y) and 104 for denormalize. Forward and clean-up are roughly even, so neither half has a cheap shortcut. Erasing y by measurement still needs y recomputed for its phase fix-up.

### 9.13 Forum scan 2026-11-17 (solo from here on)
- **User decision: solo, no merger.** Outreach shelved.
- RezOps briefing (759655): public best-per-task GM 28.107; leader 10% above the public line, and nobody knows which task holds the gap. DQI 66b4f18b (Balram, −20 Toffolis × 41 closing gadgets) has a 48-draw expected cost of 3.4788e11; our package's DQI fails 1 shot on salt s2, so the private draw may reorder the plateau.
- David Pilný's SAT certificates: no 2–4-gate local rewrite wins on the 44q invsqrt (558 windows). Gains must come from phase tricks, MBU or workspace, not peephole rewriting.
- Chih-Kai Wang (759048): a rank-173 decomposition of (H·D, H_hi·D²) over GF(2^24) (older 392k family, −546 Toffolis). The newest family has 1,376 independent GF(2^24) multiplies. **Lever D1:** measure the Toffolis per multiply in our DQI; if one is above the ~99 Chudnovsky-on-F8 bilinear rank (14·6 + 15), swap in a lower-rank decomposition, which is an exact, mechanical win.
- RezOps negative row: chemistry with 7 groups rounds to 22–24 mHa (fails 20). This confirms our B2 result.
- **58464611 scored 28.15096** (exactly the local verified GM). Our best confirmed score is now 28.15096.

### 9.14 Route to 30 via prize-forced publication (2026-11-17)
- **Prize rules force the leaders to go public:**
  - **Milestone Prize:** highest public LB at **Nov 20 23:59 UTC**; the winner must publish a notebook (deadline unclear, thread 758095 is unanswered).
  - **Longest Leader Prize:** the notebooks and datasets must be public **by Jan 4, 2027 23:59 UTC**.
  - So a ~31-level package should be public by Jan 4 at the latest, possibly around Nov 21. Final deadline Jan 20.
- **Plan:**
  - (1) `tools/frontier_watch.py` runs in the background and flags any new or updated top-3 public notebook.
  - (2) When one appears, verify with `kaggle_nb/v2_best_public` (pin by hash, all three verifiers, DQI at 3 salts), credit the lineage, and submit.
  - (3) In the meantime, build our own improvements: our own invsqrt generator (needed for prize-rule compliance anyway, plus a narrower or cheaper design), so they stack on whatever goes public.
- Kaggle's notebook API sorts by score but doesn't return it; the watcher tracks the ref and version of the top 3.
- DQI builder source: not in the current version of the syed3000 dataset, and old versions can't be downloaded by CLI. Lever D1 is parked (expected ≤1–2%).
- **Caveat on the publication route:** a top team can skip the $10k Milestone and Longest Leader prizes to protect its final rank, so publication is likely but not guaranteed.
- **Public invsqrt work is netlist surgery, not source:** the atlas dataset (asteriospanoras) and maoxin123 ship only .kmb/.kmx plus recipes (splices, dead-write fixes, ~0.05–0.2% each). No 44q generator source is public.
- **s3-qubit (q25) removal, analysed:** s3's information can't hop between host qubits, because any CX hop corrupts normal inputs (the hosts carry live data for x ≥ 64). Freeing q25 needs s capped at 7 for x < 64, with the special cases handled in the unused table rows 0–15 and a c/8-turn T/S/Z fix-up. That needs a decompiled understanding of the 0af21cb8 arithmetic. Worth ≈ +0.75% GM.
- **Own-gain search, 2026-11-17 (closed):**
  - Dead-write scan of 0af21cb8 (`tools/dead_write_scan.py`): 0 motif-A sites. The 8 motif-B sites are Toffoli → rotation → HMR controlled rotations, which are already optimal.
  - The public ledger (`kaggle/public_ds/negres/ledger.csv`) already kills, among others:
    - NR-0037: split-domain small-x handling.
    - NR-0198 / NR-0235: phase-gradient, Newton, CORDIC, bipartite and Walsh architectures.
    - NR-0275: 43 qubits via q25.
    - NR-0261/0262: divider MBU.
    - NR-0265: DQI design alternatives.
    - NR-0204/0205: window resynthesis.
  - Conclusion: dozens of agent-assisted teams have mined the public invsqrt and DQI lines. A solo +6.6% from public parts is unlikely. **Primary route to 30 = leader publication (watcher).** Our own work is worth only if a genuinely new idea appears.

### 9.15 Top-50 / top-10 push (2026-11-17, user: "top 50 is fine, top 10 possible")
- **LB (3,255 teams), score needed for each rank:** #1 31.014 · #10 28.926 · #16 28.692 · #20 28.564 · #30 28.375 · #40 28.257 · **#50 28.2006** · #75 28.1565 · #100 28.15103.
  - We are **#107 at 28.15096**, tied with 64 teams; ties go to the earlier submission.
  - Top 50 needs **+0.18% GM** (= −0.53% cost on one task). Even 1 invsqrt Toffoli (+0.02% GM) passes ~30 teams.
- **Clue:** Rithvik Redrouthu reached 30.13 in 4 entries, which suggests one big design (likely invsqrt with fewer qubits; one invsqrt qubit = +0.75% GM).
- **New asset: full builder sources (Apache-2.0, Syed Asad Ali; credit him first)**, from dataset `syed3000/vqcc-own-circuit-portfolio` **v12**.
  - The CLI can't fetch old versions, but the REST API can: `GET https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>?datasetVersionNumber=N` with `Authorization: Bearer <token>`.
  - Local copy: `kaggle/public_ds/syed_v12/v12/src_{invsqrt,dqi,dfthcsos}` (gitignored).
  - **invsqrt 46q generator (321aff83, 80,162):** frozen base `work/c10`, 13 patches, mechanical passes and an exact full-domain prover. Needs CPython 3.13 + numpy and the Linux kickmix.so (build on Kaggle). Component prices: normalization ≈95, root ≈192, product ≈228, coefficient ≈118, **divider ≈578** (biggest), 33 rotations.
  - **DQI generator** (431 files) and **chemistry compiler** (85 files) are included too.
- **Ledger facts (DQI):**
  - 81% of DQI Toffolis are field products: 1,376 × GF(2^24) at rank 99 and 2,853 × GF(2^12) at rank 42.
  - One unit of GF(2^12) rank would be worth +0.26% GM. Flip-graph search stalled at 42, and the literature check found nothing below 42 / 99.
  - The 12-Toffoli GF(4)⊕GF(16) block (1,183 sites) is worth +0.10% GM per Toffoli saved; nobody has found 11.
- **Chemistry:** symmetry shifts are locked by the verifier (NR-0135), Givens-tree changes give ~2% damage (NR-0127), and B ≥ 129 is the only Q cliff in the cost model (checked 100–420).
- **D1, DQI dead-write scan (Kaggle):** our DQI (575b043f, 57.5M instructions) has **0** dead-write sites. Closed.
- **Kaggle image runs CPython 3.13**, so Syed's invsqrt build needs no internet or uv on Kaggle.
- **Private dataset `govindmulchandani/syed-v12-builder-src`** (Apache-2.0 mirror of the v12 sources).
- **Blocked:** running Syed's build locally and on Kaggle was denied by the auto-mode safety check (third-party code execution). Needs the user's go-ahead (kernel `kaggle_nb/b3_syed_invsqrt_build`, not pushed).
- **B3 (Kaggle):** Syed's invsqrt generator rebuilds **byte-identically** (321aff83). Exact proof PASS; official PASS 80,162.73 at Q46. Build takes 144 s on Kaggle's CPython 3.13 (offline).
- **B4:** ran Syed's generic finishing passes (affine audit/rewrite, held-product hold pass, Z_POW merge) on our frontier 0af21cb8. **Zero findings, cost unchanged** (74,797.37). Mechanical passes are exhausted on the frontier.
- **Generator architecture (from source):**
  - normalize → in-place integer root n = v² + h (v 12 bits, h 10 bits) → body.
  - Body: two coefficient tables on v's top 7 bits → seed → product P15 → **divider** (applies the phase digit-serially; 270 forward + 308 cleanup) → second table → inverse product.
  - About 6 body stages sit at 44–45 live qubits (Q46 build).
- **Watcher:** the "Apex Grandmaster" #1 notebook is a repackage of our exact 3 files (28.15096), so it was a false alarm.
- **Divider source (`native/candidate01/divider.py`):** every bit position uses a hand-fitted quadratic XOR plan (custom carry polynomials, HMR erasure with exact phase repayment). Beating it with radix-4 SRT would mean re-deriving this gadget level at radix 4. Multi-day work, success uncertain. **Decision point with the user.**
- **Divider redesign (radix-4), killed on analysis, 2026-11-17:**
  - The divider is radix-2 non-restoring: 24 rows (j = 12..35), each digit ±1, one 13-bit framed add per row (≈11.25 Toffolis forward, 12.8 cleanup). The ±1 framing is CNOT-only.
  - Fusing two rows needs the digit set {±1, ±3} (a stored 3·D: +14 qubits) or a zero digit (mask/mux of D: more Toffolis than one adder). The late rows already narrow (k offset).
  - About one adder per quotient bit is the floor for this algorithm. Gains would need a different reciprocal method, and the alternatives (table + product, Newton) cost more.
- **Chemistry source review:** Syed's v12 loaders total 4,074/pass, beaten by maoxin's 4,022 (in our pick). The compiler uses C++ min-rank and planner tooling, and the "reachable-state" loader contract (a loader only has to be correct from batch k−1's words; thread 755427 is unanswered). The research scripts aren't runnable standalone.
- **Decision (2026-11-17), offered to the user:**
  - A: small sure gains (chemistry weight refit, single-Toffoli hunts) → rank ~80–100.
  - B: long-shot redesign (narrower DQI layout, best case 0.949× = +1.7% GM per NR-0234; or a 43q invsqrt, +0.75%), weeks with low odds.
  - C: watcher only.
- **43q via s-cap (3 flags), killed with the exact model (`native/candidate01/whole_model.py`):**
  - Phase = p·2^s mod 2^37 with p ∈ [2^26, 2^27), so exponent s yields phases in [2^(s−11), 2^(s−10)). For x < 64 the phase 1/√x ≥ 1/8 needs s ≥ 8.
  - x = 16 (s = 8) and x = 2^20 (s = 0) have identical body states differing only in s3, so s3 is pure information and can't be recomputed.
  - Marker encodings into unreachable n codes perturb the phase by ~1e-7, above the ~7.5e-9 tolerance.
  - Matches the NR-0071/0275 dead ends.
- **Model summary:**
  - s = (21 − lead)//2; n = x·4^s; Q = isqrt(16n) ∈ [4096, 8191]; v = Q − 4096; d = 2Q + 1; h = (64n − d²)//64.
  - C = table(v), seed U0(C, rootclass), U = U0 − Σ±C-terms over q = h + 512 (product stage).
  - Non-restoring division of U by d, rows j = 9..35, gives the phase bits; word = cyclic rotation by s.

### 9.16 Parallel research + rented compute (2026-11-17)
- The user approved 4 parallel sessions: `gf12-rank` (GF(2^12)/GF(2^24) multiply rank), `dqi-gadgets` (DQI repeated-gadget resynthesis), `chem-loaders` (ANGLE_k loader resynthesis), `invsqrt-divider` (divider/endpoint cuts). Each works in `research/<lever>/` and `kaggle_nb/<prefix>_*`; the coordinator verifies and submits.
- **Compute:** the user is renting a server. Hetzner dedicated (monthly cap, hourly billing, excl. VAT, seen 2026-11-17):
  - AX102: 16-core Ryzen 9 7950X3D / 128 GB, €259/mo (€0.415/h), €129 setup.
  - AX162: 48-core EPYC 9454P / 128 GB+, €614/mo (€0.984/h), €304 setup.
- SSH key for the server: `~/.ssh/qcc_compute` (public half to be added at order time).
- LB at 2026-11-17 ~12:40 UTC: #1 31.014, #10 28.926, #50 28.206, #100 28.15116. We're at 28.15096: 116 above, 100 tied.
- **v3 composite (2026-11-17):** maoxin123 'QCC 28.17' has invsqrt 563d0583 (74,560.91). With our DQI 575b043f + chemistry 703843d3 it projects to ~28.181. Kernel kaggle_nb/v3_composite pushed; once it PASSes, submit with -k govindmulchandani/qcc-v3-composite -f submission.zip.
- **Submitted v3 = ref 58495180** (2026-11-18): invsqrt 563d0583 (official PASS 74,560.91, 1,454.00 Toffolis, 44q) + DQI 575b043f + chemistry 703843d3. Local GM **28.18069**.
- **R1 chemistry refit, closed:** the torch model matches the official metrics to ~1e-6. Adam penalty: no feasible gain. Active-set projected gradient: "no feasible descent" at the first step. Our weights are already first-order constrained-optimal, so a fixed-word weight refit gives nothing.
- The 4 research sessions hit the 5-hour usage cap on 2026-11-17 and were resumed on 2026-11-18. dqi-gadgets census so far: GF(2^12) multiplier ≈ 46% of T, scan sharing killed, 172-T/185-T gadgets pending. The invsqrt session's baseline was moved to 563d0583.
- **New public files (2026-11-18):** invsqrt **864f81cd** (maoxin123 QCC 28.18; 74,490.95) and DQI **5f901c55** (boristown gap-3 residue repair; 3.46404204e11 at the default salt, 0 failures; 2 failures on his "notebook-verification" salt). Best per task = boristown's package (+ chemistry 703843d3), projected GM ≈ 28.192. Kernel `kaggle_nb/v4_composite` re-verifies it (invsqrt + DQI at default/s1/s2).
- **Parallel sessions, round 1 results (2026-11-18), all null:**
  - chem-loaders: all 15 loaders stay at 4,022 ANDs/pass.
  - gf12-rank: no rank-41 GF(2^12) or sub-99 GF(2^24); the 30→29 lead failed (non-bilinear blocks).
  - dqi-gadgets: the top gadget families are field products/inverses already at best-known rank.
  - invsqrt-divider: 5 ideas, Δ ≤ 0; one lead (−1 Toffoli in cleanup rows j = 15/16) sent back for testing on 864f81cd.
  - Notes: `research/*/README.md` / `NOTES.md`.
- **v3 (58495180) scored 28.18070.**
- **v4 verified:** invsqrt 864f81cd 74,490.95; DQI 5f901c55 PASS with 0 failures at default/s1/s2 (3.46404204e11 / 3.46403695e11 / 3.46403824e11); chemistry 703843d3. Local GM **28.18966**. Submitted; scored **28.18964**.
- **Chemistry attack, scoping (2026-11-18):** the public dead-end ledger has 145 chemistry rows (vs invsqrt 49, DQI 70). The J trade is already killed (NR-0128: core rank is linear in table bits, so +8 Q per word cancels it). Lower λ_eff by any refit or new factorization: killed (NR-0114/0250, B1/B2, R1). brot 7, other shapes, C pruning, 6/7-group tables: killed. Remaining measurable chemistry slack is loader products only: k0/k1 212 vs a floor of ~188, k2/k3 195 vs 189. Worth about 0.2–1% overall at most.
  - Sessions: chem-loaders → a global (ILP/annealing) product planner (k2/k3 ≤ 190, k0 ≤ 209). invsqrt-divider → pivot to a ≤ 43-qubit invsqrt design study (1 qubit ≈ 0.76% overall; it's the least-mined task and the most plausible 30 route).
- **Chemistry redesign, verifier-level review (2026-11-18), all closed:**
  - Angles decode as θ = π·word/2^(brot−1), a full 2π per edge. Only the global sign is redundant (a θ+π flip changes the downstream subtree's sign), so there is no hidden precision bit.
  - The inner-QROAM k2 is the largest k with temp < brot·75 = 600. The B ≤ 128 "cliff" goes the wrong way (temp 303 → 575, T·Q 0.70×); B 129–143 sits on the only good plateau. (8,129,44) would be 1.055× but is error-killed (NR-0225/0269).
  - S, β1 and E_upper are overwritten by the verifier, so λ_eff is pinned at ≈46.79–46.91 (NR-0250 identity), and chemistry gains come only from T·Q.
  - The single binding constraint is 8-bit rounding noise (≈14.6 mHa in quadrature). Every cheaper shape (7 groups at 0.934×, B=129 at 1.055×, C=43) dies on it. Reopen chemistry only with an integeriser whose rounding noise is below ~6–9 mHa. Until then the only live chemistry work is chem-loaders' global planner (≤1% overall).
- **invsqrt qubit round 1 (invsqrt-divider, 2026-11-18): null.** All four 43q ideas are priced in research/invsqrt_divider/NOTES.md (shift encoding, early remainder retire, folding into dead x bits, paying Toffolis per peak window: ≥26 and likely 50+ against a 39.4 budget). The current line is at its width limit.
- **invsqrt round 2 (architecture):** after the divider the state holds ~25 quotient signs plus a ~14-bit remainder (~39 bits) encoding only n + s (~25.5 bits). That redundancy exists only because the phase is applied at the end, which s-routing forces.
  - Session is pricing three routes: (A) absolute-frame digit recurrence with fixed per-digit angles, applied and freed immediately; (B) normalisation kept, flag-controlled angles only for the top digits; (C) block pebbling of the digits.
  - Budgets: Q40 → C ≤ 1,862; Q38 → C ≤ 1,960; Q36 → C ≤ 2,069.
  - Paper arXiv:2610.23832 (the benchmark source) has only the reference (260 Q, 2.08e6) and no agent-found numbers.
- **v4 scored 28.18964.** chem-loaders round 2 (global annealing planner): null; all 15 loaders stay at 4,022/pass.
- **5-frame brainstorm (2026-11-18):** speedrunner, remove-assumption, hardware, competitor and biology frames gave 30 ideas.
  - Already in the ledger: every chemistry idea (joint/noise-shaped rounding, gauge/row order, Gray-delta ROM, compressible word choice); DQI register sharing (in-place BM 1.44×), normal basis (rank is basis-invariant) and width squeezes.
  - Traps:
    - Z-measuring x-dependent bits gives an invalid circuit (it collapses superposition) and won't be submitted.
    - HMR polarity tricks fail because outcomes are uniform.
    - DQI truncation fails because R(ε) is steep.
  - Survivors, all invsqrt and sent to invsqrt-divider:
    1. Streaming digits. Crux: σ_j isn't a function of r_{j+1}, so U or x must stay live.
    2. SRT redundant digits with a carry-save remainder.
    3. Joint (s, top bits) seed table.
    4. Direct digit recurrence for 1/√x.
- **invsqrt architecture study (research/invsqrt_divider/ARCH.md): null.** Every variant is ≥30% worse or saves no qubit:
  - A/A′ absolute-frame digits: +35–40%.
  - B/B′ class-controlled rotations: +30–130%.
  - C pebbling: ≥0%.
  - Brainstorm 1–4: streaming 71q, SRT+CSA ~109q, joint seed table +2,000 T, Ercegovac–Lang ~110q.
  - Key facts:
    - The product stage and the divider stage each fill 44 on their own.
    - A digit is globally but not locally recoverable, so releasing it means re-running the division.
    - Class s needs every digit at weight 2^(i+s−37), so no digit angle is fixed.
  - A ≤40q design must never hold root + remainder + digits (~51 bits encoding 21.6) at once. No such pipeline is known.
- **Discord intel check (2026-11-18):** the user pasted the Discord channel (RezOps briefings 3–6 Nov, plus Max Freedom Pollard's claim that boristown shipped a C=43 "decoy" and Lera has passing C=40/42 bundles).
  - Checked against manwithacat's manifest of 565 public circuits (kaggle/public_ds/mwc_manifest.csv).
  - The C=39/40/42/43 tables are all 9-group (9,143,C): 312–333M, worse than our (8,143,44) 294.64M because 9 groups add 143 loader addresses.
  - No 8-group C<44 table is public, and boristown's package ships our same 703843d3.
  - Our v4 files beat every file in the manifest (invsqrt 864f81cd < 563d0583; DQI 5f901c55 < 575b043f).
  - Nothing actionable: the channel is the public crowd, and the leaders are silent.
- **v5 submitted (ref 58504681):** v4 plus chemistry 4d63be3d (Leo Tang's weights on maoxin123's loaders; official PASS 294,642,170.57, −129 vs 703843d3). Local GM 28.189660; scored **28.18965**. Kernel `kaggle_nb/v5_composite`.
  - Lesson: `kpush.py` takes inputs from `--kernel` flags, not from metadata `kernel_sources`. Kaggle auto-extracts zips, so scan loose files too.
- **Private-salt risk (Alex Chilton's notebook):** the private LB reruns the verifiers with an unknown salt, and one failed task zeroes the whole submission. DQI failures only raise cost (5f901c55 had 2 failed shots at the "notebook-verification" salt, +0.85%).
  - Kernel `kaggle_nb/r2_salts` checks invsqrt 864f81cd at 5 salts and DQI 5f901c55 at 4. Pick finals by robustness, not only by public score.
- DQI deep research session `dqi-redesign` started (it hit the usage cap once; restarted).
- **DQI deep redesign (dqi-redesign, research/dqi_redesign/NOTES.md): null.**
  - The frontier is a Thiele continued-fraction interpolation at 35 quadratic nodes (extraction ~153k T), synthesis already in evaluation form, a half-degree Euclid convert (~116k), and an additive-FFT scan (44k).
  - Half-GCD blocks: best ×1.34 (width kills it, since it needs ≤ 24 extra qubits).
  - Quartic nodes need a GF(2^48) multiplier at ≤ 136 T (none known).
  - Shared operands and constants: none to exploit.
  - Asymmetric or multiplicative reverse: +420 q.
  - A weight-conditioned path needs input-dependent measurement, so it's invalid.
- **Salt robustness (r2_salts): all PASS.**
  - invsqrt 864f81cd at 5 salts (private-a/b/c, q7, zz-final): 74,490.97–74,491.03.
  - DQI 5f901c55 at 4 salts: 0 failures each, 3.464039–3.464043e11. With the earlier default/s1/s2 that makes 7 clean salts.
  - The 2 failures boristown saw are specific to the "notebook-verification" salt.
  - v5 (28.18965) is a safe final pick; chemistry is deterministic.
- **Leader-intel sweep (2026-11-18), nothing usable:**
  - Top-12 team members' Kaggle notebooks: none for this competition. The top teams are Kaggle GMs (Z³C¹oherence; Entangled Team = James Day / Geremie Yeo / Tony Li / jackvd; aadi_stonks solo; Rithvik Redrouthu 30.13 in 4 submissions).
  - GitHub search for "kickmix" / the competition name: ~30 repos, all public-crowd tooling.
  - arXiv:2610.22367 (Gu, Kemeny, Yelin, Lukin; LLM-discovered circuits): its 1/√u circuit is 142q / 1,140 T, its RS decoder ~1.5e12 Q²T, and it has no DF-THC; all worse than our frontier.
  - DQI R = 1.19 is the Clopper–Pearson floor at 0 failures (δ = 2^-128, N = 102,400), fixed by the verifier.
  - Realistic 30+ routes are now the forced reveals: the Nov 20 milestone notebook and the Longest-Leader notebooks (public by Jan 4, final deadline Jan 20). The watcher covers both.

### 9.17 v6 push toward 28.5 (2026-11-18, user goal: 28.5 without relying on the watcher)
- **Target arithmetic:** 28.5 / 28.18965 = 1.0110 GM, so the product of task ratios must rise ×1.0334 (one task −3.2%, or a mix).
- **Small-B chemistry (new lever, closed):** the official cost model prices shapes with B ≤ 23 and many groups (e.g. (52,22,7), (72,16,5)) at 0.66–0.76× our T·Q, because the inner-QROAM temp falls from 303 to ~250 qubits and inner Toffolis drop ~3×. Capacity kills it: the error_2 target's leading eigen-components are full-rank 76×76 matrices. Truncating each to rank r leaves 8.30 Ha at r=22 and 3.11 Ha at r=60, but 7.4 mHa at r=76 (K=352; `research/chem_smallB/probe.py`). Each X^{ρc} has rank ≤ B, so B < 76 is impossible; the B ≥ 129 plateau is forced. (Kaggle GPU fit `kaggle_nb/c2_smallB_fit` as empirical check.)
- **Chemistry C reduction at 8 groups:** each copy removed is −0.53% chemistry T·Q (+0.18% GM); C=40 gives +0.72%. It needs new words (fixed-word prunes fail, NR-0043), and new words lose the loader structure (+598 T, NR-0134), so it isn't worth it.
- **invsqrt phase margin (864f81cd, `research/invsqrt_margin/`):** 50 inputs share the minimum margin 2.107e-12 turns at large x (err/tol 0.94–0.98); the LP has already equalised them. 1,026 inputs are below 8e-12. A perfect (zero-error) arithmetic would give at most 3.9 more bits = 18 Toffoli-eq (−1.07% invsqrt, +0.36% GM), and any extra precision costs Toffolis. Minor lever.
- **Rotation count:** the 33 Z_POW angles are 2^-3 … 2^-35 half-turns, one each (a binary phase word), so Hamming-weight phasing doesn't apply.
- **C-pruning at fixed words (`kaggle_nb/c3_chem_cprune`, Kaggle CPU): closed.** Starting from v5 chemistry (4d63be3d), I gauge-rotated each group's 44 weight columns to the principal axes of their X Gram (smallest eigenvalues 0.002–0.004 against largest 2.5–4.0) and dropped the weakest per group (C=43). Then I refit w and the offsets with penalty L-BFGS, raising μ from 1e3 to 1e11. error_2 floors at **21.23 mHa** (gate 20) with error_1 and E_gap held, so C=43 is infeasible with the current words; lower C is worse. Like NR-0043, the shipped table has no error_2 slack to give up a component. (The rotation also inflates Λ from 283 to 484 because the L1 terms aren't gauge-invariant; that's irrelevant, since error_2 alone is already infeasible.)
- **invsqrt margin attempt, closed (864f81cd error map):** err/tol reaches about ±0.95–0.999 in every octave (spread ≈ 0.24 tol everywhere; about ±2.5 last-rotation ulps at x > 2M). It is accumulated rounding noise scaled to the tolerance, not a few fixable spikes. The absolute minimum margin falls at large x only because tol is smallest there. The levers to widen it are dead: coefficient re-rounding (NR-0047), single-bit coefficient flips (NR-0086), angle LP (NR-0109/0208), guard bits / extra precision (NR-0224: halving the noise costs ≈ 1 bit everywhere). An extra divider row costs ≈ 32 Toffoli-eq against ≤ 9 saved.
- **Status after the v6 push:** no lever buildable in-session reaches 28.5. Every open item is ≤ 0.3% GM. 28.18965 (v5) stands; the forced leader reveals (Nov 20 milestone, Jan 4 Longest-Leader) remain the route to 30+.
- **New public files scan (2026-11-18 afternoon):** full recent-notebook sweep (not just the watcher's top 3). Candidates: invsqrt d0227f3c (maoxin123 re-run, claims 74,405.67 = −0.11%); DQI 40294f00 (Balram Srinivasan), 35fa5ef9 (manwithacat HW-30), 8a1bda64 (Atakan Aldemir), f2673ad2 (Ghazaros Barseghyan), bad74d3e (Aviral Trivedi); chemistry f05297b4 (Balram), 4a76fd2d (Arizal); invsqrt 485da015 (Aditya Ravi; 74,490.97, worse).
- **v6 (`kaggle_nb/v6_composite`) official results:** DQI 40294f00 default 3.46040797e11 (−0.105%) but **2 / 1 failed shots at salts s1 / s2** (mean over 5 salts 3.46547e11, worse than 5f901c55's 3.46404e11 with 0 failures at 7 salts). 35fa5ef9 default 3.46305e11, failures at 3 of 5 salts (mean 3.4753e11). 8a1bda64 default 3.46400e11. Chemistry: f05297b4 = 294,642,299.76 (worse than our 4d63be3d). Default-salt pick projects GM 28.1995.
- Kaggle attached an older version of maoxin123's notebook, so d0227f3c (and the loose DQI files) went into a private dataset `govindmulchandani/qcc-new1118` for `kaggle_nb/v6b_composite`, which writes a public-max zip and a salt-robust zip.
- **v6b results (official verifiers, 5 salts each):** invsqrt d0227f3c PASS at all 5 (74,405.67–74,405.72, −0.11%). DQI 8a1bda64: 0 failures at all 5 (3.46400e11). 40294f00: 2 / 1 failures at s1 / s2. f2673ad2 and bad74d3e FAIL at the default salt.
- **v6 public (ref 58516723)** = d0227f3c + 40294f00 + 4d63be3d: **LB 28.21029** (local 28.2103; mean over salts 28.1857, so private-risky).
- **v6 robust (ref 58516785)** = d0227f3c + 8a1bda64 + 4d63be3d: local 28.2005 at every salt. This is the safe final pick. Code submissions must be named submission.zip, so `kaggle_nb/v6r_robust` republishes it.
- **v6 robust scored 28.15748** against a local 28.2005: exactly one extra failed DQI shot (+0.46%). So the public LB uses its own hidden salt (not ""), and 8a1bda64 fails a shot there. Failure record across 6 draws (5 ours + LB): 40294f00 2 of 6, 8a1bda64 1 of 6, 5f901c55 0 of 8. **Final-pick plan:** (1) v6 public 28.21029 (40294f00); (2) d0227f3c + 5f901c55 + 4d63be3d (the asteriospanoras 28.20041 combo, 0 failures in every draw seen), to be submitted on the next daily slot as our safest high pick.

### 9.18 Day 4 (2026-11-19)
- **Overnight public frontier: 28.2316.** ozertuu's DQI **15a4b740** (3.45257e11, −0.33% vs 5f901c55) with d0227f3c + 4d63be3d scored 28.23162 on the LB. Manvender Singh's LP refit adds invsqrt **f7841eb4** (74,405.662) and chemistry **6ef06081** (294,642,169.51), both ≈ 0 gain. asteriospanoras estimates 0.42 expected failed DQI shots per draw for 15a4b740 (expected score ≈ 28.214), which is still better in expectation than 5f901c55.
- Also seen: ahoirg invsqrt k6c 4560ac0c (46 Q, ~79k: worse).
- `kaggle_nb/v7_composite` verifies f7841eb4 + 15a4b740 + 6ef06081 (invsqrt at 3 salts, chemistry 3, DQI 5) and writes submission.zip only if invsqrt and chemistry pass everywhere.
- The watcher's exit 4 last night was transient; restarted.
- **v7 verified (`kaggle_nb/v7_composite`):** chemistry 6ef06081 PASS at 3 salts (294,642,169.51); invsqrt f7841eb4 PASS at 3 (74,405.66–74,405.77); DQI 15a4b740 **0 failures at 5 salts** (3.452566–3.452568e11). Local GM 28.23163 (28.23164 with DQI averaged over salts). Submitted as ref 58538846.
- **v7 scored 28.23162** on the public LB (our best; 0 DQI failures on the LB salt too). It's the default final pick: 0 failures on 6 draws (5 ours + LB).
- Later public DQIs e39b1396 (boristown / Syed v13 line, 3.452555e11) and e208d350 (manwithacat, 3.452529e11; 48-draw expected 3.460e11) beat 15a4b740 by < 0.002% (+0.0001 GM): not worth a slot or the failure risk.
- **LB position (2026-11-19):** 28.23162 is tied around #158 of 3,362 (#100 28.23278, #75 28.23801, #50 28.29738, #30 28.49192). One invsqrt Toffoli ≈ +0.0056 GM (≈ 60 ranks here); one loader Toffoli per pass ≈ +0.0013; one DQI Toffoli ≈ +0.00003.
- **Free-Toffoli scans on invsqrt f7841eb4 (`research/inv_diff/`):** the 864f81cd → d0227f3c diff is one rescheduled block (an AND erased by HMR instead of uncomputed; mean Toffolis 1453.37 → 1451.44); f7841eb4 is d0227f3c with longer-precision angles. Three exact-checked scans:
  1. Toffolis whose result is affine in the live qubits (MBU + CNOT rebuild): 6 candidates on random inputs, all FAIL on powers of two; 0 once all x ≤ 4096 and 2^k ± 1 are sampled.
  2. CCX/CCZ whose pair product is affine (CX/CZ rewrite): 0 top-level.
  3. 3 conditioned CCZ with zero products, all FAIL the exact check.
  So no free Toffolis of these kinds; consistent with NR-0085.
- **v8 (ref 58542525):** invsqrt **ecf908fe** (boristown; a resynthesised block near body line 13115, mean Toffolis 1451.44 → 1450.94, cost 74,383.70, −0.029%) + DQI 15a4b740 + chemistry 6ef06081. Official: invsqrt PASS at 3 salts (74,383.69–74,383.79). Local GM **28.23441**. maoxin123's "QCC 28.23" notebook ships the same invsqrt.
- **v8 scored 28.23441** on the public LB (exactly the local figure; 15a4b740 again had 0 failures on the LB salt: 0 of 7 draws).
- **Sweep (2026-11-19 ~05:30 UTC):** maoxin123 "4,022-Toffoli chemistry loaders" ships chemistry a73b25d2 (294,643,137.73, worse than 6ef06081) with our invsqrt/DQI. Patrick Chan, Ali Haydar Ozdag, Manvender Singh, abcode80, Leo Tang (T7421), peter ju: all ≤ 28.2317 or same files. ahoirg's k6c = 4560ac0c (46q, 78,766; worse). Nothing beats v8.
- invsqrt ecf908fe has 7 static T-like gates: two conditioned controlled-S patterns (T·T·T† on q34/q19 and q34/q42) and one T on a fresh AND product (q25·q27 at line 7147). At most ~11 cost units each; not worth a resynthesis.
- **Final-pick hedge:** Kaggle scores the better of the two final picks, and DQI failures are drawn independently per file (seed = circuit hash + salt). manwithacat's 48-draw ranking (117 DQI files) puts e208d350 first by expected cost (3.460029e11), with e39b1396 and 84829f51 tied; 15a4b740 is at the same level (~0.4–0.5 failed shots per draw). Picking v8 (15a4b740) and v9 (e208d350, same invsqrt and chemistry) cuts P(both fail) from ~0.34 to ~0.13; expected private gain ≈ +0.01 GM. `kaggle_nb/v9_hedge` verifies e208d350 at 5 salts (from manwithacat/vqcc-apache-public-circuits) and packages v9.
- **v9 hedge verified (`kaggle_nb/v9_hedge`):** chemistry 6ef06081 PASS; invsqrt ecf908fe PASS at 3 salts; DQI e208d350 PASS at 5 salts but **1 failed shot at s2 and at private-a** (3.452529e11 clean, 3.468386e11 with one failure). 15a4b740 had 0 at the same 5 salts plus the LB salt. Local GM 28.23452 (DQI mean over salts 28.21725). Still the best second pick: a second pick with the same DQI hedges nothing, and 5f901c55 tops out at ~28.203. Submitted as ref 58546353 to see its LB-salt draw.
- **v9 scored 28.19142:** e208d350 lost one shot on the LB salt too (+0.46% DQI). Record now 3 failures in 6 draws (s2, private-a, LB) vs 15a4b740's 0 in 7. Still a valid hedge (independent draw; ≈ 28.2345 when clean) but only about a coin flip to be clean. Final-pick choice for the second slot (e208d350 vs 5f901c55, 0 of 8) to be revisited before the deadline.
- v10 research: session `inv-cross` (`research/inv_cross/`) generalises boristown's f7841eb4 → ecf908fe fix-up rewrite (conditioned CCX + CCZ → one CCZ by changing the upstream product; −0.5 mean Toffoli). Chemistry integeriser (NR-0178: exact CVP only 0.875× squared noise) and DQI failure fixes (NR-0055/0122, ≤ +0.007) are closed in the ledger.
