# Safety assurance for RL in energy systems — project plan

Environment: `gym-anm` / **ANM6-Easy** (6-bus distribution network, 15-min control,
24-h deterministic profiles entered at a random time of day).
Goal: compare safety mechanisms for RL under a **matched, modest training budget**,
then construct an AMLAS-RL assurance argument. Full-length training happens last.

## Reference points

| policy | violations (96 steps) | discounted return |
|---|---|---|
| passive (no control) | 95.8 % | — |
| `MPCAgentConstant` N=16 | 36.4 % | −129.1 ± 0.4 (paper) |
| `MPCAgentPerfect` N=32 | 7.9 % (random SoC) / ~0 % (pre-charged) | −14.7 ± 0.2 (paper) |
| PPO 1M (ours) | 35.3 % | −114.6 ± 6.7 |
| SAC 1M (ours) | 34.0 % | −51.7 ± 15.3 |
| **PPO 300k penalty (Phase 2 baseline)** | **~48 % (3-ep check; 10-ep pin pending)** | **−132.1 ± 7.4** |

The 35.3 % figure belongs to the 1M run. At the frozen 300k budget the penalty
baseline is ~48 %, so the room Phase 2 must close is **48 → 8**, and budget
feasibility arguments use 48 %, not 35 %.

## Established environment facts
- Action = 6 continuous **injection set-points** `[P_solar, P_wind, P_batt, Q_solar, Q_wind, Q_batt]`;
  `P_batt < 0` charges, `> 0` discharges. Nothing is discretised.
- Observation = 18 values: `dev_p(7), dev_q(7), des_soc(1), gen_p_max(2), aux(1)`; `aux` = time-of-day slot.
- Reward = −(energy loss + λ·Φ); λ = **100** in code (docs say 10³); clipping (1, 100).
  Energy loss = line losses + curtailed renewables + storage round-trip loss.
  Terminal collapse costs −r_clip/(1−γ) = **−20 000**.
- Dynamics deterministic; **only stochasticity is the random initial state** (time of day and SoC).
- Violations are **tolerated** (priced); **collapse** (power-flow divergence) terminates.
  Competent policies never collapse the network.
- Two congestion types: **export** at buses 3–4 (fixable instantly by curtailment) and
  **import** at bus 5 (EV load; fixable only by storage charged in advance).
- The shipped MPC is a **DC** OPF and does not optimise reactive power for storage/solar.
- Timing: PPO ~41 min / 100k steps; SAC ~102 min / 100k steps.
- **Verified:** `u.penalty > 0` agrees with the state-threshold violation event
  (load > 1.0 or V ∉ [0.9, 1.1]) on ~96 % of steps and is a strict superset
  (check_cost_signal.py on the trusted policy: NEW 48.3 % vs OLD 51.7 %, only-NEW = 0).
  Priced penalty magnitudes: median 0.125, max 16.5 per step; e_loss ∈ [−0.11, 0.19].

## Code inventory

| file | role |
|---|---|
| `src/train_anm.py` | penalty PPO/SAC training, stock-reward eval, best-checkpoint tracking |
| `src/eval_safety.py` | Phase 3 safety metrics vs MPC baselines |
| `src/check_cost_signal.py` | diagnostic: cost-signal validity on a trusted policy |
| `src/train_lagrangian.py` | PPO-Lagrangian 2a; state-threshold cost; `--warm_start` |
| `src/shield_eval.py` | `ModelBasedShield` (physics check) + unified shield evaluation harness |
| `src/learned_shield.py` | rollout collection, safety-critic training, `LearnedShield` class |

## Phase 0 — environment study ✅
Anatomy, action sensitivity, daily cycle, MPC lookahead sweep, initial-SoC sweep.

## Phase 1 — baselines ✅
SAC and PPO at 1M steps reproduce the published results (SAC −51.7 vs paper −56.1 ± 26.8).
Both beat MPC-constant on return. **Finding: return and safety diverge** — SAC's return is
2.2× PPO's, yet violations differ by 1.3 points and both sit at parity with MPC-constant.
SAC's return gain is economic (it stores surplus rather than curtailing), not a safety gain.

**Phase 2 algorithm: PPO at a frozen 300k-step budget.** Justification:
- safety parity with SAC (35.3 % vs 34.0 % at 1M); SAC's advantage is economic
- on-policy training makes Lagrangian constraint enforcement well-defined
- ~2.4× faster per wall-clock, making multiple methods and seeds feasible
- PPO's violation rate plateaus early (37.3 % at 100k → 35.3 % at 1M);
  note the 300k policy itself sits at ~48 %

## Phase 2 — safety mechanisms

### Two families
| family | methods | needs RL training |
|---|---|---|
| **training-time** — change what the agent learns | penalty, Lagrangian | yes |
| **deployment-time** — filter a fixed policy | physics (model-based) shield, learned shield | **no** — wraps the frozen penalty policy |

Holding the policy AND the fallback fixed across both shields isolates each
detector's contribution.

### The methods — design decisions on record

1. **Penalty** — stock reward, λΦ fixed. Control condition, shared starting point.
   **Status:** s0 done (−132.1 ± 7.4 full eval; ~48 % violations). s1 was lost to a
   save-path collision (both runs wrote `ppo_pen_s0`) — s1/s2 re-running with correct
   names. Caveat observed at s0: quick-eval "best" checkpoint scored WORSE than the
   final model on full eval (−343 vs −132) — use the FINAL model unless full eval
   says otherwise.

2. **Lagrangian** — Φ removed from reward; cost c_t = **state-threshold violation
   indicator** (the exact event eval_safety counts — SR-traceable); budget d; dual
   ascent per rollout. **Diagnosed from-scratch failure (both seeds, cost pinned at
   0.995, λ climbing linearly):** the cost reader was NOT broken (verified above).
   At random init the agent violates ~all steps; an always-1 indicator is a constant
   reward offset with zero gradient; the only dense signal (−e_loss) rewards
   never-curtailing, which drives MORE violations — the agent trains itself into the
   trap. The stock reward escapes because λΦ is a magnitude (dense gradient).
   **Design consequence: warm-start from the penalty policy AND its VecNormalize
   stats is load-bearing** (weights without their obs statistics ≈ random policy ≈
   same trap). Watch: cost rate starts ~0.45–0.5 and drifts toward d = 0.15;
   monotone λ climb with cost stuck ≳ 0.4 ⇒ d infeasible at 300k ⇒ relax to 0.25.

3. **Physics (model-based) shield** — **deterministic mode built** (`shield_eval.py`).
   Check: deepcopy the live env, apply the proposed action, run the TRUE power flow
   one step, threshold flows/voltages; reject ⇒ `MPCAgentConstant(16)` acts.
   `--margin` rejects below rating (crude robustness). `missed` column must be ~0
   (the check is exact ⇒ built-in self-test).
   **Stated assumption: perfect physics + perfect one-step forecast.** Valid
   in-distribution because profiles are deterministic — a forecaster trained on
   historical data would be exact here, and the deterministic check is that
   forecaster's limit case. One-step only: cannot pre-empt the bus-5 trap.
   **Stochastic mode — deferred to Phase 4 by design, not omission.** Architecture
   (four slots): forecast (persistence, or historical lookup **frozen on nominal
   profiles**); error model (assumed Gaussian 5–10 % of load, or conformal residual
   intervals from Phase 5); physics (simulator transition on sampled inputs);
   decision rule (robust: any scenario violates ⇒ reject; or chance-constrained:
   violating fraction > ε ⇒ reject, with **ε deliberately tied to the Lagrangian
   budget d** — one risk dial across both families). In-distribution the mode is
   **degenerate**: forecast residuals are identically zero, conformal widths are
   zero, and it collapses back to the deterministic check — hence the deferral.

4. **Learned shield** — **built** (`learned_shield.py`). ADVICE's post-shield
   structure with **threshold labels**: unsafe(s, a) = violation within the next
   k steps. Terminal labels (ADVICE's definition) would give an EMPTY unsafe set
   here, because violations are tolerated/priced and competent policies never
   collapse. **k is chosen empirically** from the collect-time prevalence table:
   with ~48 % per-step violations clustered in the daily peaks, large k saturates
   the label toward all-unsafe — the Lagrangian's no-contrast failure resurfacing
   at the label level. Pick the largest k comfortably < ~85 % positive (expect
   4–8, not the 16–32 first guessed). Collection: stochastic policy sampling +
   ε = 0.15 uniform-random actions (the critic must score MPC/candidate actions it
   never sees on-policy), seeds 10000+ (never the eval seeds 2000–2009), SoC
   stored per step so stage 2b relabels WITHOUT re-collecting. Training:
   episode-level split (step-level leaks). Deployment fallbacks: `mpc` (identical
   to the physics shield — isolates the detector) and `candidates` (ADVICE-style
   search, safe candidate nearest the proposal — minimal interference; tests
   whether the critic's knowledge alone can steer around the trap).
   **Coupling hypothesis:** MPC-constant plans under a constant forecast and never
   holds reserve (Phase 0: SoC < 20 MWh 100 % of the time), so it will NOT
   pre-charge on an early warning — learned+mpc may squander the anticipation.
   The learned(mpc) vs learned(candidates) gap MEASURES detector–fallback coupling.

5. **Hybrid** *(later)* — physics for the one-step check, critic for multi-step,
   uncertainty on the learned part. Now concretely motivated by the coupling gap.

### Stages
- **2a — line/voltage violations only.** Clean comparison; the Lagrangian has one λ.
  The violation event is **instantaneous** throughout Phase 2 (every baseline was
  measured on it); duration is captured by the longest-run metric.
- **2b — add SoC reserve** (threshold TBD: 20 or 10 MWh). The constraints *conflict*:
  the reserve wants to hold charge, line safety at bus 5 wants to discharge.
  Physics shield: one-line change to the check. Learned shield: relabel the stored
  data (SoC already saved). A **duration-based thermal event** (e.g. > 3 consecutive
  over-limit steps) is an SR refinement that belongs here or in Phase 6 — if adopted
  it must change cost, labels, checks and metrics **in one consistent move**, never
  silently.

### Training runs
| run | seeds | note |
|---|---|---|
| penalty (shared) | 3 | s0 done; s1/s2 re-running (save paths must encode seed) |
| Lagrangian 2a | 3 | `--warm_start models/ppo_pen_s0` (policy + VecNormalize stats) |
| Lagrangian 2b | 3 | |

Shields need **no** RL training; they run against the frozen penalty policy.

### Order (status)
1. penalty PPO 300k × 3 seeds — s0 ✅, s1/s2 re-running
2. physics shield, deterministic — ✅ built; run vs s0
3. Lagrangian 2a — warm-start version ready; s0–s2
4. learned shield — ✅ built; collect → prevalence → train → eval
5. stage 2b
6. hybrid
7. physics shield, stochastic — **moved to Phase 4** (degenerate in-distribution)

**Build and validate one method at a time.**

### Predictions to test
- Shields reduce **export** violations (buses 3–4) far more than **import** (bus 5),
  since import congestion requires anticipation a one-step check lacks.
- `noSafe%` (no one-step-safe action exists) concentrates at bus 5 and ≈ the
  shielded residual violation rate ⇒ the failure is anticipation, not detection.
- The learned shield outperforms the physics shield on import violations if k is
  long enough AND the fallback acts on the warning; learned(mpc) ≈ physics shield
  on the bus-5 residual despite good critic AUC ⇒ the coupling squanders the
  warning; learned(candidates) < learned(mpc) on bus 5 confirms the critic's
  knowledge is actionable.
- Label prevalence rises steeply with k; the usable horizon is shorter than the
  MPC-matched 16–32 first assumed.
- Lagrangian beats penalty on violation rate at some cost in return (unchanged).

## Phase 3 — safety metrics, not return (~½ day)
Violation rate; longest consecutive violation run; margin distribution; SoC-reserve
breaches; collapse probability. Return as a secondary column. Break violations down
by **bus / scenario** (shield_eval already emits the per-branch tally).

## Phase 4 — distribution shift (~2–3 days)
**Operational (covariate) shifts only** — dynamics unchanged: EV charging profile
(timing, magnitude, AM/PM asymmetry); seasonal renewable profiles; demand amplitude
and **peak timing**. Each is a subclass editing `P_loads` / `P_maxs`.
Re-run Phase 3 for all methods.

**The physics shield gets TWO rows under shift:**
- *deterministic / tracking* — the deepcopy check copies the SHIFTED env, so it
  keeps a silently perfect forecast even under shift. Unrealistic; report it as
  the **assumed-oracle upper bound** on attainable shield performance.
- *stochastic, forecast frozen on nominal* — historical-lookup forecast calibrated
  on nominal profiles + nominal-calibrated error model, now facing shifted reality.
  This is the **valid, realistic case**: physics never breaks, but the forecast and
  its error model can break silently — the structural analogue of a nominally
  calibrated runtime monitor failing under shift.

**The gap between the two rows is the measured cost of the perfect-forecast
assumption** — the headline number of Phase 4. Expected: learned shield and
policies degrade; tracking oracle does not; frozen-forecast shield degrades in
proportion to how far the shift exceeds its error model.
*(Line outages change P(s′|s,a) — a different MDP. Contingency section, not here.)*

## Phase 5 — detection and uncertainty (~1–2 days)
Quantify shift directly on the 18-D observations: Mahalanobis, KL, Jensen–Shannon,
plus dynamics-model discrepancy. Run-time signal: PPO **cross-seed policy
disagreement**. Uncertainty on the **learned shield** itself. **Conformal residual
intervals are the stochastic shield's error model** (plug-in point defined in
Phase 2, method 3). Key question: does the signal rise *before* violations — early
enough to hand over before a multi-step trap?

## Phase 6 — AMLAS-RL argument (~1 week, writing)
Stage 1 scoping + SRs (duration-based thermal limit — see stage-2b note on
consistent redefinition; voltage band; collapse; SoC reserve); Stage 2 ML-level
requirements; Stage 3 plans (documented from Phases 1–2); Stage 4 learning
evidence; **Stage 5** verification — DTMC from trials, PCTL in PRISM with bootstrap
intervals; Stage 6 deployment, with Phase 4 as operating-domain evidence.
**Assumption ledger for the assurance case:** (i) perfect one-step forecast
(discharged in-distribution by environment determinism; withdrawn in Phase 4);
(ii) critic coverage = on-policy + ε-uniform actions; (iii) fallback competence
(MPC-constant is myopic and reserve-blind); (iv) evaluation reuses frozen training
`VecNormalize` statistics.

## Phase 7 — full training, last
Full budgets and 5 seeds; re-run key comparisons. If short-budget conclusions hold,
report both; if not, that is a finding about how safety conclusions depend on
training budget.

## Risks
- Too-weak policies hide safety differences — raise budget **before** Phase 2, not midway.
- Collapse probability may be unmeasurable in-distribution; expect content only under shift.
- Evaluation must reuse training `VecNormalize` statistics (`training=False`).
- Learned-shield **label saturation** at large k — checked BEFORE training via the
  prevalence table; if even k=4 saturates, that is itself a finding.
- `candidates` fallback safety is bounded by the critic's off-policy correctness —
  if it underperforms `mpc`, raise ε and re-collect before blaming the mechanism.
- Quick-eval best-checkpoint selection can pick worse-than-final (observed at s0);
  prefer the final model unless full eval disagrees.
- Save-path collisions destroy seeds (observed: s1); save names must encode the seed.
- Five mechanisms = five code paths; build and validate incrementally.
