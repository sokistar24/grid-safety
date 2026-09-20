# Safety assurance for RL in energy systems — project plan

Environment: `gym-anm` / **ANM6-Easy** (6-bus distribution network, 15-min control,
24-h deterministic profiles entered at a random time of day).
Goal: compare safety treatments for RL under a **matched, modest training budget**,
then construct an AMLAS-RL assurance argument. Full-length training happens last.

## Reference points (established)

| policy | violations (96 steps) | published discounted return |
|---|---|---|
| passive (no control) | 95.8 % | — |
| `MPCAgentConstant` N=16 | 36.5 % | −129.1 ± 0.4 |
| `MPCAgentPerfect` N=32 | ~0 % (pre-charged) | −14.7 ± 0.2 |
| PPO | — | −93.6 ± 15.3 |
| SAC | — | −56.1 ± 26.8 |

Key environment facts established in Phase 0:
- Action = 6 continuous **injection set-points**: `[P_solar, P_wind, P_batt, Q_solar, Q_wind, Q_batt]`;
  `P_batt < 0` charges, `> 0` discharges. Nothing is discretised.
- Observation = 18 values: `dev_p(7), dev_q(7), des_soc(1), gen_p_max(2), aux(1)`; `aux` = time-of-day slot.
- Reward = −(energy loss + λ·Φ), λ = **100** in code (docs say 10³), clipping (1, 100).
  Terminal collapse costs −r_clip/(1−γ) = **−20 000**.
- Dynamics are deterministic; **the only stochasticity is the random initial state**
  (time of day *and* SoC) — so "probability of violation" means over initial conditions.
- Violations are **tolerated** (priced); **collapse** (power-flow divergence) terminates.
- Storage is the only temporal lever; lookahead N=32 (8 h) saturates performance.
- Timing: ~17–49 steps/s ⇒ 300 k steps ≈ 5 h; 3 M × 5 seeds ≈ 10 days.

## Phase 1 — competent-enough baselines (~4–6 h)
Train SAC and PPO at a fixed budget (start 300 k). Success = rising return curve and
beating `MPCAgentConstant` on **violation rate** — not matching −56.1.
**Freeze the chosen budget**; it is the controlled variable for all later phases.
Save models *and* `VecNormalize` statistics. 3 seeds now, 5 later.

> Evaluation must apply the **training normalisation statistics** (`obs_rms`, `training=False`).
> Evaluating a normalised-input policy on raw observations produces meaningless returns.
> Protocol (paper eq. 7): N_r = 5 rollouts × T = 3000 steps, γ-discounted.

## Phase 2 — three safety treatments (~1–2 days)
Same algorithm, same budget, same seeds; **only the safety mechanism changes**.
1. **Penalty** — stock gym-anm (λΦ inside the reward). Control condition.
2. **Constrained** — expose Φ(s) as a separate cost in `info`, optionally remove it
   from the reward, add a Lagrangian multiplier on E[cost] ≤ d.
3. **Shielded** — RL proposes; a safety check judges; `MPCAgentConstant` substitutes
   when rejected (deployable fallback, not an oracle).

Add **SoC ≥ 20 % reserve** as a constraint the reward never encodes — the clean test
of whether constrained RL handles an unpriced requirement.

## Phase 3 — safety metrics, not return (~½ day)
Violation rate; longest consecutive violation run; margin distribution (distance to
limit, not just breaches); SoC-reserve breaches; collapse probability.
Return becomes a secondary column showing what safety cost.

## Phase 4 — distribution shift (~2–3 days)
**Operational shifts only** (covariate shift; dynamics unchanged):
EV charging profile (timing, magnitude, AM/PM asymmetry); seasonal renewable profiles;
demand amplitude and **peak timing**. Each is a subclass editing `P_loads` / `P_maxs`.
Re-run Phase 3 metrics for all three treatments under each shift.
*(Line outages change P(s′|s,a) — a different MDP. Keep for a contingency section, not here.)*

## Phase 5 — detection (~1–2 days)
Quantify shift directly on the 18-D observations (no representation learning needed):
Mahalanobis, KL, Jensen–Shannon, plus **dynamics-model discrepancy** (one-step model
error isolates dynamics change from visitation change).
Run-time signal: SAC **critic-ensemble disagreement** (`policy_kwargs=dict(n_critics=5)`);
PPO **cross-seed policy disagreement** (free from the seeds already trained).
Key question: does the signal rise *before* violations do?

## Phase 6 — AMLAS-RL argument (~1 week, writing)
- **Stage 1** scoping + SRs: duration-based thermal limit, voltage band, collapse, SoC reserve.
- **Stage 2** translation to ML-level requirements.
- **Stage 3** data/training/verification plans (documented retrospectively from Phases 1–2).
- **Stage 4** learning evidence.
- **Stage 5** verification: discretise state → transitions from trials → PCTL in PRISM,
  with bootstrap confidence intervals (existing toolchain).
- **Stage 6** deployment; Phase 4 results as operating-domain evidence.

## Phase 7 — full training, last (~2–4 days compute)
3 M steps × 5 seeds to match/exceed −56.1; re-run key comparisons at full length.
If short-budget conclusions hold, report both. **If they do not, that is itself a finding**
about how safety conclusions depend on training budget.

## Risks
- If the frozen budget leaves agents too weak for safety differences to show,
  raise it **before** Phase 2 — not midway.
- Collapse probability may be unmeasurably small in-distribution for all treatments
  (as in-distribution risk was for the AV case); expect that branch to gain content
  only under Phase 4 shifts.
- Evaluation-normalisation mismatch (see Phase 1 note) silently invalidates returns.
