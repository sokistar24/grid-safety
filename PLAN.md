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
| **PPO 300k penalty s0 (Phase 2 baseline)** | **47.2 ± 1.1 %** (10 eps) | **−132.1 ± 7.4** |
| **PPO-Lagrangian 300k (1 seed, warm-start)** | **30.7 %** (10 eps) | **−354.4 ± 33.9** |

The 35.3 % figure belongs to the 1M run. At the frozen 300k budget the penalty
baseline is 47.2 %, so the room Phase 2 must close is **47 → 8**.

## Established environment facts
- Action = 6 continuous **injection set-points** `[P_solar, P_wind, P_batt, Q_solar, Q_wind, Q_batt]`;
  `P_batt < 0` charges, `> 0` discharges. Nothing is discretised.
- Observation = 18 values: `dev_p(7), dev_q(7), des_soc(1), gen_p_max(2), aux(1)`; `aux` = time-of-day slot.
- Reward = −(energy loss + λ·Φ); λ = **100** in code (docs say 10³); clipping (1, 100).
  Terminal collapse costs −r_clip/(1−γ) = **−20 000**.
- Dynamics deterministic; **only stochasticity is the random initial state** (time of day and SoC).
- Violations are **tolerated** (priced); **collapse** terminates. Competent policies never collapse.
- Two congestion types: **export** at buses 3–4 (fixable instantly by curtailment) and
  **import** at bus 5 (EV load; fixable only by storage charged in advance).
- The shipped MPC is a **DC** OPF and does not optimise reactive power for storage/solar.
- Timing: PPO ~41 min / 100k steps; SAC ~102 min / 100k steps.
- **Verified:** `u.penalty > 0` agrees with the state-threshold violation event
  (load > 1.0 or V ∉ [0.9, 1.1]) on ~96 % of steps and is a strict superset
  (check_cost_signal.py: NEW 48.3 % vs OLD 51.7 %, only-NEW = 0).
  Priced penalty magnitudes: median 0.125, max 16.5 per step; e_loss ∈ [−0.11, 0.19].
- **Real-time feasibility (measured):** one shield decision costs 0.5–1.1 s wall-clock
  in our Python harness (incl. up to ~80 simulated candidates) against a **900 s**
  control interval — ~0.1 % of the budget. A production implementation calls the
  power-flow solver directly on the current state (µs–ms per evaluation, no env
  deepcopy). Simulation-in-the-loop is standard at this cadence (gym-anm's own MPC
  solves a multi-step OPF per 15-min step; operators run SCOPF/N−1 screening).
  For sub-second loops the ordering flips: only the learned component (<1 ms per
  forward pass) is deployable — there, learning buys speed; here, it buys (or was
  meant to buy) anticipation.

## Code inventory

| file | role |
|---|---|
| `src/train_anm.py` | penalty PPO/SAC training, stock-reward eval, best-checkpoint tracking |
| `src/eval_safety.py` | Phase 3 safety metrics vs MPC baselines |
| `src/check_cost_signal.py` | diagnostic: cost-signal validity on a trusted policy |
| `src/train_lagrangian.py` | PPO-Lagrangian 2a; state-threshold cost; `--warm_start` |
| `src/shield_eval.py` | physics shield (repair × no-safe variants), hybrid wiring, unified eval harness, `--per_episode` paired output, mean±s.e. |
| `src/learned_shield.py` | rollout collection; critic training (`binary`/`count`/`v` targets, dR² diagnostic); `LearnedShield`; `HybridShield` (severity-capped) |

## Phase 0 — environment study ✅
Anatomy, action sensitivity, daily cycle, MPC lookahead sweep, initial-SoC sweep.

## Phase 1 — baselines ✅
SAC and PPO at 1M steps reproduce the published results. **Finding: return and
safety diverge** — SAC's return gain over PPO is economic (stores surplus rather
than curtailing), not a safety gain.

**Phase 2 algorithm: PPO at a frozen 300k-step budget** (safety parity with SAC,
on-policy Lagrangian well-defined, 2.4× cheaper, early violation plateau).

## Phase 2 — safety mechanisms

### Statistical standard (adopted mid-phase, applies to all claims)
All configurations are evaluated on the SAME seeds 2000–2009, so per-episode
differences are **paired**; `--per_episode` prints them and the viol column
carries ± s.e. A claim needs a clear paired majority (≥7/10 with real margins),
not a difference of means within joint noise. Policies are currently **one
training seed each** — cross-seed spread is still owed (see Remaining work).

### Two families
| family | methods | needs RL training |
|---|---|---|
| **training-time** — change what the agent learns | penalty, Lagrangian | yes |
| **deployment-time** — filter a fixed policy | physics shield, learned shield, hybrid | no |

### Method outcomes (design decisions + what happened)

1. **Penalty** — control condition. s0: −132.1 ± 7.4, **47.2 ± 1.1 %** violations,
   residual (2,5):433 + (2,4):20. Quick-eval "best" checkpoint scored WORSE than
   final on full eval (−343 vs −132) — use FINAL. s1 lost to a save-path collision
   once; re-runs must encode the seed in the save name.

2. **Lagrangian** — state-threshold indicator cost (SR-traceable); dual ascent.
   From-scratch failure diagnosed (cost pinned 0.995, λ linear): NOT a sensor bug —
   an always-1 indicator is a constant offset with zero gradient, and −e_loss then
   rewards never-curtailing. **Warm-start from the penalty policy + its VecNormalize
   stats is load-bearing** and worked: cost 0.41 at first eval → plateau ~0.27,
   λ → 2.28 still creeping. **d = 0.15 NOT met at 300k** (dual update ≈ +0.006/rollout
   — too slow, not infeasible-diverging); Phase 7 tests whether full budget closes it.
   Deterministic eval **30.7 %** — first controller in the project to beat
   MPC-constant — at return −354 vs penalty's −132. Training (stochastic) cost rate
   27.1 % vs deterministic eval 30.7 %: measured, direction opposite to the naive
   guess. NOTE: the surviving `ppo_lag_s0` artefact is the `--seed 1` run (second
   collision incident); label it "1 seed" until properly-named re-runs land.

3. **Physics shield** — exact one-step check (deepcopy → true power flow), i.e. a
   **one-interval-ahead forecast used exactly**, the most forecastable quantity in
   power systems; it never simulates further ahead. Variants: repair ∈ {mpc,
   nearest} × no-safe ∈ {fallback, keep, leastbad}. `missed` ≡ 0 held in every run
   (self-test). Stochastic mode stays deferred to Phase 4 (degenerate
   in-distribution: residuals identically zero). Default: `nearest:keep` for
   headline runs, `mpc:keep` for cheap iteration (identical safety to
   mpc:fallback at half the interventions).

4. **Learned shield (critic-only)** — ADVICE structure, threshold labels. **Closed
   by measurement, two gates:** (i) binary labels saturate — collection data
   violates 60.2 %/step, positives 88.7 % at k=4 and 97.1 % at k=8, usable k ≤ 2 =
   no anticipation; (ii) count (cost-to-go) target: action gain **dR² = +0.000
   (k=16) / −0.006 (k=8)** — a critic cannot rank actions from on-policy windows
   (action ≈ π(state)+noise collinearity; single-action advantage ≪ window
   variance). The state-value alone IS learnable (R² 0.71–0.72, MAE ≈ 1.5 steps).
   ADVICE's framing assumes rare, terminal unsafe events; neither holds in a
   priced high-violation regime. (Optional appendix run, not yet done: re-collect
   at ε = 0.5 to isolate the collinearity share of dR² = 0.)

5. **Hybrid** — pulled forward from "later" by necessity (dR² = 0):
   Q(s,a) = exact one-step cost + k·V̂(s′); relative rule (replace iff a candidate
   saves ≥ δ expected violating steps); severity cap on replacements after the
   first result. **Closed by measurement** — see results table and finding 6.
   The pre-registered union-shield diagnostic (does δ=0.5 push SoC<res below the
   unshielded 59.9 %?) came back negative (64.1 %): union not built.

### Phase 2a — results (penalty s0 / Lagrangian 1 seed; eval seeds 2000–2009, paired)

| config | viol | longest | maxLd | SoC<res | note |
|---|---|---|---|---|---|
| passive | 95.8 % | 34.0 | 1.96 | 100 % | |
| penalty s0 (unshielded) | 47.2±1.1 % | 15.9 | 3.30 | 59.9 % | |
| MPC-constant N=16 | 36.4 % | 17.0 | 1.42 | 100 % | deployable baseline |
| + hybrid, capped 1.5, δ=1.0 | 35.7±1.6 % | 15.0 | 3.16 | 62.3 % | loses to floor 9/10 paired |
| + physics (mpc repair, any no-safe rule) | 35.3 % | 17.0 | 1.42 | 100 % | collapses onto the fallback |
| Lagrangian + physics (mpc repair) | 34.3 % | 16.8 | 1.42 | 100 % | shield HURT the better policy |
| + hybrid, capped 1.5, δ=0.5 | 33.0±1.6 % | 14.7 | 3.10 | 64.1 % | interv 40.3 % ≈ floor's 40.0 %: matched budget, noisy selector; lit up previously-clean (1,2); floor wins 7/10 |
| Lagrangian (unshielded, 1 seed) | 30.7 % | 15.0 | 3.41 | 59.6 % | first to beat MPC-constant; return −354 |
| Lagrangian + physics (nearest:keep) | 29.7 % | 14.7 | 1.40 | 98.1 % | no measurable gain, reserve destroyed → **don't stack** (see 2b) |
| + physics (nearest:keep) | **29.6±0.6 %** | 14.6 | 1.43 | 81.9 % | **the one-step floor**; noSafe ≡ viol; (2,5):284 |
| + hybrid, uncapped | 28.8±1.1 % | **8.0** | **5.72** | 58.9 % | severity-financed — not a valid config (32× nominal I²t) |
| MPC-perfect N=32 | 7.9 % | 7.6 | 1.42 | 78.8 % | oracle: multi-step planning + true forecast |

### Findings on record
1. **Repair policy dominates detector quality.** MPC repair collapses ANY policy
   onto the fallback (fingerprints: maxLd 1.417, SoC 100 %, longest 17) — helping a
   policy worse than MPC (47.2→35.3) and hurting one better (30.7→34.3). Minimal
   repair (nearest safe action) reaches the floor from either policy. The hopeless-
   step no-safe rule is irrelevant (identical safety, keep/fallback/leastbad): the
   battery is already empty there, so all actions lead to ~the same state.
2. **29.6 % is the empirical one-step limit.** noSafe ≡ viol and missed = 0 in every
   physics run: every residual violation is a certified one-step trap, entirely
   bus-5 import. Two very different policies converge on it (29.6/29.7, 284/285).
   The 22 points to MPC-perfect are the anticipation prize.
3. **A per-step guarantee does not compose into a system-level one.** The shield's
   local claim held exactly (missed = 0) while the system-level rate WORSENED on
   the Lagrangian (mpc repair) — and the mechanism is physical: one-step repairs of
   import congestion discharge the battery, manufacturing the next trap. Assurance
   claims must be scoped to what the shield actually guarantees.
4. **Count objectives are severity-blind.** The uncapped hybrid INVENTED the
   accept-now-to-save-later trade (charged through the import peak): longest run
   16→8, reserve preserved, bus-5 275 — paid at 5.72× rating (≈32× nominal I²t
   heating). No single violation-rate column ranks these configurations; count,
   duration and severity move independently. The SR-definition question
   (instantaneous vs duration vs severity) is now empirical → 2b / Phase 6.
5. **A pure learned shield is inert here** (dR² ≈ 0, mechanisms in method 4) —
   the measured justification for the hybrid split, and for why ADVICE-style
   shields don't transfer to priced-violation regimes.
6. **Greedy value-guided anticipation is infeasible here, measured two ways:**
   per-action advantage (≤ ~0.5 expected steps: one 15-min action moves SoC
   5–8 MWh) < V̂ error (1.5 steps). δ=1.0 never acts (35.7 %); δ=0.5 acts on noise
   (33.0 % at the floor's own 40 % intervention budget — a directly measured
   **price of an approximate detector**, 3.4 points, plus violations on a branch
   no other config touched). The anticipation prize requires committed multi-step
   planning under a forecast (MPC-perfect's two ingredients); no greedy per-step
   filter, learned or physical, can claim it. Successor (future work, out of
   Phase 2 scope): receding-horizon planning over the simulator with V̂ as
   terminal value.

### Predictions scorecard (registered before results)
- Export ≫ import reduction — **confirmed** ((2,4) 20→0–3 everywhere; residual is (2,5)).
- noSafe concentrates at bus 5 and ≈ residual — **confirmed as an identity**.
- Learned shield beats physics on import if k long enough + fallback acts —
  **refuted** (dR² = 0; hybrid negative); the coupling half survived in stronger
  form as finding 1.
- Label prevalence steep, usable k < 16–32 — **confirmed, worse than guessed** (k ≤ 2).
- Lagrangian beats penalty on violations at return cost — **confirmed** (30.7 vs
  47.2; −354 vs −132).

### Stages
- **2a — line/voltage violations, instantaneous event.** ✅ closed (results above);
  remaining: seeds for error bars.
- **2b — add SoC reserve (threshold TBD: 20 or 10 MWh).** Now unusually well
  motivated: the reserve is THE contested resource of 2a — the floor holds it at
  81.9 % breach, MPC repair drains it to 100 %, the uncapped hybrid was the only
  config that protected it (58.9 %) and paid in severity, and stacking the shield
  on the Lagrangian destroyed its reserve for zero violation gain (**don't stack**
  is a 2a finding that 2b will re-score). Design question to settle ON PAPER
  first: adding `soc < reserve` to the physics check makes discharge-based repairs
  themselves violations — the shield loses its main repair lever exactly where it
  needs it. That conflict is 2b's thesis, not a nuisance. Learned components:
  relabel the stored data (SoC already saved). Duration-based thermal event stays
  an SR refinement here or Phase 6; if adopted, cost/labels/checks/metrics change
  in ONE consistent move.

### Remaining work to close Phase 2
1. Penalty s1/s2: `eval_safety` + `shield_eval --variants "nearest:keep"
   --per_episode` per seed (headline configs only; hybrid stays a one-seed
   ablation — it is a negative result with a paired analysis behind it).
2. Lagrangian: properly-named second/third seeds (`ppo_lag_s1`, `ppo_lag_s2`,
   warm-started); keep the "surviving artefact is seed 1" note until then.
3. Return column for shielded configs (the frontier plot needs it; nearest:keep
   preserves ~60 % of policy actions, so penalty+floor likely keeps most of −132
   vs the Lagrangian's −354 — potentially the paper's headline trade).
4. Stage 2b.

## Phase 3 — safety metrics, not return (~½ day)
Violation rate; longest run; **margin/severity distribution** (now empirically
required — finding 4); SoC-reserve breaches; collapse probability; per-branch
breakdown (harness emits it). Return as the frontier's second axis (item 3 above).
Frame as multi-objective: no single column ranks configurations.

## Phase 4 — distribution shift (~2–3 days)
Operational (covariate) shifts only: EV profile (timing, magnitude, asymmetry);
seasonal renewables; demand amplitude and peak timing. Re-run Phase 3 for all
surviving methods.

**Physics shield gets TWO rows:** *deterministic/tracking* (deepcopy silently
tracks the shifted env — assumed-oracle upper bound) vs *stochastic with forecast
frozen on nominal* (+ nominal-calibrated error model — the valid, realistic case).
**The gap = measured cost of the perfect-forecast assumption** (headline number).
**New hook from 2a:** V̂ literally memorised the nominal daily pattern — it is the
most shift-fragile component in the system, plausibly more fragile than the policy;
its degradation under shifted peak timing is a second headline comparison.
*(Line outages = different MDP → contingency section, not here.)*

## Phase 5 — detection and uncertainty (~1–2 days)
Shift metrics on the 18-D observations (Mahalanobis, KL, JS) + dynamics-model
discrepancy; PPO cross-seed disagreement as a runtime signal. **Uncertainty on V̂**
now has a measured in-distribution baseline (MAE 1.5 steps) and a measured bar to
beat: usable per-action signals here are ≤ 0.5 steps, i.e. below today's noise —
conformal residuals on V̂ quantify exactly this. Conformal intervals also feed the
Phase 4 stochastic shield's error model. Key question unchanged: does the signal
rise BEFORE violations?

## Phase 6 — AMLAS-RL argument (~1 week, writing)
Stage 1 SRs (instantaneous vs duration vs severity thermal event — now an
EMPIRICAL choice, finding 4; voltage band; collapse; SoC reserve); Stages 2–4 from
Phases 1–2; Stage 5 DTMC + PCTL in PRISM with bootstrap intervals; Stage 6 with
Phase 4 as operating-domain evidence.
**Assumption ledger:** (i) perfect one-interval forecast (discharged
in-distribution by determinism; withdrawn in Phase 4); (ii) critic coverage =
on-policy + ε-uniform; (iii) fallback competence (MPC-constant is myopic and
reserve-blind — finding 1 makes this load-bearing); (iv) frozen VecNormalize stats
at eval; (v) **per-step guarantees are not system-level guarantees** (finding 3);
(vi) real-time feasibility argued at 15-min cadence (measured ~1 s/decision vs
900 s budget; production = direct power-flow calls).

## Phase 7 — full training, last
Full budgets, 5 seeds; includes the Lagrangian d = 0.15 feasibility retest (dual
was still ascending at 300k). If short-budget conclusions hold, report both; if
not, that is itself a finding about budget-dependence of safety conclusions.

## Risks
- Too-weak policies hide safety differences — raise budget BEFORE Phase 2, not midway.
- Collapse probability likely unmeasurable in-distribution; expect content under shift.
- Evaluation must reuse training `VecNormalize` statistics (`training=False`).
- ~~Label saturation at large k~~ — **realised and measured** (method 4); kept as
  a worked example of checking labels before training.
- Save-path collisions destroy seeds — **realised twice** (penalty s1, Lagrangian
  seed 0). Save names must encode the seed; read the command back before Enter.
- Quick-eval best-checkpoint can pick worse-than-final — **realised** (s0); prefer FINAL.
- Severity-blind objectives can convert severity into count — **realised**
  (uncapped hybrid); any count-style objective needs a severity constraint.
- Greedy anticipation is closed as an avenue (finding 6); do not spend further
  effort below the 29.6 % floor without multi-step planning.
- Single-seed claims: nothing involving the Lagrangian is cross-seed yet; the
  paired-eval standard covers eval noise, not training-seed variance.
