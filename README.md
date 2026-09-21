# grid-safety

Safety assurance for reinforcement learning in energy systems — comparing
penalty-based, constrained and shielded RL on the **gym-anm ANM6-Easy**
distribution network, with distribution-shift analysis and an **AMLAS-RL**
assurance argument.

## What this is

An RL agent manages a 6-bus distribution network: curtailing solar and wind,
charging and discharging a 100 MWh battery, and setting reactive power, every
15 minutes. It must keep branch flows below their thermal ratings and bus
voltages within 0.9–1.1 pu while wasting as little renewable generation as
possible.

The research question is not whether RL can control the network — the
[gym-anm paper](https://arxiv.org/abs/2103.07932) already showed it can. It is
**how to build a safety case for an RL controller in this setting**, and how
that case behaves when the operating conditions shift away from training.

Three safety treatments are compared under a matched training budget:

| treatment | mechanism |
|---|---|
| **penalty** | violations priced into the reward (stock gym-anm, λ = 100) |
| **constrained** | violation cost separated from reward; Lagrangian constraint |
| **shielded** | unsafe actions replaced by the MPC fallback controller |

See **[PLAN.md](PLAN.md)** for the full phased plan and the environment facts
established so far.

## Status

- **Phase 0 complete** — environment dissected, MPC baselines established.
- **Phase 1 complete** — published baselines reproduced; SAC selected as the
  Phase 2 baseline at a frozen budget of 1M steps.
- **Phase 2 next** — constrained and shielded variants.

## Phase 1 result: return and safety diverge

| policy | discounted return | violations | longest run | mean load | max load | SoC < 20 MWh |
|---|---|---|---|---|---|---|
| passive | — | 95.8 % | 34.0 | 1.545 | 1.956 | 100 % |
| MPC, constant forecast (N=16) | −129.1 | 36.4 % | 17.0 | 1.087 | 1.417 | 100 % |
| **PPO, 1M steps** | −114.6 ± 6.7 | 35.3 % | 17.0 | 0.966 | 1.421 | 100 % |
| **SAC, 1M steps** | **−51.7 ± 15.3** | 34.0 % | 17.5 | **0.882** | 1.451 | **50.2 %** |
| MPC, perfect forecast (N=32) | −14.7 | 7.9 % | 7.6 | 0.993 | 1.417 | 78.8 % |

*Violations, run length, loading and SoC over 10 episodes × 96 steps from
random initial conditions. Returns: γ = 0.995, 5 rollouts × 3000 steps.*

**The pipeline reproduces the paper.** SAC at 1M steps matches the published
−56.1 ± 26.8 (3M steps) within its spread; PPO sits ~1.4σ below its published
−93.6 ± 15.3, plausible at a third of the budget. Both beat MPC-constant on
return, and the SAC/PPO ordering and variance ratio match the paper.

**But return is not safety.** SAC's return is 2.2× better than PPO's, yet the
two are separated by 1.3 points on violation rate and both sit at parity with
MPC-constant — nowhere near MPC-perfect's 7.9 %. The reward-maximising agent
improved on the reward, not on safety, because safety is only one priced term
in that reward.

**The return gain is economic.** SAC holds its battery above the 20 MWh reserve
half the time; PPO never does. SAC absorbs surplus renewable output into
storage rather than curtailing it, and since curtailed energy counts as loss in
the reward, that recovers most of the return gap — with almost no safety
consequence.

The gap between SAC's 34.0 % and MPC-perfect's 7.9 % is the room Phase 2's
safety treatments need to close, without collapsing the return.

## Layout

```
PLAN.md          phased project plan + established environment facts
src/             train_anm.py  - SAC/PPO training, best-checkpoint tracking
                 eval_safety.py - safety-metric comparison vs MPC baselines
exploration/     Phase 0 environment survey and dissection scripts
results/         training curves (*_history.csv)
```

Model checkpoints (`*.zip`, `*_venv.pkl`) are excluded from version control.
Phase 1 models: `sac_anm_1m_best`, `ppo_anm_1m_best`.

## Setup

```bash
conda create -n grid python=3.11 && conda activate grid
pip install gym-anm stable-baselines3 tqdm rich matplotlib numba
```

Explore the environment:

```bash
python exploration/anm_anatomy.py                                  # devices, ratings, reward
python exploration/watch_anm_mpc.py --agent perfect --horizon 32   # browser view
python exploration/soc_sweep.py --agent perfect --horizon 32       # initial-SoC study
```

Train and evaluate:

```bash
python src/train_anm.py --algo sac --train_steps 1000000 --eval_every 100000 --save sac_anm_1m
python src/eval_safety.py --model sac_anm_1m_best --algo sac --episodes 10
```

## References

- Henry & Ernst, *gym-anm: Reinforcement Learning Environments for Active
  Network Management Tasks in Electricity Distribution Systems* (2021)
- Imrie et al., *Assuring the Safety of Reinforcement Learning Components:
  AMLAS-RL* (2025)
- Bethell et al., *Safe Reinforcement Learning in Black-Box Environments via
  Adaptive Shielding* (ADVICE, 2025)
