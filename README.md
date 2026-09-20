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

See **[PLAN.md](PLAN.md)** for the full phased plan, reference numbers, and the
environment facts established so far.

## Status

**Phase 0 complete** — environment dissected, MPC baselines established.
**Phase 1 in progress** — training SAC/PPO to a competent-enough baseline.

Known issue: `src/train_anm.py` evaluates the policy on raw observations while
training applies `VecNormalize`, so reported discounted returns are not yet
comparable to published values. Fix pending.

## Reference points

| policy | violations (96 steps) | discounted return (paper) |
|---|---|---|
| passive (no control) | 95.8 % | — |
| MPC, constant forecast (N=16) | 36.5 % | −129.1 ± 0.4 |
| MPC, perfect forecast (N=32) | ~0 % (pre-charged) | −14.7 ± 0.2 |
| PPO | — | −93.6 ± 15.3 |
| SAC | — | −56.1 ± 26.8 |

## Layout

```
PLAN.md          phased project plan + established environment facts
src/             training harness (train_anm.py, train_smoke.py)
exploration/     Phase 0 scripts: environment anatomy, sensitivity
                 sweeps, SoC sweeps, MPC baselines, live viewers
```

## Setup

```bash
conda create -n grid python=3.11 && conda activate grid
pip install gym-anm stable-baselines3 tqdm rich matplotlib numba
```

Quick look at the environment:

```bash
python exploration/anm_anatomy.py          # devices, ratings, daily cycle, reward
python exploration/watch_anm_mpc.py --agent perfect --horizon 32   # browser view
python exploration/soc_sweep.py --agent perfect --horizon 32       # initial-SoC study
```

Train:

```bash
python src/train_anm.py --algo sac --train_steps 300000 --eval_every 50000
```

## References

- Henry & Ernst, *gym-anm: Reinforcement Learning Environments for Active
  Network Management Tasks in Electricity Distribution Systems* (2021)
- Imrie et al., *Assuring the Safety of Reinforcement Learning Components:
  AMLAS-RL* (2025)
- Bethell et al., *Safe Reinforcement Learning in Black-Box Environments via
  Adaptive Shielding* (ADVICE, 2025)
