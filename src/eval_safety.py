"""
eval_safety.py
Compare a trained RL policy against the MPC baselines on SAFETY metrics
(Phase 3), not on discounted return.

Return is an economic quantity that mixes energy loss with violation penalty;
the safety case cares about the violation behaviour itself. These metrics are
also far cheaper to compute than the paper's discounted return.

Metrics per policy, over N episodes of 96 steps (one day each), from random
initial conditions:
  violation rate      % of steps with branch loading > 1 or V outside [0.9, 1.1]
  longest run         worst consecutive violation streak (thermal exposure)
  mean / max loading  how close to the limit, not just how often over
  SoC reserve breach  % of steps below a chosen reserve (unpriced constraint)
  collapses           terminal power-flow divergences

Reference: passive 95.8 %, MPC-constant(16) 36.5 %, MPC-perfect(32) ~0 %.

Run:
    python eval_safety.py --model sac_anm --episodes 5
    python eval_safety.py --episodes 5 --skip_rl      # baselines only
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np


def make_raw():
    import gymnasium as gym, gym_anm
    from gymnasium.wrappers import TimeLimit, RescaleAction
    e = gym.make("ANM6Easy-v0")
    e = TimeLimit(e.unwrapped, max_episode_steps=5000)
    e = RescaleAction(e, -1.0, 1.0)
    return e


def safety_of(sim):
    v = sim.state["bus_v_magn"]["pu"]
    i = sim.state["branch_i_magn"]["pu"]
    volts = np.array([float(np.real(x)) for x in v.values()])[1:]
    load = max(abs(float(np.real(c))) / getattr(sim.branches[b], "rate")
               for b, c in i.items() if getattr(sim.branches[b], "rate", None))
    soc = float(np.real(list(sim.state["des_soc"]["MWh"].values())[0]))
    return load, volts.min(), volts.max(), soc


def run_policy(name, step_fn, reset_fn, get_sim, episodes, steps, reserve):
    rows = []
    for ep in range(episodes):
        reset_fn(seed=2000 + ep)
        viol, run, longest, loads, breach, collapse = 0, 0, 0, [], 0, 0
        for t in range(steps):
            done = step_fn()
            sim = get_sim()
            ld, vmn, vmx, soc = safety_of(sim)
            loads.append(ld)
            bad = ld > 1.0 or vmn < 0.9 or vmx > 1.1
            viol += bad
            run = run + 1 if bad else 0
            longest = max(longest, run)
            breach += soc < reserve
            if done:
                collapse += 1
                reset_fn(seed=2000 + ep)
        rows.append((viol / steps * 100, longest, float(np.mean(loads)),
                     max(loads), breach / steps * 100, collapse))
    a = np.array(rows)
    print(f"{name:26s} {a[:,0].mean():>7.1f}% {a[:,1].mean():>7.1f} "
          f"{a[:,2].mean():>8.3f} {a[:,3].max():>8.3f} {a[:,4].mean():>8.1f}% "
          f"{int(a[:,5].sum()):>6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="saved SB3 model (no extension)")
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--reserve", type=float, default=20.0, help="SoC reserve, MWh")
    ap.add_argument("--skip_rl", action="store_true")
    args = ap.parse_args()

    import gymnasium as gym, gym_anm
    from gym_anm.agents import MPCAgentConstant, MPCAgentPerfect

    print(f"episodes={args.episodes} x {args.steps} steps | SoC reserve {args.reserve} MWh")
    print(f"{'policy':26s} {'viol':>8} {'longest':>7} {'meanLd':>8} "
          f"{'maxLd':>8} {'SoC<res':>9} {'colls':>6}")

    # ---- baselines on a bare env ----
    env = gym.make("ANM6Easy-v0"); u = env.unwrapped
    hi = env.action_space.high.copy()
    state = {}

    def reset_bare(seed):
        state["obs"], _ = env.reset(seed=seed)

    def get_sim():
        return u.simulator

    def mk_step(policy):
        def f():
            a = policy()
            state["obs"], r, te, tr, info = env.step(a)
            return te or tr
        return f

    run_policy("passive (no control)", mk_step(lambda: np.concatenate([hi[:2], np.zeros(4)])),
               reset_bare, get_sim, args.episodes, args.steps, args.reserve)
    for label, cls, N in [("MPC constant N=16", MPCAgentConstant, 16),
                          ("MPC perfect N=32", MPCAgentPerfect, 32)]:
        ag = cls(u.simulator, u.action_space, u.gamma, safety_margin=0.96, planning_steps=N)
        run_policy(label, mk_step(lambda ag=ag: ag.act(u)),
                   reset_bare, get_sim, args.episodes, args.steps, args.reserve)

    # ---- the trained RL policy, through its training normalisation ----
    if args.model and not args.skip_rl:
        from stable_baselines3 import SAC, PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        Algo = SAC if args.algo == "sac" else PPO
        venv = VecNormalize(DummyVecEnv([make_raw]), norm_obs=True,
                            norm_reward=False, gamma=0.995)
        try:
            venv = VecNormalize.load(args.model + "_venv.pkl", venv.venv)
            venv.training = False
            print(f"(loaded normalisation stats from {args.model}_venv.pkl)")
        except Exception as e:
            print(f"(WARNING: no saved VecNormalize stats - returns/actions unreliable: {e})")
        model = Algo.load(args.model, env=venv)
        rl_state = {}

        def reset_rl(seed):
            venv.seed(seed); rl_state["obs"] = venv.reset()

        def step_rl():
            a, _ = model.predict(rl_state["obs"], deterministic=True)
            rl_state["obs"], r, done, infos = venv.step(a)
            return bool(done[0])

        run_policy(f"RL ({args.algo})", step_rl, reset_rl,
                   lambda: venv.venv.envs[0].unwrapped.simulator,
                   args.episodes, args.steps, args.reserve)

    print("\nreference: passive 95.8% | MPC-constant 36.5% | MPC-perfect ~0% (pre-charged)")
    print("an RL policy below 36.5% has beaten the deployable conventional controller.")


if __name__ == "__main__":
    main()
