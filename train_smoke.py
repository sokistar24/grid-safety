"""
train_smoke.py
Plumbing check for RL on ANM6-Easy BEFORE committing to the 3M-step runs.

Verifies, on a deliberately tiny budget:
  - env wraps correctly (TimeLimit T=5000, VecNormalize obs)
  - SAC and PPO both construct with the paper's hyperparameters and run
  - the paper's EXACT evaluation protocol computes (N_r=5 rollouts of T=3000,
    gamma-discounted) -- reused verbatim for the real runs

This does NOT train to performance. Targets to reproduce LATER (Table 2):
  SAC -56.1 +/- 26.8   PPO -93.6 +/- 15.3   (bracketed by MPC-const -129, MPC-perfect -14.7)

Paper settings baked in: gamma=0.995, eval T=3000 (truncation error <1e-2),
train horizon T=5000, obs+action normalised, 3M steps x 5 seeds for the real run.

Run:  python train_smoke.py                 # ~1-2 min, tiny budget
      python train_smoke.py --algo ppo
      python train_smoke.py --train_steps 300000 --eval   # longer, real eval
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np


def evaluate(model, make_env, gamma=0.995, n_rollouts=5, T=3000):
    """Paper eq. 7: mean over N_r rollouts of the gamma-discounted return."""
    returns = []
    for i in range(n_rollouts):
        env = make_env()
        obs, _ = env.reset(seed=1000 + i)
        G, disc, done = 0.0, 1.0, False
        for t in range(T):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            G += disc * r
            disc *= gamma
            if term or trunc:
                obs, _ = env.reset()
        returns.append(G)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--train_steps", type=int, default=3000,
                    help="tiny by default; 3_000_000 for the real run")
    ap.add_argument("--eval", action="store_true",
                    help="run the full paper eval (N_r=5, T=3000); slow")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import gymnasium as gym
    import gym_anm
    from gymnasium.wrappers import TimeLimit, RescaleAction
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_raw():
        e = gym.make("ANM6Easy-v0")
        e = TimeLimit(e.unwrapped, max_episode_steps=5000)   # paper train horizon
        e = RescaleAction(e, -1.0, 1.0)                       # normalised actions
        return e

    venv = VecNormalize(DummyVecEnv([make_raw]), norm_obs=True, norm_reward=False,
                        gamma=0.995)

    common = dict(learning_rate=3e-4, gamma=0.995, seed=args.seed, verbose=0)
    if args.algo == "sac":
        model = SAC("MlpPolicy", venv, buffer_size=1_000_000, batch_size=256,
                    tau=0.005, ent_coef="auto", **common)
    else:
        model = PPO("MlpPolicy", venv, n_steps=2048, batch_size=64, n_epochs=10,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                    **common)

    print(f"algo={args.algo}  train_steps={args.train_steps}  gamma=0.995")
    print(f"targets (Table 2): SAC -56.1+/-26.8  PPO -93.6+/-15.3")
    t0 = time.time()
    model.learn(total_timesteps=args.train_steps, progress_bar=False)
    print(f"training ran {args.train_steps} steps in {time.time()-t0:.0f}s -> plumbing OK")

    if args.eval:
        print("running paper eval (N_r=5, T=3000)... this is slow")
        mean, std = evaluate(model, make_raw)
        print(f"discounted return: {mean:.1f} +/- {std:.1f}")
    else:
        print("(skip full eval on smoke run; add --eval for the real metric)")


if __name__ == "__main__":
    main()
