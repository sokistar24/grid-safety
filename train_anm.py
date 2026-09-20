"""
train_anm.py
Train SAC/PPO on ANM6-Easy with a live progress bar and periodic evaluation,
so you can watch the discounted return climb toward the paper's targets.

Reproduction targets (Table 2): SAC -56.1+/-26.8, PPO -93.6+/-15.3
MPC brackets: constant -129.1, perfect -14.7.

Timing note: ~49 steps/sec on a laptop => 300k steps ~1.7h, 3M steps ~17h/seed.
Progress bar shows steps done / ETA; the eval callback prints the return every
--eval_every steps using the paper's exact protocol (N_r rollouts, T=3000, gamma).

Run:
  python train_anm.py --algo sac --train_steps 300000 --eval_every 50000
  python train_anm.py --algo sac --train_steps 3000000 --eval_every 100000 --save sac_anm
Resume / evaluate a saved model:
  python train_anm.py --algo sac --load sac_anm --eval_only
"""
import argparse, time, warnings, os
warnings.filterwarnings("ignore")
import numpy as np


def make_raw():
    import gymnasium as gym, gym_anm
    from gymnasium.wrappers import TimeLimit, RescaleAction
    e = gym.make("ANM6Easy-v0")
    e = TimeLimit(e.unwrapped, max_episode_steps=5000)
    e = RescaleAction(e, -1.0, 1.0)
    return e


def evaluate(model, gamma=0.995, n_rollouts=5, T=3000, quick=False):
    """Paper eq.7: mean gamma-discounted return over N_r rollouts of length T."""
    if quick:
        n_rollouts, T = 3, 1000     # faster, for mid-training progress
    returns = []
    for i in range(n_rollouts):
        env = make_raw()
        obs, _ = env.reset(seed=1000 + i)
        G, disc = 0.0, 1.0
        for t in range(T):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            G += disc * r; disc *= gamma
            if term or trunc:
                obs, _ = env.reset()
        returns.append(G)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--train_steps", type=int, default=300000)
    ap.add_argument("--eval_every", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    import gymnasium as gym
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback

    venv = VecNormalize(DummyVecEnv([make_raw]), norm_obs=True, norm_reward=False,
                        gamma=0.995)
    Algo = SAC if args.algo == "sac" else PPO

    if args.load:
        model = Algo.load(args.load, env=venv)
        print(f"loaded {args.load}")
    else:
        common = dict(learning_rate=3e-4, gamma=0.995, seed=args.seed, verbose=0)
        if args.algo == "sac":
            model = SAC("MlpPolicy", venv, buffer_size=1_000_000, batch_size=256,
                        tau=0.005, ent_coef="auto", **common)
        else:
            model = PPO("MlpPolicy", venv, n_steps=2048, batch_size=64, n_epochs=10,
                        gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                        **common)

    if args.eval_only:
        print("full paper eval (N_r=5, T=3000)...")
        m, s = evaluate(model)
        print(f"discounted return: {m:.1f} +/- {s:.1f}")
        return

    print(f"algo={args.algo}  steps={args.train_steps}  eval_every={args.eval_every}")
    print(f"targets: SAC -56.1+/-26.8  PPO -93.6+/-15.3  | brackets -129 / -14.7")
    print(f"{'step':>10} {'quick return':>14} {'elapsed':>9} {'ETA':>9}")

    class EvalCB(BaseCallback):
        def __init__(self): super().__init__(); self.t0 = time.time()
        def _on_step(self):
            if self.num_timesteps % args.eval_every == 0:
                m, s = evaluate(self.model, quick=True)
                el = time.time() - self.t0
                frac = self.num_timesteps / args.train_steps
                eta = el / frac - el if frac > 0 else 0
                print(f"{self.num_timesteps:>10} {m:>8.1f}+/-{s:>4.1f} "
                      f"{el/60:>7.1f}m {eta/60:>7.1f}m")
            return True

    t0 = time.time()
    model.learn(total_timesteps=args.train_steps, progress_bar=True, callback=EvalCB())
    print(f"\ntrained {args.train_steps} steps in {(time.time()-t0)/60:.1f} min")

    if args.save:
        model.save(args.save); venv.save(args.save + "_venv.pkl")
        print(f"saved -> {args.save}")

    print("final full eval (N_r=5, T=3000)...")
    m, s = evaluate(model)
    print(f"FINAL discounted return: {m:.1f} +/- {s:.1f}")


if __name__ == "__main__":
    main()
