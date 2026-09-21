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


def evaluate(model, venv_stats, gamma=0.995, n_rollouts=5, T=3000, quick=False):
    """Paper eq.7: mean gamma-discounted return over N_r rollouts of length T.

    CRITICAL: the policy was trained on VecNormalize-normalised observations, so
    evaluation MUST apply the same statistics. model.predict() does NOT do this
    automatically - feeding raw observations to a policy trained on normalised
    ones yields meaningless returns (this was the original bug).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    if quick:
        n_rollouts, T = 3, 1000          # cheaper, for mid-training progress only

    eval_venv = VecNormalize(DummyVecEnv([make_raw]), norm_obs=True,
                             norm_reward=False, gamma=gamma)
    eval_venv.obs_rms = venv_stats.obs_rms   # reuse the TRAINING statistics
    eval_venv.training = False               # freeze them
    eval_venv.norm_reward = False

    returns, collapses = [], 0
    for i in range(n_rollouts):
        eval_venv.seed(1000 + i)
        obs = eval_venv.reset()
        G, disc = 0.0, 1.0
        for t in range(T):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = eval_venv.step(a)
            G += disc * float(r[0])
            disc *= gamma
            if done[0]:
                collapses += 1               # terminal collapse costs -r_clip/(1-gamma)
        returns.append(G)
    eval_venv.close()
    return float(np.mean(returns)), float(np.std(returns)), collapses


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
        m, s, c = evaluate(model, venv)
        print(f"discounted return: {m:.1f} +/- {s:.1f}   terminal collapses: {c}")
        return

    print(f"algo={args.algo}  steps={args.train_steps}  eval_every={args.eval_every}")
    print(f"targets: SAC -56.1+/-26.8  PPO -93.6+/-15.3  | brackets -129 / -14.7")
    print(f"{'step':>10} {'quick return':>20} {'col':>3} {'elapsed':>9} {'ETA':>9}")

    class EvalCB(BaseCallback):
        """Evaluate periodically; keep the BEST checkpoint, not the last.

        Returns oscillate on this task (a later policy is often worse than an
        earlier one), so ending an 8-hour run on a bad swing would waste it.
        Every improvement is written to <save>_best; the final model is also
        saved to <save> so both are available.
        """
        def __init__(self):
            super().__init__()
            self.t0 = time.time()
            self.best = -np.inf
            self.history = []

        def _on_step(self):
            if self.num_timesteps % args.eval_every == 0:
                m, s, c = evaluate(self.model, venv, quick=True)
                el = time.time() - self.t0
                frac = self.num_timesteps / args.train_steps
                eta = el / frac - el if frac > 0 else 0
                star = ""
                if m > self.best:
                    self.best = m
                    star = "  <- best"
                    if args.save:
                        self.model.save(args.save + "_best")
                        venv.save(args.save + "_best_venv.pkl")
                self.history.append((self.num_timesteps, m, s, c))
                print(f"{self.num_timesteps:>10} {m:>10.1f}+/-{s:>7.1f} {c:>3d} "
                      f"{el/60:>7.1f}m {eta/60:>7.1f}m{star}")
            return True

    t0 = time.time()
    cb = EvalCB()
    model.learn(total_timesteps=args.train_steps, progress_bar=True, callback=cb)
    print(f"\ntrained {args.train_steps} steps in {(time.time()-t0)/60:.1f} min")

    if args.save:
        model.save(args.save); venv.save(args.save + "_venv.pkl")
        print(f"saved final -> {args.save}")
        if cb.best > -np.inf:
            print(f"saved best  -> {args.save}_best   (quick return {cb.best:.1f})")
        import csv as _csv
        with open(args.save + "_history.csv", "w", newline="") as f:
            w = _csv.writer(f); w.writerow(["step", "quick_return", "std", "collapses"])
            w.writerows(cb.history)
        print(f"saved history -> {args.save}_history.csv")

    print("final full eval of the LAST model (N_r=5, T=3000)...")
    m, s, c = evaluate(model, venv)
    print(f"LAST  discounted return: {m:.1f} +/- {s:.1f}   terminal collapses: {c}")

    if args.save and cb.best > -np.inf:
        Algo2 = SAC if args.algo == "sac" else PPO
        best = Algo2.load(args.save + "_best", env=venv)
        m2, s2, c2 = evaluate(best, venv)
        print(f"BEST  discounted return: {m2:.1f} +/- {s2:.1f}   terminal collapses: {c2}")
        print(f"-> use {args.save}_best for evaluation if it scores higher.")


if __name__ == "__main__":
    main()
