"""
explore_grid2op.py
Kick-the-tyres exploration of Grid2Op - the analog of donkey_manual_test.py.
NO safety machinery: just find out how the environment behaves.

Answers: does it run clean? what's in the observation? is there a continuous
safety margin with ground truth? how is game-over signalled? what does the
do-nothing baseline do on its own? how fast does it step? what do the
scenarios (the distribution-shift axis) look like? does rendering work?

Install:  pip install grid2op numba matplotlib
Run:      python explore_grid2op.py
          python explore_grid2op.py --env l2rpn_case14_sandbox --scenarios 5
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="rte_case5_example")
    ap.add_argument("--scenarios", type=int, default=5, help="how many to try")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--test", action="store_true", default=True,
                    help="use bundled data (no download)")
    ap.add_argument("--render", default="grid_state.png")
    args = ap.parse_args()

    import grid2op
    from grid2op.Agent import DoNothingAgent
    print(f"grid2op {grid2op.__version__}")

    env = grid2op.make(args.env, test=args.test)
    obs = env.reset()
    print(f"\nENV '{args.env}': {env.n_sub} substations, {env.n_line} lines, "
          f"{env.n_gen} generators, {env.n_load} loads")

    # --- what's in the observation? ---
    print("\n--- observation (the 'telemetry') ---")
    for a in ["rho", "line_status", "a_or", "v_or", "load_p", "gen_p", "topo_vect"]:
        v = getattr(obs, a, None)
        if v is not None and hasattr(v, "shape"):
            print(f"  {a:12s} shape {str(v.shape):8s} e.g. {np.round(np.asarray(v, float)[:3], 3)}")
    print("\n  rho = line current / thermal limit. rho > 1 is an OVERLOAD.")
    print(f"  max rho at reset: {float(obs.rho.max()):.4f}")

    # --- scenarios = the distribution-shift axis ---
    try:
        subpaths = env.chronics_handler.real_data.subpaths
        print(f"\n{len(subpaths)} scenarios ('chronics') available - the shift axis.")
    except Exception:
        print("\n(could not list scenarios)")

    # --- do-nothing baseline across scenarios ---
    print(f"\n--- do-nothing baseline, {args.scenarios} scenarios "
          f"(max {args.max_steps} steps) ---")
    agent = DoNothingAgent(env.action_space)
    t0 = time.time(); total_steps = 0
    print(f"{'scenario':>9} {'steps':>7} {'game_over':>10} {'max rho':>9} "
          f"{'overload %':>11}")
    for k in range(args.scenarios):
        obs = env.reset(); done = False; s = 0; rhos = []
        info = {}
        while not done and s < args.max_steps:
            obs, reward, done, info = env.step(agent.act(obs, 0.0, done))
            rhos.append(float(obs.rho.max())); s += 1
        total_steps += s
        r = np.array(rhos) if rhos else np.array([0.0])
        print(f"{k:>9} {s:>7} {str(done):>10} {r.max():>9.3f} "
              f"{(r > 1).mean()*100:>10.1f}%")
        if done and info.get("exception"):
            print(f"           cause: {str(info['exception'])[:90]}")
    el = time.time() - t0
    print(f"\nspeed: {total_steps/el:.0f} steps/sec "
          f"({total_steps} steps in {el:.1f}s)")
    print(f"  -> 300 bootstrap replicates x 100 steps ~= "
          f"{300*100/(total_steps/el)/60:.1f} min")

    # --- rendering ---
    try:
        import matplotlib; matplotlib.use("Agg")
        from grid2op.PlotGrid import PlotMatplot
        fig = PlotMatplot(env.observation_space).plot_obs(obs)
        fig.savefig(args.render, dpi=110)
        print(f"\nrendered grid state -> {args.render}")
    except Exception as e:
        print(f"\nrendering failed: {e}")

    print("\nWhat to look at next:")
    print("  - rho is the CTE analog: continuous margin, ground truth, hard threshold at 1.0")
    print("  - game_over is the absorbing failure (the 'off-road' state)")
    print("  - scenario-to-scenario variation is the natural distribution-shift axis")
    print("  - for live viewing inside a loop: env.render() (slows things down)")
    print("  - for post-hoc analysis: grid2op Runner + grid2viz (web dashboard)")


if __name__ == "__main__":
    main()
