"""
watch_anm_mpc.py
Watch the BASE CONTROLLER (MPC) operate the network in the browser.

Runs gym-anm with an MPC agent while rendering, so you can see the controller
curtailing generation (red bars = curtailed active power, per the legend) and
the constraint-violation crosses appearing/disappearing as the day progresses.

    python watch_anm_mpc.py --agent constant
    python watch_anm_mpc.py --agent perfect --margin 0.96 --horizon 8
    python watch_anm_mpc.py --agent passive        # contrast: no control at all
    python watch_anm_mpc.py --scale 1.3            # stressed peak

Open the URL it prints. Compare 'passive' against 'constant' against 'perfect'
to see the controller earning its keep.
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np


def safety(sim):
    v = sim.state["bus_v_magn"]["pu"]; i = sim.state["branch_i_magn"]["pu"]
    volts = np.array([float(np.real(x)) for x in v.values()])[1:]
    load = [abs(float(np.real(c))) / getattr(sim.branches[b], "rate")
            for b, c in i.items() if getattr(sim.branches[b], "rate", None)]
    return volts, (max(load) if load else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="constant",
                    choices=["passive", "constant", "perfect"])
    ap.add_argument("--margin", type=float, default=0.96)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--steps", type=int, default=192)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.25)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale the 24h load/gen profiles (peak stress)")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    import gymnasium, gym_anm
    from gym_anm.agents import MPCAgentConstant, MPCAgentPerfect
    from gym_anm.envs.anm6_env.anm6_easy import ANM6Easy

    if args.scale != 1.0:
        sc = args.scale
        class ScaledANM6(ANM6Easy):
            def __init__(self):
                super().__init__()
                self.P_loads = [np.array(p) * sc for p in self.P_loads]
                self.P_maxs = [np.array(p) * sc for p in self.P_maxs]
        env = ScaledANM6(); u = env
    else:
        env = gymnasium.make("ANM6Easy-v0"); u = env.unwrapped

    obs, _ = env.reset(seed=0)
    hi = env.action_space.high.copy()

    if args.agent == "passive":
        pol = lambda o: np.concatenate([hi[:2], np.zeros(4)])
        label = "PASSIVE (no curtailment)"
    elif args.agent == "constant":
        ag = MPCAgentConstant(u.simulator, u.action_space, u.gamma,
                              safety_margin=args.margin, planning_steps=args.horizon)
        pol = lambda o: ag.act(u)
        label = f"MPC constant forecast (m={args.margin}, N={args.horizon})"
    else:
        ag = MPCAgentPerfect(u.simulator, u.action_space, u.gamma,
                             safety_margin=args.margin, planning_steps=args.horizon)
        pol = lambda o: ag.act(u)
        label = f"MPC PERFECT forecast (m={args.margin}, N={args.horizon})"

    if not args.no_render:
        print("starting the browser renderer...")
        u.render(mode="human", skip_frames=args.skip)
        print(">>> OPEN THE URL ABOVE, then watch the controller work <<<\n")
        time.sleep(3)

    print(f"policy: {label}   profile scale: {args.scale}")
    print(f"{'step':>5} {'hour':>6} {'maxload':>8} {'Vmin':>7} {'reward':>8}  state")
    lds, viols = [], 0
    try:
        for s in range(1, args.steps + 1):
            obs, r, te, tr, info = env.step(pol(obs))
            if not args.no_render:
                u.render(mode="human", skip_frames=args.skip)
            v, ml = safety(u.simulator)
            lds.append(ml)
            bad = ml > 1.0 or v.min() < 0.95 or v.max() > 1.05
            viols += int(bad)
            if s % (args.skip + 1) == 0:
                print(f"{s:>5} {s*0.25:>6.2f} {ml:>8.3f} {v.min():>7.4f} "
                      f"{r:>8.2f}  {'VIOLATION' if bad else 'within envelope'}")
            time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\ninterrupted")

    n = len(lds)
    print(f"\n{label}")
    print(f"  steps {n}   max loading {max(lds):.3f}   "
          f"violations {viols}/{n} = {viols/max(n,1)*100:.1f}%")
    print("  (in the browser: red bars = curtailed generation, "
          "red X = constraint violated)")


if __name__ == "__main__":
    main()
