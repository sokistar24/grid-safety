"""
anm_playground.py
Explore gym-anm's BASE CONTROLLER (the MPC agent) and how to stress the network
at peak times - preparation for the dispatch case study.

Two experiments:

 A) CONTROLLER PARAMETERS. gym_anm.agents.MPCAgent solves a DC optimal power
    flow over a planning horizon. Its two knobs are:
      safety_margin  - operate to this fraction of the limits (0.9 = 90%).
                       This IS the "defined operating envelope" dial.
      planning_steps - MPC horizon N (how far ahead it optimises).
    Sub-classes differ ONLY in forecast():
      MPCAgentConstant - assumes the present persists (naive, deployable)
      MPCAgentPerfect  - looks up the true future (oracle upper bound)
    => an AI forecaster is a third subclass implementing forecast().

 B) PEAK STRESS. ANM6Easy holds 24h profiles in self.P_loads (devices 1,3,5)
    and self.P_maxs (devices 2,4). Scaling them raises peak demand/generation,
    so you can ask: at what stress level does the conventional controller stop
    holding the envelope? That is where an AI has something to add.

Run:  python anm_playground.py
      python anm_playground.py --steps 96 --scales 1.0,1.2,1.4
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np


def safety(sim):
    v = sim.state["bus_v_magn"]["pu"]
    i = sim.state["branch_i_magn"]["pu"]
    volts = np.array([float(np.real(x)) for x in v.values()])[1:]   # drop slack
    load = [abs(float(np.real(c))) / getattr(sim.branches[b], "rate")
            for b, c in i.items() if getattr(sim.branches[b], "rate", None)]
    return volts, (max(load) if load else np.nan)


def run(env, policy, steps, seed=0):
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    lds, vmin, rew, viol = [], [], [], 0
    for _ in range(steps):
        obs, r, te, tr, info = env.step(policy(obs))
        v, ml = safety(u.simulator)
        lds.append(ml); vmin.append(v.min()); rew.append(r)
        if ml > 1.0 or v.min() < 0.95 or v.max() > 1.05:
            viol += 1
    return dict(maxload=max(lds), meanload=float(np.mean(lds)),
                vmin=min(vmin), viol_pct=viol / steps * 100,
                reward=float(np.mean(rew)))


def make_scaled_env(scale_load, scale_gen):
    """Subclass ANM6Easy and scale the 24h profiles to raise peak stress."""
    import gymnasium
    from gym_anm.envs.anm6_env.anm6_easy import ANM6Easy

    class ScaledANM6(ANM6Easy):
        def __init__(self):
            super().__init__()
            self.P_loads = [np.array(p) * scale_load for p in self.P_loads]
            self.P_maxs = [np.array(p) * scale_gen for p in self.P_maxs]

    return ScaledANM6()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=96, help="96 = 24h at 15 min")
    ap.add_argument("--scales", default="1.0,1.2,1.5",
                    help="load/gen scale factors for the peak-stress sweep")
    args = ap.parse_args()

    import gymnasium, gym_anm
    from gym_anm.agents import MPCAgentConstant, MPCAgentPerfect

    # ---------------- A) controller parameters ----------------
    print("=" * 74)
    print("A) BASE CONTROLLER: MPC DC-OPF. Only forecast() differs between agents.")
    print("=" * 74)
    env = gymnasium.make("ANM6Easy-v0"); u = env.unwrapped
    hi = env.action_space.high.copy()
    passive = lambda o: np.concatenate([hi[:2], np.zeros(4)])

    print(f"{'policy':34s} {'maxload':>8} {'viol%':>7} {'Vmin':>7} {'reward':>8}")
    r = run(env, passive, args.steps)
    print(f"{'passive (no curtailment)':34s} {r['maxload']:>8.3f} "
          f"{r['viol_pct']:>6.1f}% {r['vmin']:>7.4f} {r['reward']:>8.2f}")

    for margin in (0.9, 0.96, 1.0):
        for N in (1, 8):
            ag = MPCAgentConstant(u.simulator, u.action_space, u.gamma,
                                  safety_margin=margin, planning_steps=N)
            r = run(env, lambda o: ag.act(u), args.steps)
            print(f"{'MPC constant m=%.2f N=%d' % (margin, N):34s} "
                  f"{r['maxload']:>8.3f} {r['viol_pct']:>6.1f}% "
                  f"{r['vmin']:>7.4f} {r['reward']:>8.2f}")
    ag = MPCAgentPerfect(u.simulator, u.action_space, u.gamma,
                         safety_margin=0.96, planning_steps=8)
    r = run(env, lambda o: ag.act(u), args.steps)
    print(f"{'MPC PERFECT m=0.96 N=8 (oracle)':34s} {r['maxload']:>8.3f} "
          f"{r['viol_pct']:>6.1f}% {r['vmin']:>7.4f} {r['reward']:>8.2f}")
    print("\n  safety_margin = the operating-envelope dial; lower = more conservative.")
    print("  the CONSTANT-vs-PERFECT gap is the value of forecast quality:")
    print("  that gap is exactly the slot an AI forecaster fills.")

    # ---------------- B) peak stress ----------------
    print()
    print("=" * 74)
    print("B) PEAK STRESS: scale the 24h profiles, find where the conventional")
    print("   controller stops holding the envelope.")
    print("=" * 74)
    print(f"{'scale':>6} {'policy':>16} {'maxload':>8} {'viol%':>7} {'reward':>8}")
    for sc in [float(x) for x in args.scales.split(",")]:
        e2 = make_scaled_env(sc, sc)
        u2 = e2
        hi2 = e2.action_space.high.copy()
        pas = lambda o: np.concatenate([hi2[:2], np.zeros(4)])
        r = run(e2, pas, args.steps)
        print(f"{sc:>6.2f} {'passive':>16} {r['maxload']:>8.3f} "
              f"{r['viol_pct']:>6.1f}% {r['reward']:>8.2f}")
        try:
            ag = MPCAgentConstant(e2.simulator, e2.action_space, e2.gamma,
                                  safety_margin=0.96, planning_steps=8)
            r = run(e2, lambda o: ag.act(e2), args.steps)
            print(f"{'':>6} {'MPC constant':>16} {r['maxload']:>8.3f} "
                  f"{r['viol_pct']:>6.1f}% {r['reward']:>8.2f}")
        except Exception as ex:
            print(f"{'':>6} {'MPC constant':>16}  failed: {str(ex)[:40]}")
    print("\n  rising scale = higher peak demand/generation = a harder dispatch")
    print("  problem. The scale at which MPC-constant starts violating is where")
    print("  the case study becomes interesting, and it is also a clean,")
    print("  physically-meaningful DISTRIBUTION SHIFT axis.")


if __name__ == "__main__":
    main()
