"""
soc_sweep.py
How does the storage unit's INITIAL state of charge affect safety?

ANM6Easy's documented design intent is that "the agent must strategically plan
ahead to ensure a sufficient charge level at the storage unit" so that the path
bus 0 -> bus 5 is not over-rated during the EV charging peaks. This sweep tests
that directly: vary the initial SoC and watch violations, time-to-first-violation,
and WHICH documented scenario each violation falls in.

Documented periods (each day, 96 x 15-min steps):
  scenario 1  23:00-06:00  windy night, low demand      -> export congestion
  scenario 2  08:00-11:00  EV charging peak             -> import congestion (bus 5)
  scenario 3  13:00-16:00  sunny+windy midday           -> export congestion
  scenario 2  18:00-21:00  EV charging peak again       -> import congestion (bus 5)
  (2-hour linear transitions between periods)

Subclassing does NOT modify ANM6Easy - it creates a new class, so the original
remains available in the same session.

Run:  python soc_sweep.py
      python soc_sweep.py --agent perfect --horizon 32
      python soc_sweep.py --socs 0,10,25,50,75,100
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np

# documented period boundaries, in 15-min steps
PERIODS = [(92, 96, "sc1 night"), (0, 24, "sc1 night"), (24, 32, "transition"),
           (32, 44, "sc2 EV am"), (44, 52, "transition"), (52, 64, "sc3 midday"),
           (64, 72, "transition"), (72, 84, "sc2 EV pm"), (84, 92, "transition")]


def period_of(step):
    for lo, hi, name in PERIODS:
        if lo <= step < hi:
            return name
    return "transition"


def make_env(soc0):
    from gym_anm.envs.anm6_env.anm6_easy import ANM6Easy

    class ANM6SoC(ANM6Easy):
        def __init__(self):
            self._soc0 = soc0
            super().__init__()

        def init_state(self):
            s = super().init_state()
            s[14] = self._soc0          # index 14 = des_soc (MWh, 0-100)
            return s

    return ANM6SoC()


def worst_loading(sim):
    i = sim.state["branch_i_magn"]["pu"]
    return max((abs(float(np.real(c))) / getattr(sim.branches[b], "rate"), b)
               for b, c in i.items() if getattr(sim.branches[b], "rate", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socs", default="0,25,50,75,100")
    ap.add_argument("--agent", default="perfect",
                    choices=["passive", "constant", "perfect"])
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.96)
    ap.add_argument("--steps", type=int, default=96)
    args = ap.parse_args()

    from gym_anm.agents import MPCAgentConstant, MPCAgentPerfect
    from collections import Counter

    socs = [float(x) for x in args.socs.split(",")]
    print(f"agent = {args.agent}" + (f" (N={args.horizon}, margin={args.margin})"
                                      if args.agent != "passive" else ""))
    print(f"{'SoC0 MWh':>9} {'violations':>12} {'1st viol step':>14} "
          f"{'min SoC':>9} {'worst branch':>13}   by period")
    for soc0 in socs:
        env = make_env(soc0)
        hi = env.action_space.high.copy()
        if args.agent == "passive":
            act = lambda: np.concatenate([hi[:2], np.zeros(4)])
        else:
            cls = MPCAgentPerfect if args.agent == "perfect" else MPCAgentConstant
            ag = cls(env.simulator, env.action_space, env.gamma,
                     safety_margin=args.margin, planning_steps=args.horizon)
            act = lambda: ag.act(env)

        obs, _ = env.reset()
        viol, first, socs_seen, branches, by_per = 0, None, [], [], Counter()
        for s in range(args.steps):
            obs, r, te, tr, info = env.step(act())
            sim = env.simulator
            ml, br = worst_loading(sim)
            socs_seen.append(float(np.real(list(sim.state["des_soc"]["MWh"].values())[0])))
            if ml > 1.0:
                viol += 1
                branches.append(br)
                by_per[period_of(s)] += 1
                if first is None:
                    first = s
        wb = Counter(branches).most_common(1)
        print(f"{soc0:>9.0f} {viol:>5}/{args.steps} ({viol/args.steps*100:>4.1f}%) "
              f"{str(first) if first is not None else '-':>14} "
              f"{min(socs_seen):>9.1f} {str(wb[0][0]) if wb else '-':>13}   "
              f"{dict(by_per) if by_per else 'none'}")

    print("\n  scenario 2 (EV peaks, 08-11 and 18-21) is where stored energy is needed:")
    print("  violations there mean the battery had too little charge in time.")


if __name__ == "__main__":
    main()
