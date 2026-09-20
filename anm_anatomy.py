"""
anm_anatomy.py
Understand what ANM6Easy actually IS and how it responds - no safety case,
no RL, just dissection and poking.

Four sections:
  1. ANATOMY     - devices, branches, action/observation semantics, reward params
  2. DAILY CYCLE - what the loads and generators do over 24h, and when it binds
  3. SENSITIVITY - sweep one action dimension, watch loading and reward respond
  4. REWARD      - decompose cost into energy loss vs constraint-violation penalty

Run:
    python anm_anatomy.py                    # all sections
    python anm_anatomy.py --section sens     # just the sensitivity sweep
    python anm_anatomy.py --sweep_dim 0      # which action dim to sweep
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np

ACTION_MEANING = [
    "a[0] solar P ceiling (curtailment): 0..30  (0 = fully curtailed)",
    "a[1] wind  P ceiling (curtailment): 0..50  (0 = fully curtailed)",
    "a[2] storage P setpoint: -30..30  (<0 charge, >0 discharge)",
    "a[3] solar Q setpoint:  -50..50",
    "a[4] wind  Q setpoint:  -50..50",
    "a[5] storage Q setpoint: -50..50",
]


def safety(sim):
    v = sim.state["bus_v_magn"]["pu"]
    i = sim.state["branch_i_magn"]["pu"]
    volts = np.array([float(np.real(x)) for x in v.values()])[1:]
    load = {b: abs(float(np.real(c))) / getattr(sim.branches[b], "rate")
            for b, c in i.items() if getattr(sim.branches[b], "rate", None)}
    return volts, load


def anatomy(env):
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    sim = u.simulator
    print("=" * 72)
    print("1. ANATOMY")
    print("=" * 72)
    print(f"  control step delta_t = {u.delta_t} h ({u.delta_t*60:.0f} min)  "
          f"| discount gamma = {u.gamma}  | violation weight lambda = {u.lamb}")
    print(f"\n  DEVICES ({len(sim.devices)}):")
    for did, d in sim.devices.items():
        t = type(d).__name__
        extra = ""
        if hasattr(d, "soc_max") and getattr(d, "soc_max") is not None:
            extra = f"  soc=[{d.soc_min},{d.soc_max}] eff={d.eff}"
        print(f"    {did}: {t:14s} bus {getattr(d,'bus_id','?')}  "
              f"P[{getattr(d,'p_min',None)}, {getattr(d,'p_max',None)}]{extra}")
    print(f"\n  BRANCHES ({len(sim.branches)}):  (rate = thermal limit, pu)")
    for bid, b in sim.branches.items():
        print(f"    {bid}: rate {getattr(b,'rate',None)}   "
              f"r={getattr(b,'r',None)}  x={getattr(b,'x',None)}")
    print("\n  ACTION SEMANTICS:")
    for m in ACTION_MEANING:
        print("    " + m)
    print(f"\n  observation: {env.observation_space.shape[0]} values "
          f"(device P/Q, storage SoC, max renewable P, time-of-day aux)")


def daily_cycle(env, steps=96):
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    hi = env.action_space.high.copy()
    print("\n" + "=" * 72)
    print("2. DAILY CYCLE (passive: no curtailment, no storage action)")
    print("=" * 72)
    obs, _ = env.reset(seed=0)
    rows = []
    for s in range(steps):
        a = np.concatenate([hi[:2], np.zeros(4)])
        obs, r, te, tr, info = env.step(a)
        sim = u.simulator
        dp = sim.state["dev_p"]["MW"]
        vals = {k: float(np.real(v)) for k, v in dp.items()}
        dem = -sum(v for v in vals.values() if v < 0)
        gen = sum(v for v in vals.values() if v > 0)
        volts, load = safety(sim)
        worst = max(load, key=load.get)
        rows.append((s, dem, gen, load[worst], worst, r))
    print(f"{'hour':>6} {'demand MW':>10} {'gen MW':>8} {'worst branch':>14} "
          f"{'loading':>8} {'reward':>8}")
    for s, dem, gen, ld, br, r in rows[::6]:
        flag = " <-- OVER" if ld > 1 else ""
        print(f"{s*0.25:>6.1f} {dem:>10.1f} {gen:>8.1f} {str(br):>14} "
              f"{ld:>8.3f} {r:>8.2f}{flag}")
    lds = np.array([r[3] for r in rows])
    print(f"\n  loading over the day: min {lds.min():.2f}  max {lds.max():.2f}  "
          f"over-limit {100*(lds>1).mean():.0f}% of steps")
    from collections import Counter
    c = Counter(r[4] for r in rows if r[3] > 1)
    print(f"  which branch binds when over-limit: {dict(c)}")


def sensitivity(env, dim=0, n=9, at_step=48):
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    lo, hi = env.action_space.low[dim], env.action_space.high[dim]
    base_hi = env.action_space.high.copy()
    print("\n" + "=" * 72)
    print(f"3. SENSITIVITY: sweep {ACTION_MEANING[dim]}")
    print(f"   (measured at step {at_step} = hour {at_step*0.25:.1f})")
    print("=" * 72)
    print(f"{'value':>8} {'maxload':>9} {'Vmin':>8} {'reward':>9}")
    for val in np.linspace(lo, hi, n):
        obs, _ = env.reset(seed=0)
        for s in range(at_step):
            a = np.concatenate([base_hi[:2], np.zeros(4)])
            env.step(a)
        a = np.concatenate([base_hi[:2], np.zeros(4)])
        a[dim] = val
        obs, r, te, tr, info = env.step(a)
        volts, load = safety(u.simulator)
        ml = max(load.values())
        print(f"{val:>8.1f} {ml:>9.3f} {volts.min():>8.4f} {r:>9.2f}")
    print("\n  read this as: how much does this one lever move the constraint?")


def reward_decomp(env, steps=32):
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    hi = env.action_space.high.copy()
    print("\n" + "=" * 72)
    print("4. REWARD DECOMPOSITION  (cost = energy loss + lambda x violation)")
    print("=" * 72)
    print(f"  lambda = {u.lamb}: a violation is weighted {u.lamb}x an energy loss unit,")
    print("  so the agent is strongly pushed to respect the envelope.\n")
    for label, act in [("passive (no curtail)", np.concatenate([hi[:2], np.zeros(4)])),
                       ("full curtailment", np.zeros(6)),
                       ("half curtailment", np.concatenate([hi[:2]/2, np.zeros(4)]))]:
        obs, _ = env.reset(seed=0)
        rs, lds = [], []
        for s in range(steps):
            obs, r, te, tr, info = env.step(act)
            rs.append(r)
            _, load = safety(u.simulator)
            lds.append(max(load.values()))
        print(f"  {label:22s} mean reward {np.mean(rs):>8.2f}   "
              f"max loading {max(lds):.3f}   over-limit {100*np.mean(np.array(lds)>1):.0f}%")
    print("\n  full curtailment protects the network but wastes all renewable output;")
    print("  passive uses it all but breaks the envelope. The controller lives between.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all",
                    choices=["all", "anatomy", "cycle", "sens", "reward"])
    ap.add_argument("--sweep_dim", type=int, default=0)
    ap.add_argument("--at_step", type=int, default=48)
    args = ap.parse_args()

    import gymnasium, gym_anm
    env = gymnasium.make("ANM6Easy-v0")
    if args.section in ("all", "anatomy"):
        anatomy(env)
    if args.section in ("all", "cycle"):
        daily_cycle(env)
    if args.section in ("all", "sens"):
        sensitivity(env, dim=args.sweep_dim, at_step=args.at_step)
    if args.section in ("all", "reward"):
        reward_decomp(env)


if __name__ == "__main__":
    main()
