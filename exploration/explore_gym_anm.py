"""
explore_gym_anm.py
Kick-the-tyres exploration of gym-anm - the distribution-network counterpart of
explore_grid2op.py. NO safety machinery: just find out how it behaves.

Answers: does it install/run clean? what's in the observation? where are the
SAFETY variables (bus voltage in pu, branch current vs thermal rate) and their
limits? what do simple baseline policies do - do they cause violations? how fast
does it step? is there a renderer?

gym-anm 2.0.1 uses gymnasium >= 1.0, so it coexists with grid2op in one env:
    pip install gym-anm
Run:
    python explore_gym_anm.py
    python explore_gym_anm.py --policy random --episodes 3
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np


def safety_state(sim):
    """Pull ground-truth safety variables out of the simulator."""
    v = sim.state["bus_v_magn"]["pu"]
    i = sim.state["branch_i_magn"]["pu"]
    volts = {k: float(np.real(x)) for k, x in v.items()}
    currs = {k: abs(float(np.real(x))) for k, x in i.items()}
    return volts, currs


def limits(sim):
    vlim, rates = {}, {}
    for bid, bus in sim.buses.items():
        vlim[bid] = (getattr(bus, "v_min", None), getattr(bus, "v_max", None))
    for brid, br in sim.branches.items():
        rates[brid] = getattr(br, "rate", None)
    return vlim, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="ANM6Easy-v0")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--policy", default="nocurt",
                    choices=["nocurt", "zero", "random"],
                    help="nocurt = no curtailment (upper action bound), "
                         "zero = all-zero action, random = sampled")
    args = ap.parse_args()

    import gymnasium, gym_anm
    found = [e for e in gymnasium.registry.keys() if "anm" in e.lower()]
    print("gym-anm envs registered:", found)

    env = gymnasium.make(args.env)
    obs, info = env.reset()
    sim = env.unwrapped.simulator

    print(f"\nENV '{args.env}': {len(sim.buses)} buses, {len(sim.branches)} branches, "
          f"{len(sim.devices)} devices")
    print(f"observation: shape {np.shape(obs)}   action space: {env.action_space}")

    vlim, rates = limits(sim)
    volts, currs = safety_state(sim)
    print("\n--- SAFETY VARIABLES (ground truth from the simulator) ---")
    print("bus voltages (pu)  [bus 0 is the slack bus, pinned at 1.0]:")
    for b, val in volts.items():
        lo, hi = vlim[b]
        print(f"   bus {b}: {val:.4f}   limits [{lo}, {hi}]")
    print("branch currents (pu) vs thermal rate:")
    for br, val in currs.items():
        r = rates[br]
        frac = val / r if r else float("nan")
        print(f"   branch {br}: |i| {val:.4f}   rate {r}   loading {frac*100:5.1f}%")

    # --- baseline policies ---
    hi_act = env.action_space.high.copy()
    lo_act = env.action_space.low.copy()

    def act():
        if args.policy == "nocurt":
            a = hi_act.copy(); a[2:] = 0.0      # max generation allowed, no DES/Q action
            return a
        if args.policy == "zero":
            return np.zeros_like(hi_act)
        return env.action_space.sample()

    print(f"\n--- baseline '{args.policy}', {args.episodes} episodes x {args.steps} steps ---")
    print(f"{'ep':>3} {'steps':>6} {'term':>6} {'Vmin':>7} {'Vmax':>7} "
          f"{'V viol%':>8} {'maxload':>8} {'I viol%':>8}")
    t0 = time.time(); total = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        vmins, vmaxs, loads, vviol, iviol, s = [], [], [], 0, 0, 0
        term = trunc = False
        while not (term or trunc) and s < args.steps:
            obs, r, term, trunc, info = env.step(act())
            volts, currs = safety_state(env.unwrapped.simulator)
            vv = np.array(list(volts.values()))
            # exclude the slack bus (index 0) from violation stats
            vv_n = vv[1:] if len(vv) > 1 else vv
            vmins.append(vv_n.min()); vmaxs.append(vv_n.max())
            ld = max((currs[b] / rates[b]) for b in currs if rates[b])
            loads.append(ld)
            if vv_n.min() < 0.95 or vv_n.max() > 1.05:
                vviol += 1
            if ld > 1.0:
                iviol += 1
            s += 1
        total += s
        print(f"{ep:>3} {s:>6} {str(term or trunc):>6} {min(vmins):>7.4f} "
              f"{max(vmaxs):>7.4f} {vviol/max(s,1)*100:>7.1f}% "
              f"{max(loads):>8.3f} {iviol/max(s,1)*100:>7.1f}%")
    el = time.time() - t0
    print(f"\nspeed: {total/el:.0f} steps/sec ({total} steps in {el:.1f}s)")

    print("\nNotes:")
    print("  - V viol% uses the EN 50160 / ANSI style band 0.95-1.05 pu (our choice,")
    print("    not the env's): that is the regulatory-grounded safety property.")
    print("  - maxload = worst branch current / thermal rate; >1.0 is an overload.")
    print("  - gym-anm renders in a WEB BROWSER: env = gymnasium.make(id,")
    print("    render_mode='human'); env.render()  (it serves a local page).")
    print("  - ANM6Easy is deterministic (fixed 24h profiles) -> a clean baseline;")
    print("    design your own env to inject the distribution shift.")


if __name__ == "__main__":
    main()
